import datetime
import os
import random
from argparse import Namespace
from warnings import warn

import deepspeed
import torch
import torch.distributed as dist
from torch import nn
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vision_aided_loss import Discriminator
from tqdm.auto import tqdm

from .generator import GANTurboGenerator, get_trainable_params
from .get_dataset import UnpairedDataset
from .losses import CycleLoss, IdentityLoss
from .segmentation import SegmentCharacter
from .metric_storage import DiffusionMetricStorage
from ..utils.buffer import ImageBuffer
from ..utils.parameter_storages.diffusion_parameters import DiffusionTrainingParams


class DiffusionLearning:
    def __init__(
        self,
        init_args: Namespace,
        params: DiffusionTrainingParams,
        train_data: list[list[str], list[str]],
        val_data: list[list[str], list[str]] | None,
    ) -> None:

        self.init_args = init_args
        self.epochs = params.main.epochs
        self.save_step = params.main.save_step
        self.rank = self.init_args.local_rank
        self.device = torch.device(self.rank)

        self.metrics = DiffusionMetricStorage(self.rank, 0)

        # Create a folder to store intermediate results at s3 storage (Yandex Object Storage)
        assert init_args.st is not None, "In this pipline s3 is the only way to save model state"
        if self.rank == 0:
            self.s3_storage = os.path.join(init_args.st, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            if not os.path.exists(self.s3_storage):
                os.makedirs(self.s3_storage)

        train_set = UnpairedDataset(
            init_args.dataset, train_data, size=params.data.size, mean=params.data.mean, std=params.data.std
        )
        val_set = UnpairedDataset(
            init_args.dataset,
            val_data,
            size=params.data.size,
            mean=params.data.mean,
            std=params.data.std,
            for_train=False,
        )

        cur_model_mean = torch.tensor(params.data.mean)
        cur_model_std = torch.tensor(params.data.std)

        inv_model_std = 1 / cur_model_std
        inv_model_mean = -cur_model_mean * inv_model_std

        self.renorm_for_fid = Normalize(inv_model_mean, inv_model_std, inplace=False)

        # Create forward(A) and reverse(B) models
        self.genA = GANTurboGenerator(params.main.caption_forward, params.generator)
        self.genB = GANTurboGenerator(params.main.caption_reverse, params.generator)

        self.discA = Discriminator(params.discriminator.cv_type, loss_type=params.discriminator.loss_type)

        self.discB = Discriminator(params.discriminator.cv_type, loss_type=params.discriminator.loss_type)

        # To allow Deep Speed correctly cast the discriminator model on the device
        self.discA.cv_ensemble.models = nn.ModuleList(self.discA.cv_ensemble.models)
        self.discB.cv_ensemble.models = nn.ModuleList(self.discB.cv_ensemble.models)

        self.modifier = None
        if params.main.segmentation_model is not None:
            self.modifier = SegmentCharacter(
                params.main.segmentation_model,
                params.main.segmentation_model_type,
                device=self.device,
                mean=params.data.mean,
                std=params.data.std,
                warm_up=params.main.warm_up,
            )
        else:
            self.modifier = None
            warn("The segmentation model isn't provided, the segmentation part will be skiped")

        self.genA, self.optim_genA, self.data_loader, _ = deepspeed.initialize(
            model=self.genA, training_data=train_set, config=self.init_args.ds_config
        )
        self.genB, self.optim_genB, _, _ = deepspeed.initialize(
            model=self.genB, config=self.init_args.ds_config
        )
        self.discA, self.optim_discA, self.val_loader, _ = deepspeed.initialize(
            model=self.discA, training_data=val_set, config=self.init_args.ds_config_disc
        )
        self.discB, self.optim_discB, _, _ = deepspeed.initialize(
            model=self.discB, config=self.init_args.ds_config_disc
        )
        # self.modelA = nn.ModuleList([self.genA, self.discA])
        # self.modelB = nn.ModuleList([self.genB, self.discB])

        # self.gens = nn.ModuleList([self.genA, self.genB])
        # self.discs = nn.ModuleList([self.discA, self.discB])

        # self.gens_trainable_params = get_trainable_params(self.gens, True)

        # self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.fake_Y_buffer = ImageBuffer(self.world_size, params.main.buffer_size)
        self.fake_X_buffer = ImageBuffer(self.world_size, params.main.buffer_size)

        self.start_epoch = 0
        if init_args.imodel is not None:
            assert (
                init_args.imodel.find("restart_from_epoch:") != -1
            ), "Wrong format of path to the loading model"

            path_to_model, version = init_args.imodel.split(
                "restart_from_epoch:"
            )  # Path to the model should be set in the format path/reastart_from_epoch:some_number
            version = int(version)
            self.start_epoch = self.load_model(path_to_model, version)
            self.modifier.warm_up_update(-self.start_epoch * self.batch_size)

        self.epochs += self.start_epoch

        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", "mean", sync_on_compute=False).to(
            self.device
        )
        self.lpips.compile()

        self.adv_alpha = params.loss.adversarial.adv_alpha
        self.cycle_loss = CycleLoss(
            self.lpips,
            params.loss.cycle.ltype,
            params.loss.cycle.lambda_A,
            params.loss.cycle.lambda_B,
            params.loss.cycle.lambda_lpips,
            self.device,
        )
        self.idn_loss = IdentityLoss(
            self.lpips,
            params.loss.identity.ltype,
            params.loss.identity.lambda_idn,
            params.loss.identity.lambda_lpips,
            self.device,
        )

        # to get different (from previous use) random numbers after loading the model
        random.seed(self.rank + self.start_epoch)

    def save_model(self, epoch: int) -> None:
        state = {}
        state["epoch"] = epoch
        state["bufferX"] = self.fake_X_buffer.state_dict()
        state["bufferY"] = self.fake_Y_buffer.state_dict()
        self.genA.save_checkpoint(
            self.s3_storage,
            tag="epoch_{}_{}".format(str(epoch), "genA"),
            client_state=state,
            exclude_frozen_parameters=True,
        )
        self.genB.save_checkpoint(
            self.s3_storage, tag="epoch_{}_{}".format(str(epoch), "genB"), exclude_frozen_parameters=True
        )
        self.discA.save_checkpoint(
            self.s3_storage, tag="epoch_{}_{}".format(str(epoch), "discA"), exclude_frozen_parameters=True
        )
        self.discB.save_checkpoint(
            self.s3_storage, tag="epoch_{}_{}".format(str(epoch), "discB"), exclude_frozen_parameters=True
        )

    def load_model(self, path: str, version: int) -> int:
        _, state = self.genA.load_checkpoint(
            path, "epoch_{}_{}".format(version, "genA"), load_optimizer_states=True
        )

        self.genB.load_checkpoint(
            path, "epoch_{}_{}".format(version, "genB"), load_optimizer_states=True
        )

        self.discA.load_checkpoint(
            path, "epoch_{}_{}".format(version, "discA"), load_optimizer_states=True
        )

        self.discB.load_checkpoint(
            path, "epoch_{}_{}".format(version, "discB"), load_optimizer_states=True
        )

        self.fake_X_buffer.load_state_dict(state["bufferX"])
        self.fake_Y_buffer.load_state_dict(state["bufferY"])

        return state["epoch"] + 1

    def prepare_dataloader(
        self, data: UnpairedDataset, rank: int, world_size: int, batch_size: int, seed: int
    ) -> DataLoader:
        """
        Split dataset into N parts.

        Returns: DataLoader instance for current part.
        """
        sampler = DistributedSampler(
            data, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
        )
        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
            pin_memory=self.device[0].type != "cpu",
            num_workers=16,
            prefetch_factor=16,
            multiprocessing_context="spawn",
            persistent_workers=self.device[0].type != "cpu",
            pin_memory_device=self.device[0].type if self.device[0].type != "cpu" else "",
        )
        return data_loader

    def forward_gen(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        with torch.autocast(
            device_type=self.device[0].type, dtype=torch.float16, enabled=self.device[0].type == "cuda"
        ):
            self.discs.requires_grad_(False)

            fakeY_unmodif = self.genA(X)
            fakeY = self.modifier(fakeY_unmodif)
            cycle_fakeX = self.genB(fakeY_unmodif)

            fakeX_unmodif = self.genB(Y)
            fakeX = self.modifier(fakeX_unmodif)
            cycle_fakeY = self.genA(fakeX_unmodif)

            self.fake_X_buffer.add(fakeX.detach().clone())
            self.fake_Y_buffer.add(fakeY.detach().clone())

            adv_lossA = self.discA(fakeY, for_G=True).mean()
            adv_lossB = self.discB(fakeX, for_G=True).mean().to(self.device[0])

            cycle_loss = self.cycle_loss(cycle_fakeX, cycle_fakeY, X, Y)

            idn_loss = self.idn_loss(self.genB(X), self.genA(Y), X, Y)

            loss = adv_lossA + adv_lossB + cycle_loss + idn_loss

            self.metrics.update("Total_loss", "gens", loss.detach().clone())
            self.metrics.update("Adv_gen", "discA", adv_lossA.detach().clone())
            self.metrics.update("Adv_gen", "discB", adv_lossB.detach().clone())
            self.metrics.update("Cycle", "", cycle_loss.detach().clone())
            self.metrics.update("Identity", "", idn_loss.detach().clone())

        return loss

    def forward_disc(
        self, X: torch.Tensor, Y: torch.Tensor, adv_alpha: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.autocast(
            device_type=self.device[0].type, dtype=torch.float16, enabled=self.device[0].type == "cuda"
        ):
            self.discs.requires_grad_(True)

            lossA_false = self.discA(self.fake_Y_buffer.get(), for_real=False).mean() * adv_alpha
            lossB_false = self.discB(self.fake_X_buffer.get(), for_real=False).mean() * adv_alpha

            lossA_true = self.discA(Y.to(device=self.device[0]), for_real=True).mean() * adv_alpha
            lossB_true = self.discB(X.to(device=self.device[1]), for_real=True).mean() * adv_alpha

            lossA = lossA_true + lossA_false
            lossB = lossB_true + lossB_false

            self.metrics.update("Total_loss", "disc_A", lossA.detach().clone())
            self.metrics.update("Total_loss", "disc_B", lossB.detach().clone())
            self.metrics.update("Adv_discA", "True", lossA_true.detach().clone())
            self.metrics.update("Adv_discA", "False", lossA_false.detach().clone())
            self.metrics.update("Adv_discB", "True", lossB_true.detach().clone())
            self.metrics.update("Adv_discB", "False", lossB_false.detach().clone())

        return lossA, lossB

    def backward_gen(self, loss: torch.Tensor) -> None:
        self.gens.zero_grad(True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim_gen)
        nn.utils.clip_grad_norm_(self.gens_trainable_params, 10)
        self.scaler.step(self.optim_gen)

    def backward_disc(self, lossA: torch.Tensor, lossB: torch.Tensor) -> None:
        self.discs.zero_grad(True)
        self.scaler.scale(lossA).backward()
        self.scaler.scale(lossB).backward()
        self.scaler.unscale_(self.optim_discA)
        self.scaler.unscale_(self.optim_discB)
        nn.utils.clip_grad_norm_(self.discs.parameters(), 10)
        self.scaler.step(self.optim_discA)
        self.scaler.step(self.optim_discB)

        # Update scaler after last "step"
        self.scaler.update()

    def execute(
        self,
    ) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            self.train_loader.sampler.set_epoch(epoch)

            self.models.train()

            for x_batch, y_batch in tqdm(self.train_loader):
                loss = self.forward_gen(x_batch, y_batch)
                self.backward_gen(loss)
                loss_disc_A, loss_disc_B = self.forward_disc(
                    self.modifier(x_batch), self.modifier(y_batch), self.adv_alpha
                )
                self.backward_disc(loss_disc_A, loss_disc_B)

                self.fake_X_buffer.step()
                self.fake_Y_buffer.step()

                del x_batch, y_batch, loss, loss_disc_A, loss_disc_B

            # Calculate FID
            self.gens.eval()
            for x_batch, y_batch in tqdm(self.val_loader):
                with torch.no_grad():
                    x_batch = x_batch.to(self.device[0], non_blocking=True)
                    y_batch = y_batch.to(self.device[1], non_blocking=True)
                    fakeY = self.genA(x_batch)
                    fakeX = self.genB(y_batch)
                    self.renorm_for_fid.to(y_batch.device)
                    self.metrics.update("FID", "Forward", fakeY, self.renorm_for_fid(y_batch))
                    self.renorm_for_fid.to(x_batch.device)
                    self.metrics.update("FID", "Backward", fakeX, self.renorm_for_fid(x_batch))
            self.gens.train()

            self.metrics.update(
                "Lr", "gens", torch.tensor(self.scheduler_gen.get_last_lr()[0], device=self.device[0])
            )
            self.metrics.update(
                "Lr", "disc_A", torch.tensor(self.scheduler_discA.get_last_lr()[0], device=self.device[0])
            )
            self.metrics.update(
                "Lr", "disc_B", torch.tensor(self.scheduler_discB.get_last_lr()[0], device=self.device[0])
            )
            self.metrics.epoch = epoch

            self.scheduler_gen.step()
            self.scheduler_discA.step()
            self.scheduler_discB.step()

            # Send metrics into stdout. This channel going to be transferred into initial machine.
            self.metrics.compute()
            self.metrics.reset()

            if self.rank == 0:
                if (epoch + 1) % self.save_step == 0:
                    if self.s3_storage is not None:
                        # Save model weights at S3 storage if the path to a bucket provided
                        torch.save(self.save_model(epoch), os.path.join(self.s3_storage, str(epoch) + ".pt"))
                    else:
                        # Otherwise, save at a remote machine
                        warn(
                            " ".join(
                                (
                                    "Intermediate model weights are saved at the remote machine",
                                    "and will be lost after the end of the training process",
                                )
                            )
                        )
                        torch.save(
                            self.save_model(epoch), os.path.join(self.model_weights_dir, str(epoch) + ".pt")
                        )
        if self.rank == 0 and self.s3_storage is not None:
            # Save final results at s3 storage
            torch.save(
                self.save_model(self.epochs - 1), os.path.join(self.s3_storage, str(self.epochs - 1) + ".pt")
            )
        dist.destroy_process_group()
