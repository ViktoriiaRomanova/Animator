import os
import random
from argparse import Namespace
from warnings import warn

import torch
import torch.distributed as dist
from torch import nn
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vision_aided_loss import Discriminator
from tqdm.auto import tqdm

from animator.base_distributed._distributed_model import BaseDist
from .generator import GANTurboGenerator, get_trainable_params
from .get_dataset import UnpairedDataset
from .losses import CycleLoss, IdentityLoss
from .segmentation import SegmentCharacter
from .metric_storage import DiffusionMetricStorage
from ..utils.buffer import ImageBuffer
from ..utils.parameter_storages.diffusion_parameters import DiffusionTrainingParams


class DiffusionDistLearning:
    def __init__(
        self,
        rank: int,
        init_args: Namespace,
        params: DiffusionTrainingParams,
        train_data: list[list[str], list[str]],
        val_data: list[list[str], list[str]] | None,
    ) -> None:

        self.init_args = init_args
        self.batch_size = params.main.batch_size
        self.epochs = params.main.epochs
        self.save_step = params.main.save_step

        self.metrics = DiffusionMetricStorage(self.rank, 0)

        # Create a folder to store intermediate results at s3 storage (Yandex Object Storage)
        self.s3_storage = None
        if rank == 0 and init_args.st is not None:
            self.s3_storage = os.path.join(init_args.st, os.path.basename(self.model_weights_dir))
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

        self.train_loader = self.prepare_dataloader(
            train_set, rank, self.world_size, self.batch_size, self.random_seed
        )
        self.val_loader = self.prepare_dataloader(
            val_set, rank, self.world_size, self.batch_size, self.random_seed
        )
        cur_model_mean = torch.tensor(params.data.mean)
        cur_model_std = torch.tensor(params.data.std)

        inv_model_std = 1 / cur_model_std
        inv_model_mean = -cur_model_mean * inv_model_std

        self.renorm_for_fid = Normalize(inv_model_mean, inv_model_std, inplace=False)

        # Create forward(A) and reverse(B) models
        self.genA = GANTurboGenerator(params.main.caption_forward, params.generator, self.device[0])
        self.genB = GANTurboGenerator(params.main.caption_reverse, params.generator, self.device[1])

        self.discA = Discriminator(
            params.discriminator.cv_type, loss_type=params.discriminator.loss_type, device=self.device[0]
        )

        self.discB = Discriminator(
            params.discriminator.cv_type, loss_type=params.discriminator.loss_type, device=self.device[1]
        )

        self.modifier = None
        if params.main.segmentation_model is not None:
            self.modifier = SegmentCharacter(
                params.main.segmentation_model,
                params.main.segmentation_model_type,
                device=self.device[0],
                mean=params.data.mean,
                std=params.data.std,
                warm_up=params.main.warm_up,
            )
        else:
            self.modifier = None
            warn("The segmentation model isn't provided, the segmentation part will be skiped")

        self.genA = self._ddp_wrapper(
            self.genA.to(self.device[0]), is_multy_gpu=True, broadcast_buffers=False
        )
        self.genB = self._ddp_wrapper(
            self.genB.to(self.device[1]), is_multy_gpu=True, broadcast_buffers=False
        )
        self.discA = self._ddp_wrapper(
            self.discA.to(self.device[0]), is_multy_gpu=True, broadcast_buffers=False
        )
        self.discB = self._ddp_wrapper(
            self.discB.to(self.device[1]), is_multy_gpu=True, broadcast_buffers=False
        )

        self.modelA = nn.ModuleList([self.genA, self.discA])
        self.modelB = nn.ModuleList([self.genB, self.discB])

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        self.gens_trainable_params = get_trainable_params(self.gens, True)

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.scaler = torch.amp.GradScaler(enabled=self.device[0].type == "cuda")

        self.fake_Y_buffer = ImageBuffer(self.world_size, params.main.buffer_size)
        self.fake_X_buffer = ImageBuffer(self.world_size, params.main.buffer_size)

        self.optim_gen = torch.optim.AdamW(
            self.gens_trainable_params,
            lr=params.optimizers.gen.lr,
            betas=params.optimizers.gen.betas,
            weight_decay=params.optimizers.gen.weight_decay,
        )
        self.optim_discA = torch.optim.AdamW(
            self.discA.parameters(),
            lr=params.optimizers.discA.lr,
            betas=params.optimizers.discA.betas,
            weight_decay=params.optimizers.discA.weight_decay,
        )
        self.optim_discB = torch.optim.AdamW(
            self.discB.parameters(),
            lr=params.optimizers.discB.lr,
            betas=params.optimizers.discB.betas,
            weight_decay=params.optimizers.discB.weight_decay,
        )

        def lambda_rule(epoch: int) -> float:
            return 1.0

        self.scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.optim_gen, lr_lambda=lambda_rule)
        self.scheduler_discA = torch.optim.lr_scheduler.LambdaLR(self.optim_discA, lr_lambda=lambda_rule)
        self.scheduler_discB = torch.optim.lr_scheduler.LambdaLR(self.optim_discB, lr_lambda=lambda_rule)

        self.save_load_params = {
            "genA": self.genA,
            "discA": self.discA,
            "genB": self.genB,
            "discB": self.discB,
            "optim_gen": self.optim_gen,
            "optim_discA": self.optim_discA,
            "optim_discB": self.optim_discB,
            "scaler": self.scaler,
            "scheduler_gen": self.scheduler_gen,
            "scheduler_discA": self.scheduler_discA,
            "scheduler_discB": self.scheduler_discB,
            "bufferX": self.fake_X_buffer,
            "bufferY": self.fake_Y_buffer,
        }

        self.start_epoch = 0
        if init_args.imodel is not None:
            self.start_epoch = self.load_model(init_args.imodel, self.device)
            self.modifier.warm_up_update(-self.start_epoch * self.batch_size)

        self.epochs += self.start_epoch

        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", "mean", sync_on_compute=False).to(
            self.device[0]
        )
        self.lpips.compile()

        self.adv_alpha = params.loss.adversarial.adv_alpha
        self.cycle_loss = CycleLoss(
            self.lpips,
            params.loss.cycle.ltype,
            params.loss.cycle.lambda_A,
            params.loss.cycle.lambda_B,
            params.loss.cycle.lambda_lpips,
            self.device[0],
        )
        self.idn_loss = IdentityLoss(
            self.lpips,
            params.loss.identity.ltype,
            params.loss.identity.lambda_idn,
            params.loss.identity.lambda_lpips,
            self.device[0],
        )

        # to get different (from previous use) random numbers after loading the model
        random.seed(rank + self.start_epoch)

        #for model in self.models:
            #model.compile()

    def save_model(self, epoch: int) -> dict:
        state = {}
        for key, param in self.save_load_params.items():
            if key == "genA" or key == "genB":
                # Save only LoRa parameters
                generator_state = {}
                original_state_dict = param.module.state_dict()
                for name in original_state_dict:
                    if name.find("lora") != -1 or name.find("modules_to_save") != -1:
                        generator_state[name] = original_state_dict[name]
                state[key] = generator_state
            elif isinstance(param, nn.Module):
                state[key] = param.module.state_dict()
            else:
                state[key] = param.state_dict()
        state["epoch"] = epoch
        return state

    def load_model(self, path: str, device: torch.device) -> int:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, path)
        state = torch.load(weights_dir, map_location=device)

        for key, param in self.save_load_params.items():
            if key not in state:
                warn("Loaded state dict doesn`t contain {} its loading omitted".format(key))
                continue
            if key == "genA" or key == "genB":
                # Load only LoRa parameters
                remains = param.module.load_state_dict(state[key], strict=False)
                if len(remains.unexpected_keys) > 0:
                    warn("Some parameters weren't loaded {}".format(remains.unexpected_keys))
            elif isinstance(param, nn.Module):
                param.module.load_state_dict(state[key])
            else:
                param.load_state_dict(state[key])

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
            lossB_true = self.discB(X.to(device=self.device[1]), for_real=True).mean()* adv_alpha

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
