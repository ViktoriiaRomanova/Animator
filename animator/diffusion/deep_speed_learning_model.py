from datetime import datetime
import os
import random
from argparse import Namespace
from warnings import warn

import deepspeed
import torch
from torch import nn
from torchrl.data import ReplayBuffer, ListStorage
from torchvision.transforms import Normalize
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vision_aided_loss import Discriminator
from tqdm.auto import tqdm

from .generator import GANTurboGenerator, get_trainable_params
from .get_dataset import UnpairedDataset
from .losses import CycleLoss, IdentityLoss
from .segmentation import SegmentCharacter
from .metric_storage import DiffusionMetricStorage
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

        self.renorm_for_fid = Normalize(inv_model_mean, inv_model_std, inplace=False).to(self.device)

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

        self.genA, self.optim_genA, self.train_loader, _ = deepspeed.initialize(
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

        self.batch_size = self.train_loader.batch_size

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        # self.gens_trainable_params = get_trainable_params(self.gens, True)

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.fake_Y_buffer = ReplayBuffer(
            storage=ListStorage(params.main.buffer_size * self.batch_size),
            batch_size=self.batch_size,
            pin_memory=True,
            prefetch=2,
        )
        self.fake_X_buffer = ReplayBuffer(
            storage=ListStorage(params.main.buffer_size * self.batch_size),
            batch_size=self.batch_size,
            pin_memory=True,
            prefetch=2,
        )

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
        #self.lpips.compile()

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
        if (epoch + 1) % self.save_step != 0:
            return
        state = {}
        state["epoch"] = epoch
        #self.fake_X_buffer.dumps(self.s3_storage)
        #self.fake_Y_buffer.dumps(self.s3_storage)
        state["X_storage"] = self.fake_X_buffer.state_dict()["_storage"]
        state["Y_storage"] = self.fake_Y_buffer.state_dict()["_storage"]
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

        self.genB.load_checkpoint(path, "epoch_{}_{}".format(version, "genB"), load_optimizer_states=True)

        self.discA.load_checkpoint(path, "epoch_{}_{}".format(version, "discA"), load_optimizer_states=True)

        self.discB.load_checkpoint(path, "epoch_{}_{}".format(version, "discB"), load_optimizer_states=True)

        #self.fake_X_buffer.loads(self.s3_storage)
        #self.fake_Y_buffer.loads(self.s3_storage)

        self.fake_X_buffer.load_state_dict({"_storage": state["X_storage"]})
        self.fake_Y_buffer.load_state_dict({"_storage": state["Y_storage"]})

        return state["epoch"] + 1

    def forward_gen(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        self.discs.requires_grad_(False)

        fakeY_unmodif = self.genA(X)
        fakeY = self.modifier(fakeY_unmodif)
        cycle_fakeX = self.genB(fakeY_unmodif)

        fakeX_unmodif = self.genB(Y)
        fakeX = self.modifier(fakeX_unmodif)
        cycle_fakeY = self.genA(fakeX_unmodif)

        self.fake_X_buffer.extend(fakeX.detach().clone().to("cpu"))
        self.fake_Y_buffer.extend(fakeY.detach().clone().to("cpu"))

        adv_lossA = self.discA(fakeY, for_G=True).mean()
        adv_lossB = self.discB(fakeX, for_G=True).mean()

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

        self.discs.requires_grad_(True)

        lossA_false = (
            self.discA(self.fake_Y_buffer.sample().to(self.device), for_real=False).mean() * adv_alpha
        )
        lossB_false = (
            self.discB(self.fake_X_buffer.sample().to(self.device), for_real=False).mean() * adv_alpha
        )

        lossA_true = self.discA(Y, for_real=True).mean() * adv_alpha
        lossB_true = self.discB(X, for_real=True).mean() * adv_alpha

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
        loss.backward()
        self.genA._backward_epilogue()
        self.genB._backward_epilogue()
        self.genA.step()
        self.genB.step()
        self.genA.optimizer.zero_grad()
        self.genB.optimizer.zero_grad()

    def backward_disc(self, lossA: torch.Tensor, lossB: torch.Tensor) -> None:
        self.discs.zero_grad(True)
        self.discA.backward(lossA)
        self.discB.backward(lossB)
        self.discA.step()
        self.discB.step()

    def execute(
        self,
    ) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            self.train_loader.data_sampler.set_epoch(epoch)

            self.models.train()

            for x_batch, y_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.forward_gen(x_batch, y_batch)
                self.backward_gen(loss)
                loss_disc_A, loss_disc_B = self.forward_disc(
                    self.modifier(x_batch), self.modifier(y_batch), self.adv_alpha
                )
                self.backward_disc(loss_disc_A, loss_disc_B)

                del x_batch, y_batch, loss, loss_disc_A, loss_disc_B

            # Calculate FID
            self.gens.eval()
            for x_batch, y_batch in tqdm(self.val_loader):
                with torch.no_grad():
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    fakeY = self.genA(x_batch)
                    fakeX = self.genB(y_batch)
                    self.metrics.update("FID", "Forward", fakeY, self.renorm_for_fid(y_batch))
                    self.metrics.update("FID", "Backward", fakeX, self.renorm_for_fid(x_batch))
            
            self.metrics.epoch = epoch

            # Send metrics into stdout. This channel going to be transferred into initial machine.
            self.metrics.compute()
            self.metrics.reset()

            # Save model checkpoint every "save_step"(hyperparameters -> main)
            self.save_model(epoch)
