import os
import random
from argparse import Namespace
from warnings import warn

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vision_aided_loss import Discriminator

from animator.base_distributed._distributed_model import BaseDist
from .generator import GANTurboGenerator
from .get_dataset import UnpairedDataset
from .losses import CycleLoss, IdentityLoss
from .metric_storage import DiffusionMetricGroup
from ..figure_extraction.unet_model import UNet
from ..utils.buffer import ImageBuffer
from ..utils.parameter_storages.diffusion_parameters import DiffusionTrainingParams


class DiffusionDistLearning(BaseDist):
    def __init__(
        self,
        rank: int,
        init_args: Namespace,
        params: DiffusionTrainingParams,
        train_data: list[list[str], list[str]],
        val_data: list[list[str], list[str]] | None,
    ) -> None:
        super().__init__(rank, params.distributed, params.main.random_state)

        self.init_args = init_args
        self.batch_size = params.main.batch_size
        self.epochs = params.main.epochs

        self.metrics = DiffusionMetricGroup(self.rank, 0)

        # Create a folder to store intermediate results at s3 storage (Yandex Object Storage)
        self.s3_storage = None
        if rank == 0 and init_args.st is not None:
            self.s3_storage = os.path.join(init_args.st, os.path.basename(self.model_weights_dir))
            if not os.path.exists(self.s3_storage):
                os.makedirs(self.s3_storage)

        train_set = UnpairedDataset(
            init_args.dataset, train_data, size=params.data.size, mean=params.data.mean, std=params.data.std
        )

        self.train_loader = self.prepare_dataloader(
            train_set, rank, self.world_size, self.batch_size, self.random_seed
        )

        # Create forward(A) and reverse(B) models
        self.genA = GANTurboGenerator(params.main.caption_forward, params.generator, self.device)
        self.genB = GANTurboGenerator(params.main.caption_reverse, params.generator, self.device)

        self.discA = Discriminator(
            params.discriminator.cv_type, params.discriminator.loss_type, device=self.device
        )
        self.discB = Discriminator(
            params.discriminator.cv_type, params.discriminator.loss_type, device=self.device
        )

        self.modifier = None
        if params.main.segmentation_model is not None:
            self.modifier = UNet(params.main.segmentation_model_type).to(self.device)
            state = torch.load(params.main.segmentation_model, map_location=self.device, weights_only=True)[
                "model"
            ]
            self.modifier.load_state_dict(state)
            self.modifier.eval()
            self.modifier.requires_grad_(False)
        else:
            self.modifier = None
            warn("The segmentation model isn't provided, the segmentation part will be skiped")

        self.genA = self._ddp_wrapper(self.genA.to(self.device))
        self.genB = self._ddp_wrapper(self.genB.to(self.device))
        self.discA = self._ddp_wrapper(self.discA.to(self.device))
        self.discB = self._ddp_wrapper(self.discB.to(self.device))

        self.modelA = nn.ModuleList([self.genA, self.discA])
        self.modelB = nn.ModuleList([self.genB, self.discB])

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        self.fake_Y_buffer = ImageBuffer(self.world_size, params.main.buffer_size)
        self.fake_X_buffer = ImageBuffer(self.world_size, params.main.buffer_size)

        self.optim_gen = torch.optim.AdamW(
            self.gens.parameters(),
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

        self.epochs += self.start_epoch

        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", "mean", sync_on_compute=False).to(
            self.device
        )

        self.cycle_loss = CycleLoss(
            self.lpips,
            params.loss.cycle.ltype,
            params.loss.cycle.lambda_A,
            params.loss.cycle.lambda_B,
            params.loss.cycle.lambda_lpips,
        )
        self.idn_loss = IdentityLoss(
            self.lpips,
            params.loss.identity.ltype,
            params.loss.identity.lambda_idn,
            params.loss.identity.lambda_lpips,
        )

        # to get different (from previous use) random numbers after loading the model
        random.seed(rank + self.start_epoch)

        for model in self.models:
            model.compile()

    def save_model(self, epoch: int) -> dict:
        state = {}
        for key, param in self.save_load_params.items():
            if isinstance(param, GANTurboGenerator):
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
            if isinstance(param, GANTurboGenerator):
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
            pin_memory=self.device.type != "cpu",
            num_workers=2,
            prefetch_factor=32,
            multiprocessing_context="spawn",
            persistent_workers=self.device.type != "cpu",
            pin_memory_device=self.device.type if self.device.type != "cpu" else "",
        )
        return data_loader
