import os
import random
from argparse import Namespace

import torch
from torch import nn
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

        self.modifier = UNet("B").to(self.device)
        state = torch.load(init_args.imodel, map_location=self.device, weights_only=True)["model"]
        self.modifier.load_state_dict(state)
        self.modifier.eval()
        self.modifier.requires_grad_(False)

        self.genA = self._ddp_wrapper(self.genA.to(self.device))
        self.genB = self._ddp_wrapper(self.genB.to(self.device))
        self.discA = self._ddp_wrapper(self.discA.to(self.device))
        self.discB = self._ddp_wrapper(self.discB.to(self.device))
        self.modifier = self._ddp_wrapper(self.modifier)

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

        self.start_epoch = 0
        # TODO add model loading

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
