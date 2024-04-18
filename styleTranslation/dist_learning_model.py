import torch
from torch import nn
from torch.utils.data import DataLoader
from base_distributed.distributed_model import BaseDist
from argparse import Namespace

from cycle_gan_model import Generator, Discriminator
from losses import AdversarialLoss, CycleLoss, IdentityLoss


class DistLearning(BaseDist):
    def __init__(self, rank: int, world_size: int, seed: int,
                 init_args: Namespace, train_data: list[str],
                 val_data: list[str], batch_size: int,
                 epochs: int) -> None:
        super().__init__(rank, world_size, seed, True)

        # Create forward(A) and reverse(B) models
        self.genA = self._ddp_wrapper(Generator().to(self.device))
        self.genB = self._ddp_wrapper(Generator().to(self.device))
        self.discA = self._ddp_wrapper(Discriminator().to(self.device))
        self.discB = self._ddp_wrapper(Discriminator().to(self.device))

        self.modelA = nn.ModuleList([self.genA, self.discA])
        self.modelB = nn.ModuleList([self.genB, self.discB])

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        if init_args.imodel is None:
            # Initialze model weights with Gaussian distribution N(0, 0.2)
            self.start_epoch = 0
            self._init_weights(self.models, mean = 0.0, std = 0.2)
        else:
            self.start_epoch = self.load_model(self.models, init_args.imodel, self.device)
        
        self.epochs = self.start_epoch + epochs
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        
        self.opim_gen = torch.optim.Adam(nn.ParameterList(model.parameters() for model in self.gens),
                                        lr = 0.0002, betas = (0.5, 0.999))
        
        self.opim_disc = torch.optim.Adam(nn.ParameterList(model.parameters() for model in self.discs),
                                        lr = 0.0002, betas = (0.5, 0.999))

        self.adv_loss = torch.colmpile(AdversarialLoss(ltype = 'MSE'))
        self.cycle_loss = torch.compile(CycleLoss('L1'))
        self.idn_loss = torch.compile(IdentityLoss('L1'))

        self.scaler = torch.cuda.amp.GradScaler(enabled = True)

        for model in self.models:
            model.compile()

    def prepare_dataloader(self) -> DataLoader:
        pass
    
    @torch.autocast(device_type = 'cuda', dtype = torch.float16)
    def forward(self, X: torch.Tensor, Y: torch.Tensor, adv_alpha: float = 0.5) -> torch.Tensor:
        fakeY = self.genA(X)
        cycle_fakeX = self.genB(fakeY)
        fakeX = self.genB(Y)
        cycle_fakeY = self.genA(fakeX)

        #self.set_requires_grad(self.discs, False)

        loss = adv_alpha * (self.adv_loss(self.discA(X), True) + self.adv_loss(self.discA(fakeX), True)) + \
            + self.cycle_loss(cycle_fakeX, cycle_fakeY, X, Y) + \
            + self.idn_loss(self.genB(X), self.genA(Y), X, Y)
        
        return loss
    
    def backward_gen(self, loss: torch.Tensor):
        self.gens.zero_grad(True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opim_gen)
        self.scaler.update()

    
    #def execute():
        
