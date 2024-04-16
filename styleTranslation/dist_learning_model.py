import torch
from torch import nn
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

        for model in self.models:
            model.compile()
    
    #def forward():

    
    #def initiate():
        
