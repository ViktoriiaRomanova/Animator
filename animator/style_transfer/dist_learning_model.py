from argparse import Namespace
from tqdm import tqdm
import os
from json import dumps as jdumps

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from animator.base_distributed._distributed_model import BaseDist
from .cycle_gan_model import Generator, Discriminator
from .losses import AdversarialLoss, CycleLoss, IdentityLoss
from ..utils.buffer import ImageBuffer
from .preprocessing_data import GetDataset
from ..utils.parameter_storages import TrainingParams

class DistLearning(BaseDist):
    def __init__(self, rank: int, init_args: Namespace, 
                 params: TrainingParams,
                 train_data: list[str], val_data: list[str] | None,
                ) -> None:
        super().__init__(rank, params.distributed, params.main.random_state)

        self.init_args = init_args
        self.batch_size = params.main.batch_size
        self.epochs = params.main.epochs

        train_set = GetDataset(init_args.dataset, train_data,
                               size = params.data.size,
                               mean = params.data.mean,
                               std = params.data.std)
        self.train_loader = self.prepare_dataloader(train_set, rank,
                                                    self.world_size, self.batch_size, 
                                                    self.random_seed)

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

        self.scaler = torch.cuda.amp.GradScaler(enabled = True)

        self.fake_Y_buffer = ImageBuffer(params.main.buffer_size)
        self.fake_X_buffer = ImageBuffer(params.main.buffer_size)

        self.optim_gen = torch.optim.Adam(self.gens.parameters(),
                                          lr = params.optimizers.gen.lr,
                                          betas = params.optimizers.gen.betas)        
        self.optim_discA = torch.optim.Adam(self.discA.parameters(),
                                            lr = params.optimizers.discA.lr,
                                            betas = params.optimizers.discA.betas)        
        self.optim_discB = torch.optim.Adam(self.discB.parameters(),
                                            lr = params.optimizers.discB.lr,
                                            betas = params.optimizers.discB.betas)
        
        if init_args.imodel is None:
            # Initialze model weights with Gaussian distribution
            self.start_epoch = 0
            self._init_weights(self.models, mean = params.models.mean,
                               std = params.models.std)
        else:
            self.start_epoch = self.load_model(init_args.imodel, self.device)
        
        self.epochs += self.start_epoch

        self.adv_loss = torch.compile(AdversarialLoss(params.loss.adversarial.ltype,
                                                       params.loss.adversarial.real_val,
                                                       params.loss.adversarial.fake_val))
        self.cycle_loss = torch.compile(CycleLoss(params.loss.cycle.ltype,
                                                  params.loss.cycle.lambda_A,
                                                  params.loss.cycle.lambda_B))
        self.idn_loss = torch.compile(IdentityLoss(params.loss.identity.ltype,
                                                   params.loss.cycle.lambda_idn))

        for model in self.models:
            model.compile()

    def load_model(self, path: str, device: torch.device) -> int:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, path)
        state = torch.load(weights_dir, map_location = device)
        self.models.load_state_dict(state['models'])
        self.optim_gen.load_state_dict(state['optim_gen'])
        self.optim_discA.load_state_dict(state['optim_discA'])
        self.optim_discB.load_state_dict(state['optim_discB'])
        self.scaler.load_state_dict(state['staler'])
        return state['epoch'] + 1

    def prepare_dataloader(self, data: GetDataset, rank: int,
                    world_size: int, batch_size: int,
                    seed: int) -> DataLoader:
        """
            Split dataset into N parts.

            Returns: DataLoader instance for current part.
        """
        sampler = DistributedSampler(data, num_replicas = world_size,
                                    rank = rank, shuffle = True,
                                    seed = seed, drop_last = True)
        data_loader = DataLoader(data, batch_size = batch_size,
                                shuffle = False, drop_last = True,
                                sampler = sampler, pin_memory = True,
                                num_workers = 1,
                                prefetch_factor = 16,
                                multiprocessing_context = 'spawn',
                                persistent_workers = True,
                                pin_memory_device = self.device.type)
        return data_loader
    
    def forward_gen(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        with torch.autocast(device_type = self.device.type, dtype = torch.float16):
            self.set_requires_grad(self.discs, False)

            fakeY = self.genA(X)
            cycle_fakeX = self.genB(fakeY)
            fakeX = self.genB(Y)
            cycle_fakeY = self.genA(fakeX)

            self.fake_X_buffer.add(fakeX.detach())
            self.fake_Y_buffer.add(fakeY.detuch())

            loss = self.adv_loss(self.discB(fakeX), True) + self.adv_loss(self.discA(fakeY), True) + \
                + self.cycle_loss(cycle_fakeX, cycle_fakeY, X, Y) + \
                + self.idn_loss(self.genB(X), self.genA(Y), X, Y)
        
        return loss

    def forward_disc(self, X: torch.Tensor, Y: torch.Tensor,
                     adv_alpha: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.autocast(device_type = self.device.type, dtype = torch.float16):
            self.set_requires_grad(self.discs, True)

            ans_disc_A = self.discA(self.fake_Y_buffer.get())
            ans_disc_B = self.discB(self.fake_X_buffer.get()) 

            lossA = adv_alpha * (self.adv_loss(ans_disc_A, False) + \
                + self.adv_loss(Y, True))

            lossB = adv_alpha * (self.adv_loss(ans_disc_B, False) + \
                + self.adv_loss(X, True))
        
            del ans_disc_A, ans_disc_B
        return lossA, lossB

    
    def backward_gen(self, loss: torch.Tensor) -> None:
        self.gens.zero_grad(True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim_gen)
    
    def backward_disc(self, lossA: torch.Tensor, lossB: torch.Tensor) -> None:

        self.discs.zero_grad(True)

        self.scaler.scale(lossA).backward()
        self.scaler.scale(lossB).backward()

        self.scaler.step(self.opitm_discA)
        self.scaler.step(self.opitm_discB)

        # Update scaler after last "step"
        self.scaler.update()

    def execute(self,) -> None:

        for epoch in range(self.start_epoch, self.epochs):
            self.train_loader.sampler.set_epoch(epoch - self.start_epoch)

            avg_loss_gens, avg_loss_disc_A, avg_loss_disc_B = 0, 0, 0
            self.models.train()

            for x_batch, y_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, non_blocking = True)
                y_batch = y_batch.to(self.device, non_blocking = True)
                loss = self.forward(x_batch, y_batch)
                self.backward_gen(loss)
                loss_disc_A, loss_disc_B = self.forward(x_batch, y_batch)
                self.backward_disc(loss_disc_A, loss_disc_B)
            
                # Calculate average train loss
                avg_loss_gens += (loss / len(self.train_loader)).detach()
                avg_loss_disc_A += (loss_disc_A / len(self.train_loader)).detach()
                avg_loss_disc_B += (loss_disc_B / len(self.train_loader)).detach()

                del x_batch, y_batch, loss, loss_disc_A, loss_disc_B

            self.models.eval()
            # Share metrics
            metrics = torch.tensor([avg_loss_gens / self.world_size,
                                    avg_loss_disc_A / self.world_size,
                                    avg_loss_disc_B / self.world_size], device = self.device)
            dist.all_reduce(metrics, op = dist.ReduceOp.SUM)

            if self.device.index == 0:
                # Store metrics in JSON format to simplify parsing and transferring them into tensorboard at initial machine
                json_metrics = jdumps({'gens_loss' : metrics[0].item(),
                                       'disc_A_loss' : metrics[1].item(),
                                       'disc_B_loss' : metrics[2].item(),
                                       'epoch': epoch})
                # Send metrics into stdout. This channel going to be transferred into initial machine. 
                print(json_metrics)
            
                if (epoch + 1) % 20 == 0:                   
                    torch.save({'models': self.models.state_dict(),
                                'optim_gen': self.optim_gen.state_dict(),
                                'optim_discA': self.optim_discA.state_dict(),
                                'optim_discB': self.optim_discB.state_dict(),
                                'epoch': epoch,
                                'scaler': self.scaler.state_dict()}, 
                            os.path.join(self.model_weights_dir, str(epoch) + '.pt'))
        if self.device.index == 0:           
            self.make_archive(self.model_weights_dir, self.init_args.omodel)
        dist.destroy_process_group()
