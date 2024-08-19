import random

from argparse import Namespace
from tqdm import tqdm
import os
from warnings import warn

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from animator.base_distributed._distributed_model import BaseDist
from .cycle_gan_model import Generator, Discriminator
from .losses import AdversarialLoss, CycleLoss, IdentityLoss
from ..utils.buffer import ImageBuffer
from .metric_storage import MetricStorage
from .get_dataset import GetDataset
from ..utils.parameter_storages.transfer_parameters import TrainingParams


__all__ = ['DistLearning']

class DistLearning(BaseDist):
    def __init__(self, rank: int, init_args: Namespace, 
                 params: TrainingParams,
                 train_data: list[list[str], list[str]],
                 val_data: list[list[str], list[str]] | None
                ) -> None:
        super().__init__(rank, params.distributed, params.main.random_state)

        self.init_args = init_args
        self.batch_size = params.main.batch_size
        self.epochs = params.main.epochs
        self.metrics = MetricStorage(self.rank, 0)

        # Create a folder to store intermediate results at s3 storage (Yandex Object Storage)
        self.s3_storage = None
        if rank == 0 and init_args.st is not None:
            self.s3_storage = os.path.join(init_args.st,
                                           os.path.basename(self.model_weights_dir))
            if not os.path.exists(self.s3_storage):
                os.makedirs(self.s3_storage)

        train_set = GetDataset(init_args.dataset, train_data,
                               size = params.data.size,
                               mean = params.data.mean,
                               std = params.data.std)
        
        self.train_loader = self.prepare_dataloader(train_set, rank,
                                                    self.world_size, self.batch_size, 
                                                    self.random_seed)
        
        # Create forward(A) and reverse(B) models
        # and initialize weights with Gaussian or Kaiming distribution
        self.genA = Generator()
        self.genB = Generator()

        self.discA = Discriminator()
        self.discB = Discriminator()

        self._init_weights(nn.ModuleList([self.genA, self.genB, self.discA, self.discB]),
                           init_type = params.models.init_type,
                           mean = params.models.mean,
                           std = params.models.std)

        self.genA = self._ddp_wrapper(self.genA.to(self.device))
        self.genB = self._ddp_wrapper(self.genB.to(self.device))
        self.discA = self._ddp_wrapper(self.discA.to(self.device))
        self.discB = self._ddp_wrapper(self.discB.to(self.device))

        self.modelA = nn.ModuleList([self.genA, self.discA])
        self.modelB = nn.ModuleList([self.genB, self.discB])

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.scaler = torch.cuda.amp.GradScaler(enabled = False) # self.device.type == 'cuda')

        self.fake_Y_buffer = ImageBuffer(self.world_size, params.main.buffer_size)
        self.fake_X_buffer = ImageBuffer(self.world_size, params.main.buffer_size)

        self.optim_gen = torch.optim.Adam(self.gens.parameters(),
                                          lr = params.optimizers.gen.lr,
                                          betas = params.optimizers.gen.betas)        
        self.optim_discA = torch.optim.Adam(self.discA.parameters(),
                                            lr = params.optimizers.discA.lr,
                                            betas = params.optimizers.discA.betas)        
        self.optim_discB = torch.optim.Adam(self.discB.parameters(),
                                            lr = params.optimizers.discB.lr,
                                            betas = params.optimizers.discB.betas)
        
        def lambda_rule(epoch: int) -> float:
            return (1.0 - (epoch - 100.0) / 101.0) if epoch > 100 else 1.0 
        
        self.scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.optim_gen, lr_lambda=lambda_rule)
        self.scheduler_discA = torch.optim.lr_scheduler.LambdaLR(self.optim_discA, lr_lambda=lambda_rule)
        self.scheduler_discB = torch.optim.lr_scheduler.LambdaLR(self.optim_discB, lr_lambda=lambda_rule)
        
        self.save_load_params = {'genA': self.genA,
                                 'discA': self.discA,
                                 'genB': self.genB,
                                 'discB': self.discB,
                                 'optim_gen': self.optim_gen,
                                 'optim_discA': self.optim_discA,
                                 'optim_discB': self.optim_discB,
                                 'scaler': self.scaler,
                                 'scheduler_gen': self.scheduler_gen,
                                 'scheduler_discA': self.scheduler_discA,
                                 'scheduler_discB': self.scheduler_discB,
                                 'bufferX': self.fake_X_buffer,
                                 'bufferY': self.fake_Y_buffer}
        
        self.start_epoch = 0
        if init_args.imodel is not None:
            self.start_epoch = self.load_model(init_args.imodel, self.device)
        
        self.epochs += self.start_epoch

        self.adv_loss = AdversarialLoss(params.loss.adversarial.ltype,
                                      params.loss.adversarial.real_val,
                                      params.loss.adversarial.fake_val,
                                      self.device)
        self.cycle_loss = CycleLoss(params.loss.cycle.ltype,
                                        params.loss.cycle.lambda_A,
                                        params.loss.cycle.lambda_B)
        self.idn_loss = IdentityLoss(params.loss.identity.ltype,
                                      params.loss.identity.lambda_idn)
        
        # to get different (from previous use) random numbers after loading the model
        random.seed(rank + self.start_epoch)

        #for model in self.models:
            #model.compile()

    def load_model(self, path: str, device: torch.device) -> int:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, path)
        state = torch.load(weights_dir, map_location = device)

        for key, param in self.save_load_params.items():
            if key not in state:
                warn('Loaded state dict doesn`t contain {} its loading omitted'.format(key))
                continue
            if isinstance(param, nn.Module):
                param.module.load_state_dict(state[key])
            else:
                param.load_state_dict(state[key])

        return state['epoch'] + 1

    def save_model(self, epoch: int) -> dict:
        state = {}
        for key, param in self.save_load_params.items():
            if isinstance(param, nn.Module):
                state[key] = param.module.state_dict()
            else:
                state[key] = param.state_dict()
        state['epoch'] = epoch
        return state

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
                                sampler = sampler,
                                pin_memory = self.device.type != 'cpu',
                                num_workers = 1,
                                prefetch_factor = 16,
                                multiprocessing_context = 'spawn',
                                persistent_workers = self.device.type != 'cpu',
                                pin_memory_device = self.device.type if self.device.type != 'cpu' else '')
        return data_loader
    
    def forward_gen(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        with torch.autocast(device_type = self.device.type,
                            dtype = torch.float16,
                            enabled = False): #self.device.type == 'cuda'):
            self.set_requires_grad(self.discs, False)

            fakeY = self.genA(X)
            cycle_fakeX = self.genB(fakeY)
            fakeX = self.genB(Y)
            cycle_fakeY = self.genA(fakeX)

            self.fake_X_buffer.add(fakeX.detach().clone())
            self.fake_Y_buffer.add(fakeY.detach().clone())

            adv_lossA = self.adv_loss(self.discA(fakeY), True)
            adv_lossB = self.adv_loss(self.discB(fakeX), True)
            cycle_lossA, cycle_lossB = self.cycle_loss(cycle_fakeX, cycle_fakeY, X, Y)
            idn_lossX, idn_lossY = self.idn_loss(self.genB(X), self.genA(Y), X, Y)

            loss = adv_lossA + adv_lossB + cycle_lossA + cycle_lossB + idn_lossX + idn_lossY

            self.metrics.update('Total_loss', 'gens', loss.detach().clone())
            self.metrics.update('Adv_gen', 'discA', adv_lossA.detach().clone())
            self.metrics.update('Adv_gen', 'discB', adv_lossB.detach().clone())
            self.metrics.update('Cycle', 'A', cycle_lossA.detach().clone())
            self.metrics.update('Cycle', 'B', cycle_lossB.detach().clone())
            self.metrics.update('Identity', 'X', idn_lossX.detach().clone())
            self.metrics.update('Identity', 'Y', idn_lossY.detach().clone())

        return loss

    def forward_disc(self, X: torch.Tensor, Y: torch.Tensor,
                     adv_alpha: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.autocast(device_type = self.device.type,
                            dtype = torch.float16,
                            enabled = False): #self.device.type == 'cuda'):
            self.set_requires_grad(self.discs, True)

            ans_disc_A_false = self.discA(self.fake_Y_buffer.get())
            ans_disc_B_false = self.discB(self.fake_X_buffer.get())

            ans_disc_A_true = self.discA(Y)
            ans_disc_B_true = self.discB(X)

            lossA_true = self.adv_loss(ans_disc_A_true, True) * adv_alpha
            lossA_false = adv_alpha * self.adv_loss(ans_disc_A_false, False)
            lossA = lossA_true + lossA_false

            lossB_true = self.adv_loss(ans_disc_B_true, True) * adv_alpha
            lossB_false = adv_alpha * self.adv_loss(ans_disc_B_false, False)
            lossB = lossB_true + lossB_false

            self.metrics.update('Total_loss', 'disc_A', lossA.detach().clone())
            self.metrics.update('Total_loss', 'disc_B', lossB.detach().clone())
            self.metrics.update('Adv_discA', 'True', lossA_true.detach().clone())
            self.metrics.update('Adv_discA', 'False', lossA_false.detach().clone())
            self.metrics.update('Adv_discB', 'True', lossB_true.detach().clone())
            self.metrics.update('Adv_discB', 'False', lossB_false.detach().clone())

            self.metrics.update('Accuracy', 'disc_A', ans_disc_A_true, torch.tensor(1.0, device=self.device).expand_as(ans_disc_A_true))
            self.metrics.update('Accuracy', 'disc_A', ans_disc_A_false, torch.tensor(0.0, device=self.device).expand_as(ans_disc_A_false))

            self.metrics.update('Accuracy', 'disc_B', ans_disc_B_true, torch.tensor(1.0, device=self.device).expand_as(ans_disc_B_true))
            self.metrics.update('Accuracy', 'disc_B', ans_disc_B_false, torch.tensor(0.0, device=self.device).expand_as(ans_disc_B_false))
        
            del ans_disc_A_false, ans_disc_A_true, ans_disc_B_false, ans_disc_B_true
        return lossA, lossB

    
    def backward_gen(self, loss: torch.Tensor) -> None:
        self.gens.zero_grad(True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim_gen)
    
    def backward_disc(self, lossA: torch.Tensor, lossB: torch.Tensor) -> None:
        self.discs.zero_grad(True)
        self.scaler.scale(lossA).backward()
        self.scaler.scale(lossB).backward()

        self.scaler.step(self.optim_discA)
        self.scaler.step(self.optim_discB)

        # Update scaler after last "step"
        self.scaler.update()

    def execute(self,) -> None:

        for epoch in range(self.start_epoch, self.epochs):
            self.train_loader.sampler.set_epoch(epoch)

            self.models.train()

            for x_batch, y_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, non_blocking = True)
                y_batch = y_batch.to(self.device, non_blocking = True)
                loss = self.forward_gen(x_batch, y_batch)
                self.backward_gen(loss)
                loss_disc_A, loss_disc_B = self.forward_disc(x_batch, y_batch)
                self.backward_disc(loss_disc_A, loss_disc_B)

                self.fake_X_buffer.step()
                self.fake_Y_buffer.step()

                del x_batch, y_batch, loss, loss_disc_A, loss_disc_B
            
            self.metrics.update('Lr', 'gens',
                                torch.tensor(self.scheduler_gen.get_last_lr()[0], device=self.device))
            self.metrics.update('Lr', 'disc_A',
                                torch.tensor(self.scheduler_discA.get_last_lr()[0], device=self.device))
            self.metrics.update('Lr', 'disc_B',
                                torch.tensor(self.scheduler_discB.get_last_lr()[0], device=self.device))
            self.metrics.epoch = epoch
            
            self.scheduler_gen.step()
            self.scheduler_discA.step()
            self.scheduler_discB.step()

            # Send metrics into stdout. This channel going to be transferred into initial machine.
            self.metrics.compute()
            self.metrics.reset()

            if self.rank == 0:            
                if (epoch + 1) % 1 == 0:
                    if self.s3_storage is not None:
                        # Save model weights at S3 storage if the path to a bucket provided
                        torch.save(self.save_model(epoch),
                                   os.path.join(self.s3_storage, str(epoch) + '.pt'))
                    else:
                        # Otherwise, save at a remote machine
                        warn(' '.join(('Intermediate model weights are saved at the remote machine and will be lost',
                                       'after the end of the training process')))
                        torch.save(self.save_model(epoch),
                                   os.path.join(self.model_weights_dir, str(epoch) + '.pt'))               
        if self.rank == 0 and self.s3_storage is not None:
            # Save final results at s3 storage          
            torch.save(self.save_model(self.epochs - 1),
                       os.path.join(self.s3_storage, str(self.epochs - 1) + '.pt'))
        dist.destroy_process_group()
