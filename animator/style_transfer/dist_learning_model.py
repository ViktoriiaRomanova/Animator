from argparse import Namespace
from tqdm import tqdm
import os
from json import dumps as jdumps
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

        # Create a folder to store intermediate results at s3 storage (Yandex Object Storage)
        self.s3_storage = None
        if rank == 0 and init_args.st is not None:
            self.s3_storage = os.path.join(init_args.st,
                                           os.path.basename(self.model_weights_dir))
            if not os.path.exists(self.s3_storage):
                os.makedirs(self.s3_storage)

        datasetX = os.path.join(init_args.dataset, 'domainX/')
        datasetY = os.path.join(init_args.dataset, 'domainY/')

        train_setX = GetDataset(datasetX, train_data[0],
                               size = params.data.size,
                               mean = params.data.mean,
                               std = params.data.std)
        
        train_setY = GetDataset(datasetY, train_data[1],
                               size = params.data.size,
                               mean = params.data.mean,
                               std = params.data.std)
        
        self.train_loaderX = self.prepare_dataloader(train_setX, rank,
                                                    self.world_size, self.batch_size, 
                                                    self.random_seed)
        
        self.train_loaderY = self.prepare_dataloader(train_setY, rank,
                                                    self.world_size, self.batch_size, 
                                                    self.random_seed)

        # Create forward(A) and reverse(B) models
        # and initialize weights with Gaussian or Kaiming distribution
        self.genA = Generator().to(self.device)
        self.genB = Generator().to(self.device)

        self.discA = Discriminator().to(self.device)
        self.discB = Discriminator().to(self.device)

        self._init_weights(nn.ModuleList([self.genA, self.genB, self.discA, self.discB]),
                           init_type = params.models.init_type,
                           mean = params.models.mean,
                           std = params.models.std)

        self.genA = self._ddp_wrapper(self.genA)
        self.genB = self._ddp_wrapper(self.genB)
        self.discA = self._ddp_wrapper(self.discA)
        self.discB = self._ddp_wrapper(self.discB)

        self.modelA = nn.ModuleList([self.genA, self.discA])
        self.modelB = nn.ModuleList([self.genB, self.discB])

        self.gens = nn.ModuleList([self.genA, self.genB])
        self.discs = nn.ModuleList([self.discA, self.discB])

        self.models = nn.ModuleList([self.genA, self.discA, self.genB, self.discB])

        self.scaler = torch.cuda.amp.GradScaler(enabled = False) # self.device.type == 'cuda')

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
        
        def lambda_rule(epoch: int) -> float:
            return (1.0 - (epoch - 200.0) / 201.0) if epoch > 200 else 1.0 
        
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
                                 'scheduler_discB': self.scheduler_discB}
        
        self.start_epoch = 0
        if init_args.imodel is not None:
            self.start_epoch = self.load_model(init_args.imodel, self.device)
        
        self.epochs += self.start_epoch

        self.adv_loss = torch.compile(AdversarialLoss(params.loss.adversarial.ltype,
                                      params.loss.adversarial.real_val,
                                      params.loss.adversarial.fake_val,
                                      self.device))
        self.cycle_loss = torch.compile(CycleLoss(params.loss.cycle.ltype,
                                        params.loss.cycle.lambda_A,
                                        params.loss.cycle.lambda_B))
        self.idn_loss = torch.compile(IdentityLoss(params.loss.identity.ltype,
                                      params.loss.identity.lambda_idn))

        for model in self.models:
            model.compile()

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
                            enabled = False): # self.device.type == 'cuda'):
            self.set_requires_grad(self.discs, False)

            fakeY = self.genA(X)
            cycle_fakeX = self.genB(fakeY)
            fakeX = self.genB(Y)
            cycle_fakeY = self.genA(fakeX)

            self.fake_X_buffer.add(fakeX.detach())
            self.fake_Y_buffer.add(fakeY.detach())

            loss = self.adv_loss(self.discB(fakeX), True) + self.adv_loss(self.discA(fakeY), True) + \
                + self.cycle_loss(cycle_fakeX, cycle_fakeY, X, Y) + \
                + self.idn_loss(self.genB(X), self.genA(Y), X, Y)
        
        return loss

    def forward_disc(self, X: torch.Tensor, Y: torch.Tensor,
                     adv_alpha: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.autocast(device_type = self.device.type,
                            dtype = torch.float16,
                            enabled = False): #self.device.type == 'cuda'):
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

        self.scaler.step(self.optim_discA)
        self.scaler.step(self.optim_discB)

        # Update scaler after last "step"
        self.scaler.update()

    def execute(self,) -> None:

        for epoch in range(self.start_epoch, self.epochs):
            self.train_loaderX.sampler.set_epoch(epoch - self.start_epoch)
            self.train_loaderY.sampler.set_epoch(epoch - self.start_epoch)

            avg_loss_gens, avg_loss_disc_A, avg_loss_disc_B = 0, 0, 0
            num_butch = min(len(self.train_loaderX), len(self.train_loaderY))
            self.models.train()

            for x_batch, y_batch in tqdm(zip(self.train_loaderX, self.train_loaderY),
                                         total = num_butch):
                x_batch = x_batch.to(self.device, non_blocking = True)
                y_batch = y_batch.to(self.device, non_blocking = True)
                loss = self.forward_gen(x_batch, y_batch)
                self.backward_gen(loss)
                loss_disc_A, loss_disc_B = self.forward_disc(x_batch, y_batch)
                self.backward_disc(loss_disc_A, loss_disc_B)
            
                # Calculate average train loss
                avg_loss_gens += (loss / num_butch).detach()
                avg_loss_disc_A += (loss_disc_A / num_butch).detach()
                avg_loss_disc_B += (loss_disc_B / num_butch).detach()

                del x_batch, y_batch, loss, loss_disc_A, loss_disc_B
            
            self.scheduler_gen.step()
            self.scheduler_discA.step()
            self.scheduler_discB.step()

            self.models.eval()
            # Share metrics
            metrics = torch.tensor([avg_loss_gens / self.world_size,
                                    avg_loss_disc_A / self.world_size,
                                    avg_loss_disc_B / self.world_size], device = self.device)
            dist.all_reduce(metrics, op = dist.ReduceOp.SUM)

            if self.rank == 0:
                # Store metrics in JSON format to simplify parsing and transferring them into tensorboard at initial machine
                json_metrics = jdumps({'Loss': 
                                            {'gens' : metrics[0].item(),
                                             'disc_A' : metrics[1].item(),
                                             'disc_B' : metrics[2].item()
                                            },
                                        'Lr':
                                            {'gens': self.scheduler_gen.get_last_lr()[0],
                                             'discA': self.scheduler_discA.get_last_lr()[0],
                                             'discB': self.scheduler_discB.get_last_lr()[0]
                                            },
                                       'epoch': epoch})
                
                # Send metrics into stdout. This channel going to be transferred into initial machine. 
                print(json_metrics)
            
                if (epoch + 1) % 5 == 0:
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
            torch.save(self.save_model(epoch),
                       os.path.join(self.s3_storage, str(epoch) + '.pt'))
        dist.destroy_process_group()
