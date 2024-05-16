import os
from json import dumps as jdumps
from argparse import Namespace
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchmetrics import JaccardIndex 

from .get_dataset import MaskDataset
from .segnet_model import SegNet
from .unet_model import UNet
from tqdm import tqdm

from animator.base_distributed._distributed_model import BaseDist
from ..utils.parameter_storages.extraction_parameters import ExtTrainingParams


class ExtractionDistLearning(BaseDist):
    def __init__(self, rank, init_args: Namespace,
                 params: ExtTrainingParams,
                 train_data: list[str],
                 val_data: list[str]) -> None:
        super().__init__(rank, params.distributed, params.main.random_state)

        self.init_args = init_args
        self.batch_size = params.main.batch_size
        self.epochs = params.main.epochs
        self.start_epoch = 0

        # prepare data
        transform = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p = 0.1),
            transforms.RandomVerticalFlip(p = 0.1),
            transforms.RandomPerspective(p = 0.1)])

        train_set = MaskDataset(init_args.dataset, train_data,
                                size = params.data.size,
                                mean = params.data.mean,
                                std = params.data.std,
                                transform = transform)
        val_set = MaskDataset(init_args.dataset, val_data,
                              size = params.data.size,
                              mean = params.data.mean,
                              std = params.data.std,
                              transform = transform)

        self.train_loader = self.prepare_dataloader(train_set, rank,
                                               self.world_size,
                                               self.batch_size,
                                               self.random_seed)
        
        self.val_loader = self.prepare_dataloader(val_set, rank,
                                             self.world_size,
                                             self.batch_size,
                                             self.random_seed)

        # prepare model
        if params.model.mtype == 'UNet':
            model = UNet(params.model.marchitecture)
        elif params.model.mtype == 'SegNet':
            model = SegNet(params.model.marchitecture)
        else:
            raise NotImplementedError('model type {} is not found'.format(params.model.mtype))
        
        self.model = self._ddp_wrapper(model.to(self.device))

        self.scaler = torch.cuda.amp.GradScaler(enabled = self.device.type == 'cuda')

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr = params.optimizers.lr,
                                      betas = params.optimizers.betas)
        
        self.loss = torch.compile(nn.BCEWithLogitsLoss())
        self.metric = torch.compile(JaccardIndex('binary',
                                                 threshold = params.metrics.threshhold))
        
        self.save_load_params = {'model': self.model,
                                 'optim': self.optim,
                                 'scaler': self.scaler}

        if init_args.imodel is not None:
            self.start_epoch = self.load_model(init_args.imodel, self.device)
            self.epochs += self.start_epoch
        
        self.model.compile()
    
    def load_model(self, path: str, device: torch.device) -> int:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, path)
        state = torch.load(weights_dir, map_location = device)

        for key, param in self.save_load_params.items():
            param.load_state_dict(state[key])

        return state['epoch'] + 1

    def save_model(self, epoch: int) -> dict:
        state = {}
        for key, param in self.save_load_params.items():
            state[key] = param.state_dict()
        state['epoch'] = epoch
        return state


    def prepare_dataloader(self, data: MaskDataset, rank: int,
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
                             num_workers = 8,
                             prefetch_factor = 16,
                             multiprocessing_context = 'spawn',
                             persistent_workers = self.device.type != 'cpu',
                             pin_memory_device = self.device.type if self.device.type != 'cpu' else '')
        return data_loader
    
    def forward(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
        with torch.autocast(device_type = self.device.type,
                            dtype = torch.float16,
                            enabled = self.device.type == 'cuda'):
            y_pred = self.model(x_batch)
            loss = self.loss(y_pred, y_batch)
            metric = self.metric(y_pred, y_batch)
        return loss, metric

    def backward(self, loss: torch.Tensor) -> None:
        
        self.model.zero_grad(True)

        self.scaler(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()        
    
    def execute(self,) -> None:

        for epoch in range(self.start_epoch, self.epochs):
            self.train_loader.sampler.set_epoch(epoch - self.start_epoch)
            self.val_loader.sampler.set_epoch(epoch - self.start_epoch)

            self.model.train()
            train_loss, train_metric = 0, 0
            for x_batch, y_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, non_blocking = True)
                y_batch = y_batch.to(self.device, non_blocking = True)
                loss, metric = self.forward(x_batch, y_batch)
                self.backward(loss)

                train_loss += (loss.detach() / len(self.train_loader))
                # !it is not final result, to get real metric need to divide it into num_batches
                train_metric += metric

                del x_batch, y_batch, y_pred, loss
            train_metric /= len(self.train_loader)

            self.model.eval()
            val_loss, val_metric = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in tqdm(self.val_loader):
                    x_batch = x_batch.to(self.device, non_blocking = True)
                    y_batch = y_batch.to(self.device, non_blocking = True)
                    loss, metric = self.forward(x_batch, y_batch)

                    val_loss += (loss.detach() / len(self.val_loader))
                    # !it is not final result, to get real metric need to divide it into num_batches
                    val_metric += metric

                del x_batch, y_batch, y_pred, loss
            val_metric /= len(self.val_loader)

            # Share metrics
            metrics = torch.tensor([train_loss / self.world_size,
                                    val_loss / self.world_size,
                                    train_metric / self.world_size,
                                    val_metric / self.world_size], device = self.device)
            dist.all_reduce(metrics, op = dist.ReduceOp.SUM)

            if self.rank == 0:
                # Store metrics in JSON format to simplify parsing and transferring them into tensorboard at initial machine
                json_metrics = jdumps({'Loss': 
                                            {'train' : metrics[0].item(),
                                             'val' : metrics[1].item()
                                            },
                                        'IoU':
                                            {'train' : metrics[2].item(),
                                             'val' : metrics[3].item()
                                            },
                                       'epoch': epoch})
                # Send metrics into stdout. This channel going to be transferred into initial machine. 
                print(json_metrics)
            
                if (epoch + 1) % 1 == 0:                   
                    torch.save(self.save_model(epoch), 
                            os.path.join(self.model_weights_dir, str(epoch) + '.pt'))
        if self.rank == 0:           
            self.make_archive(self.model_weights_dir, self.init_args.omodel)
        dist.destroy_process_group()
