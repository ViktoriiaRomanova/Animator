import os
from json import dumps as jdumps
from typing import Any, List, Optional, Tuple
from argparse import Namespace
import shutil
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from datetime import datetime

from figure_extraction.processing_dataset import MaskDataset
from SegNetModel import SegNet
from UNet_model import UNet
from tqdm import tqdm

from figure_extraction.processing_dataset import MaskDataset
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

        train_set = MaskDataset(init_args.dataset, train_data, transform)
        val_set = MaskDataset(init_args.dataset, val_data, transform)

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
