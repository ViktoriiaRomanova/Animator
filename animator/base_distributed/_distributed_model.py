import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..utils.parameter_storages.transfer_parameters import DistParams


class BaseDist(ABC):
    """
    This class is an abstract base class (ABC) for distributed training model.

    It is a common core for implementing distributed training on N CPU/GPU.
    It is obligatory to call the parent __init__ function in a descendants __init__ method.
    To create a subclass, you need to implement the following functions:
        - prepare_dataloader
        - load_model
        - save_model
    """

    def __init__(self, rank: int, params: DistParams, random_state: int, shared_gpu: int) -> None:
        """Initialize the BaseDist class.

        Parameters:
            rank -- process index
            world_size -- number of workers(processes) which was spawned
            random_seed -- a number used to initialize a pseudorandom number generator

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseDist.__init__(self, rank, world_size)>
        Then, you need to define:
            - init_args - storage of params to get access to DataSphere resources
            - batch_size
            - epochs
            - DataLoader(s)
            - Model(s)
            - loss(s)
            - optimizer(s)
            - metric(s)
            - scaler(optional)
        """
        self.rank = rank
        self.world_size = params.world_size
        self.random_seed = random_state
        self.shared_gpu = shared_gpu

        # Set GPU number for this process
        if params.backend == "nccl":
            self.device = []
            self.generator = []
            for sub_device in range(self.shared_gpu):
                self.device.append(torch.device(rank * self.shared_gpu + sub_device))
                self.generator.append(torch.Generator(device=self.device[-1]).manual_seed(random_state))
        else:
            self.device = [torch.device("cpu")]
            self.generator = [torch.Generator(device=self.device).manual_seed(random_state)]

        # Setup the process group
        self.__setup(rank, params)

        # Create/check directories for model weights storage
        if rank == 0:
            self.model_weights_dir = self.__prepare_strorage_folders()

    def __setup(self, rank: int, params: DistParams) -> None:
        """Set up the process group."""
        os.environ["MASTER_ADDR"] = params.address
        os.environ["MASTER_PORT"] = params.port

        if self.device[0].type == "cpu":
            torch.set_num_threads(1)  # for test purposes

        # initialize the process group
        # 'nccl' -- for GPU
        dist.init_process_group(params.backend, rank=rank, world_size=self.world_size)

    def __prepare_strorage_folders(
        self,
    ) -> str:
        """Create/check directories for model weights storage."""
        working_directory = os.getcwd()
        model_weights_dir = os.path.join(
            working_directory, "train_checkpoints/", datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

        if not os.path.exists(model_weights_dir):
            os.makedirs(model_weights_dir)

        return model_weights_dir

    def _init_weights(self, module: nn.Module, init_type: str, mean: float, std: float) -> None:
        """Initialize model weights by a torch.nn.init function."""

        def init_func(sub_mod: nn.Module) -> None:
            module_to_init = {nn.Conv2d, nn.Linear, nn.ConvTranspose2d}
            if type(sub_mod) in module_to_init:
                if init_type == "normal":
                    nn.init.normal_(sub_mod.weight, mean, std)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(sub_mod.weight, a=0, mode="fan_in")
                else:
                    raise NotImplementedError("Initialization method {} is not implemented".format(init_type))
                nn.init.constant_(sub_mod.bias, 0.0)
            elif isinstance(sub_mod, nn.BatchNorm2d):
                nn.init.normal_(sub_mod.weight, 1.0, std)
                nn.init.constant_(sub_mod.bias, 0.0)

        module.apply(init_func)

    def _ddp_wrapper(self, model: nn.Module, is_multy_gpu: bool = False, **kwargs) -> nn.Module:
        is_multy_gpu |= self.device[0].type == "cpu"
        return DDP(
            model,
            device_ids=[self.device[0]] if not is_multy_gpu else None,
            output_device=self.device[0] if not is_multy_gpu else None,
            find_unused_parameters=False,
            **kwargs
        )

    def make_archive(
        self, source: str, destination: str, name: str = "train_checkpoints", f_format: str = "zip"
    ) -> None:
        """
        Archive model checkpoints.

        Name and directory set at initialization (config.yaml).
        """
        # Gets current model weights folder name
        archived_dir = os.path.basename(source)
        # Gets the catalogue address where the folder is stored
        root_dir = os.path.dirname(source)
        # Creates archive at main directory: /job/
        shutil.make_archive(name, f_format, root_dir, archived_dir)
        # Moves archive to requested directory
        shutil.move("{}.{}".format(name, f_format), destination)

    @abstractmethod
    def prepare_dataloader(
        self,
    ) -> DataLoader:
        """
        Split dataset into N parts.

        Returns: DataLoader instance for current part.
        """
        pass

    def set_requires_grad(self, models: nn.ModuleList, state: bool) -> None:
        """Set gradients state for each model."""
        for model in models:
            for param in model.parameters():
                param.requires_grad = state

    @abstractmethod
    def load_model(self, path: str, device: torch.device) -> int:
        """Load model, optimizer, etc. saved state."""
        pass

    @abstractmethod
    def save_model(
        self,
    ) -> dict:
        """Save model, optimizer, etc. state."""
        pass
