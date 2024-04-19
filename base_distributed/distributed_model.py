import os
from abc import ABC, abstractmethod
import shutil

from datetime import datetime
from torch import nn
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseDist(ABC):
    """
        This class is an abstract base class (ABC) for distributed training model.
        To create a subclass, you need to implement the following five functions:

            !!!ADD FINAL LIST OF FUNCTIONS!!!
    """
    def __init__(self, rank: int, world_size: int,
                 seed: int = 42) -> None:
        """Initialize the BaseDist class.

        Parameters:
            rank -- process index
            world_size -- number of workers(processes) which was spawned
            random_seed -- a number used to initialize a pseudorandom number generator 

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseDist.__init__(self, rank, world_size)>
        Then, you need to define:

        !!!ADD DESCRIPTION HERE!!!
 
        """        
        self.world_size = world_size
        self.random_seed = seed
        # Set GPU number for this process
        self.device = torch.device(rank)

        # Setup the process group
        self.__setup(rank)

        #Create/check directories for model weights storage
        if rank == 0:
            self.model_weights_dir = self.__prepare_strorage_folders()

                

    def __setup(self, rank: int) -> None:
        """Setup the process group."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # initialize the process group
        # 'nccl' -- for GPU
        dist.init_process_group('nccl', rank = rank, world_size = self.world_size)
    
    def __prepare_strorage_folders(self,) -> str:
        """Create/check directories for model weights storage."""
        working_directory = os.getcwd()
        model_weights_dir = os.path.join(working_directory, 'train_checkpoints/',
                                         datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

        if not os.path.exists(model_weights_dir):
            os.makedirs(model_weights_dir)

        return model_weights_dir
    
    def _init_weights(self, module: nn.Module, mean: float, std: float) -> None:
        """Initialize model weights by a torch.nn.init function."""
        def init_func(sub_mod: nn.Model) -> None:
            module_to_init = {nn.Conv2d, nn.Linear, nn.BatchNorm2d}
            if type(sub_mod) in module_to_init:
                nn.init.normal_(sub_mod.weight, mean, std)
                nn.init.constant_(sub_mod.bias, 0.0)

        module.apply(init_func)
    
    def _ddp_wrapper(self, model: nn.Module) -> nn.Module:
        return DDP(model, device_ids = self.device, output_device = self.device,
                   find_unused_parameters = False)

    def make_archive(self, source: str, destination: str) -> None:
        """ 
            Archives model checkpoints with name and directory 
            set at initialization (config.yaml).
        """
        # Gets the name of a resulted file 
        name, f_format = os.path.basename(destination).split('.')
        # Gets current model weights folder name
        archived_dir = os.path.basename(source)
        # Gets the catalogue address where the folder is stored
        root_dir = os.path.dirname(source)
        # Creates archive at main directory: /job/
        shutil.make_archive(name, f_format, root_dir, archived_dir)
        # Moves archive to requested directory
        shutil.move('{}.{}'.format(name, f_format), os.path.dirname(destination))
       
    @abstractmethod
    def prepare_dataloader(self,) -> DataLoader:
        """
            Split dataset into N parts.

            Returns: DataLoader instance for current part.
        """
        pass

    def set_requires_grad(self, models: list[nn.Module], state: bool) -> None:
        """Set gradients state for each model."""
        for model in models:
            for param in model.parameters():
                param.requires_grad = state

    def load_model(self, to_populate: dict[str, nn.Module | torch.cuda.amp.GradScaler],
                   path: str, device: torch.device) -> None:
        working_directory = os.getcwd()
        weights_dir = os.path.join(working_directory, path)
        state = torch.load(weights_dir, map_location = device)
        for key, val in to_populate.items():
            val.load_state_dict(state[key])
        return state['epoch'] + 1


        