import unittest
from argparse import Namespace
import yaml
import os
import time

import torch.multiprocessing as mp
import torch

from animator.style_transfer.dist_learning_model import DistLearning
from animator.utils.parameter_storages import TrainingParams
from animator.style_transfer.preprocessing_data import PreprocessingData

DATA_PATH = 'tests/style_transfer/test_img'
MODEL_CHECKPOINTS = 'tests/style_transfer/checkpoints'
HYPERPARAMETERS = 'animator/train_eval/style_transfer/hyperparameters.yaml'

def worker_init(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:
            
            dist_process = DistLearning(rank, args, params, train_data, val_data)

def worker_run(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:
            
            dist_process = DistLearning(rank, args, params, train_data, val_data)
            dist_process.execute()
    


class MainTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(MODEL_CHECKPOINTS):
            os.makedirs(MODEL_CHECKPOINTS)
        
        cls.base_param = Namespace(datasetX = DATA_PATH,
                                   datasetY = DATA_PATH,
                                   omodel = MODEL_CHECKPOINTS, imodel = None)
        with open(HYPERPARAMETERS, 'r') as file:
            cls.params = TrainingParams(**yaml.safe_load(file))

        
        # Change default params for test purposes
        cls.params.main.epochs = 1
        cls.params.main.buffer_size = 2
        # local tests on CPU
        cls.params.distributed.backend = 'gloo'
        cls.params.data.data_part = 0.5
        cls.params.data.sub_part_data = 0.4

        pr_data = PreprocessingData(cls.params.data.data_part)
        train_dataX, val_dataX = pr_data.get_data(cls.base_param.datasetX,
                                                cls.params.main.random_state,
                                                cls.params.data.data_part)
        
        train_dataY, val_dataY = pr_data.get_data(cls.base_param.datasetY,
                                                cls.params.main.random_state,
                                                cls.params.data.data_part)
        cls.train_data = [train_dataX, train_dataY]
        cls.val_data = [val_dataX, val_dataY]
    
    def test_DistLearning_init_setup(self,) -> None:        
        context = mp.spawn(worker_init, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        self.assertTrue(len(context.pids()) == self.params.distributed.world_size)
        time.sleep(5)
        self.assertTrue(context.join())

    def test_DistLearning_run(self,) -> None: 
        print(torch.get_num_threads())       
        context = mp.spawn(worker_run, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        time.sleep(10)
        self.assertTrue(context.join())
