import unittest
from argparse import Namespace
import yaml
import os
import time

import torch.multiprocessing as mp

from animator.style_transfer.dist_learning_model import DistLearning
from animator.utils.parameter_storages import TrainingParams
from animator.style_transfer.preprocessing_data import PreprocessingData

DATA_PATH = 'tests/style_transfer/test_img'
MODEL_CHECKPOINTS = 'tests/style_transfer/checkpoints'
HYPERPARAMETERS = 'animator/train_eval/style_transfer/hyperparameters.yaml'

def worker_init(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:
            
            dist_process = DistLearning(rank, args, params, train_data, val_data)


class MainTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(MODEL_CHECKPOINTS):
            os.makedirs(MODEL_CHECKPOINTS)
        
        cls.base_param = Namespace(dataset = DATA_PATH, omodel = MODEL_CHECKPOINTS, imodel = None)
        with open(HYPERPARAMETERS, 'r') as file:
            cls.params = TrainingParams(**yaml.safe_load(file))

        
        # Change default params for test purposes
        cls.params.main.epochs = 1
        cls.params.main.buffer_size = 2
        # local tests on CPU
        cls.params.distributed.backend = 'gloo'

        pr_data = PreprocessingData(cls.params.data.data_part)
        cls.train_data, cls.val_data = pr_data.get_data(cls.base_param.dataset,
                                                cls.params.main.random_state,
                                                cls.params.data.data_part)
    
    def test_DistLearning_init_setup(self,) -> None:        
        context = mp.spawn(worker_init, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        time.sleep(1)
        self.assertTrue(context.join())
