import unittest
from argparse import Namespace
import yaml
import os
import time

import torch.multiprocessing as mp

from animator.style_transfer.dist_learning_model import DistLearning
from animator.utils.parameter_storages import TrainingParams
from animator.style_transfer.preprocessing_data import PreprocessingData

DATA_PATH = 'animator/tests/style_transfer/test_img/'
MODEL_CHECKPOINTS = 'animator/tests/style_transfer/checkpoints'
HYPERPARAMETERS = 'animator/train_eval/style_transfer/hyperparameters.yaml'


class MainTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def sutUpClass(cls) -> None:
        if not os.path.exists(MODEL_CHECKPOINTS):
            os.makedirs(MODEL_CHECKPOINTS)
        
        cls.args = Namespace(dataset = DATA_PATH, omodel = MODEL_CHECKPOINTS, imodel = None)
        with open(HYPERPARAMETERS, 'r') as file:
            params = TrainingParams(**yaml.safe_load(file))

        
        # Change default params for test purposes
        cls.params.main.epochs = 1
        cls.params.main.buffer_size = 2
        # local tests on CPU
        cls.params.distributed.backend = 'gloo'

        pr_data = PreprocessingData(params.data.data_part)
        cls.train_data, cls.val_data = pr_data.get_data(cls.args.dataset,
                                                cls.params.main.random_state,
                                                cls.params.data.data_part)
    
    def test_DistLerning_init_setup() -> None:
        def worker(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:
            
            dist_process = DistLearning(rank, args, params, train_data, val_data)

            time.sleep(1)


