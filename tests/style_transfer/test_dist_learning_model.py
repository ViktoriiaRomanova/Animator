import unittest
from argparse import Namespace
import yaml
import os
import time
import shutil
import multiprocessing

import torch.multiprocessing as mp
import torch

from animator.style_transfer.dist_learning_model import DistLearning
from animator.utils.parameter_storages import TrainingParams
from animator.style_transfer.preprocessing_data import PreprocessingData

DATA_PATH = 'tests/style_transfer/test_img'
MODEL_CHECKPOINTS = 'tests/style_transfer/checkpoints/train_checkpoints.zip'
HYPERPARAMETERS = 'animator/train_eval/style_transfer/hyperparameters.yaml'

def worker_init(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:

            torch.set_num_threads(1)
            dist_process = DistLearning(rank, args, params, train_data, val_data)

def worker_run(rank: int, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:

            torch.set_num_threads(1)
            dist_process = DistLearning(rank, args, params, train_data, val_data)
            dist_process.execute()

def worker_load(rank: int, conn_queue: multiprocessing.Queue, args: Namespace, params: TrainingParams,
                   train_data: list[str], val_data: list[str]) -> None:

            torch.set_num_threads(1)
            dist_process = DistLearning(rank, args, params, train_data, val_data)
            state = {'models': dist_process.models.state_dict(),
                                'optim_gen': dist_process.optim_gen.state_dict(),
                                'optim_discA': dist_process.optim_discA.state_dict(),
                                'optim_discB': dist_process.optim_discB.state_dict(),
                                'epoch': dist_process.start_epoch - 1,
                                'scaler': dist_process.scaler.state_dict()}.__str__()
            conn_queue.put(state)
            dist_process.execute()
    


class MainTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dir = os.path.dirname(MODEL_CHECKPOINTS)
        if not os.path.exists(dir):
           os.makedirs(dir)
       
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
    
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('./train_checkpoints')
        shutil.rmtree(os.path.dirname(MODEL_CHECKPOINTS))
    
    def tearDown(self) -> None:
        os.remove(MODEL_CHECKPOINTS)

    def test_DistLearning_init_setup(self,) -> None:        
        context = mp.spawn(worker_init, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        self.assertTrue(len(context.pids()) == self.params.distributed.world_size)
        time.sleep(5)
        self.assertTrue(context.join())

    def test_DistLearning_run_save_model(self,) -> None: 
    
        context = mp.spawn(worker_run, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        time.sleep(100)
        self.assertTrue(os.path.getsize(MODEL_CHECKPOINTS) > 10 * 1024 and context.join())
       
    def test_DistLearning_load_save_model(self,) -> None:

        self.base_param.imodel = 'tests/style_transfer/test_weights/0.pt'
        multiprocessing.set_start_method('spawn')

        conn_queue = multiprocessing.Queue()

        context = mp.spawn(worker_load, args = (conn_queue, self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        time.sleep(10)
        init_state = torch.load(self.base_param.imodel).__str__()
        is_equal = True
        while conn_queue.qsize() > 0:
              state = conn_queue.get()
              is_equal &= state == init_state

        time.sleep(100)
        
        self.assertTrue(context.join() and
                        is_equal and
                        os.path.getsize(MODEL_CHECKPOINTS) > 10 * 1024)
