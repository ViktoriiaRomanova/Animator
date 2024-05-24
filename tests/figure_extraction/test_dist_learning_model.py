import unittest
from argparse import Namespace
import os
import time
import shutil
import pickle
from queue import Empty

import torch.multiprocessing as mp
import torch

from animator.figure_extraction.dist_learning_model import ExtractionDistLearning
from animator.utils.parameter_storages.params_holder import ParamsHolder
from animator.utils.parameter_storages.extraction_parameters import ExtTrainingParams
from animator.utils.preprocessing_data import PreprocessingData
from animator.figure_extraction.get_dataset import checker
from tests.figure_extraction import DATA_PATH, MODEL_CHECKPOINTS, HYPERPARAMETERS

TIME_DATA_LOADING = 5.0
TIME_MODEL_EXEC = 70

def setUpModule() -> None:
    mp.set_start_method('spawn', force = True)

    dir = os.path.dirname(MODEL_CHECKPOINTS)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tearDownModule() -> None:
    dir = './train_checkpoints'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    dir = os.path.dirname(MODEL_CHECKPOINTS)
    if os.path.exists(dir):
        shutil.rmtree(dir)


def worker_init(rank: int, args: Namespace, params: ExtTrainingParams,
                   train_data: list[str],
                   val_data: list[str] | None) -> None:

            torch.set_num_threads(1)
            _ = ExtractionDistLearning(rank, args, params, train_data, val_data)

def worker(rank: int, conn_queue: mp.Queue, args: Namespace, params: ExtractionDistLearning,
                   train_data: list[list[str], list[str]],
                   val_data: list[list[str], list[str]] | None) -> None:

            torch.set_num_threads(1)
            dist_process = ExtractionDistLearning(rank, args, params, train_data, val_data)
            state = pickle.dumps(dist_process.save_model(max(0, dist_process.start_epoch - 1)))
            conn_queue.put(state)
            dist_process.execute()

def worker_sampler(rank: int, conn_queue: mp.Queue, args: Namespace, params: ExtTrainingParams,
                   train_data: list[list[str], list[str]],
                   val_data: list[list[str], list[str]] | None) -> None:

            torch.set_num_threads(1)
            dist_process = ExtractionDistLearning(rank, args, params, train_data, val_data)

            conn_queue.put({sample for sample in dist_process.train_loader.sampler})


def compare_states(state1: dict, state2: dict) -> bool:
    if len(state1) != len(state2): return False
    ans = True
    for key in state1:
        if key not in state2: 
             return False
        ans &= state1[key].__str__() == state2[key].__str__()
        if not ans: return ans
    return ans    


class MainTrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        base_param = Namespace(dataset = DATA_PATH,
                               omodel = MODEL_CHECKPOINTS, imodel = None)

        with unittest.mock.patch('argparse.ArgumentParser.parse_args', return_value = base_param):
            cls.holder = ParamsHolder(HYPERPARAMETERS, 'Extraction')
        
        cls.params = cls.holder.hyper_params
        cls.base_param = cls.holder.datasphere_params

        # Change default params for test purposes
        cls.params.main.epochs = 1
        cls.params.main.batch_size = 1
        # local tests on CPU
        cls.params.distributed.backend = 'gloo'

        cls.params.data.data_part = 0.5
        cls.params.data.sub_part_data = 0.4

        # test for two distributed process
        cls.params.distributed.world_size = 2

        cls.params.model.mtype = 'UNet'
        cls.params.model.marchitecture = 'C'

        pr_data = PreprocessingData(cls.params.data.data_part, checker = checker)
        cls.train_data, cls.val_data = pr_data.get_data(os.path.join(cls.base_param.dataset, 'input'),
                                                cls.params.main.random_state,
                                                cls.params.data.sub_part_data)
        
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.train_data, cls.val_data, cls.params, cls.base_param
        
    def tearDown(self) -> None:
        self.base_param.imodel = None
        if os.path.exists(MODEL_CHECKPOINTS):
            os.remove(MODEL_CHECKPOINTS)

    def test_ExtractionDistLearning_init_setup(self,) -> None:     
        context = mp.spawn(worker_init, args = (self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        self.assertTrue(len(context.pids()) == self.params.distributed.world_size)
        time.sleep(5)
        self.assertTrue(context.join(1))

    def test_ExtractionDistLearning_init_save_model(self,) -> None: 
        conn_queue = mp.Queue()
    
        context = mp.spawn(worker, args = (conn_queue, self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)
        try:
            state = pickle.loads(conn_queue.get(timeout = TIME_DATA_LOADING))
            is_equal = compare_states(state, pickle.loads(conn_queue.get(timeout = TIME_DATA_LOADING)))
        except Empty:
            context.join(0)
        
        while not context.join(TIME_MODEL_EXEC):
            pass

        self.assertTrue(is_equal and
                        os.path.getsize(MODEL_CHECKPOINTS) > 10 * 1024)
        del state


    def test_DistLearning_load_save_model(self,) -> None:
        print('test1')
        self.base_param.imodel = 'tests/figure_extraction/test_weights/0.pt'

        conn_queue = mp.Queue()

        context = mp.spawn(worker, args = (conn_queue, self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        init_state = torch.load(self.base_param.imodel)
        # Erase the scaler state as it is not going to be loaded while the scaler is disabled
        # (this test works on CPU)
        init_state['scaler'] = {}       
        try:
            state1 = pickle.loads(conn_queue.get(timeout = TIME_DATA_LOADING))
            state2 = pickle.loads(conn_queue.get(timeout = TIME_DATA_LOADING))

            is_equal = compare_states(state1, init_state) & compare_states(state2, init_state)
        except Empty:
            context.join(0)
        
        while not context.join(TIME_MODEL_EXEC):
            pass
        
        self.assertTrue(is_equal and
                        os.path.getsize(MODEL_CHECKPOINTS) > 10 * 1024)
        del state1, state2, init_state


class DistSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
       
        base_param = Namespace(dataset = DATA_PATH,
                               omodel = MODEL_CHECKPOINTS, imodel = None)

        with unittest.mock.patch('argparse.ArgumentParser.parse_args', return_value = base_param):
            cls.holder = ParamsHolder(HYPERPARAMETERS, 'Extraction')
        
        cls.params = cls.holder.hyper_params
        cls.base_param = cls.holder.datasphere_params
        
        # Change default params for test purposes
        cls.params.main.epochs = 1
        cls.params.main.batch_size = 1
        # local tests on CPU
        cls.params.distributed.backend = 'gloo'
        cls.params.data.data_part = 0.8
        cls.params.data.sub_part_data = 1.0

        # test for two distributed process
        cls.params.distributed.world_size = 2

        cls.params.model.mtype = 'UNet'
        cls.params.model.marchitecture = 'C'

        pr_data = PreprocessingData(cls.params.data.data_part, checker = checker)
        cls.train_data, cls.val_data = pr_data.get_data(os.path.join(cls.base_param.dataset, 'input'),
                                                cls.params.main.random_state,
                                                cls.params.data.sub_part_data)
        
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.train_data, cls.val_data, cls.params, cls.base_param

    def test_DistLearning_sampler(self,) -> None:
        conn_queue = mp.Queue()
    
        context = mp.spawn(worker_sampler, args = (conn_queue, self.base_param, self.params, self.train_data, self.val_data),
                           join = False, nprocs = self.params.distributed.world_size)

        try:
            samples_0 = conn_queue.get(timeout = TIME_DATA_LOADING)
            samples_1 = conn_queue.get(timeout = TIME_DATA_LOADING)

            is_not_intersect = len(samples_0) == len(samples_1) == (len(self.train_data) // 2) \
                        and (len(samples_0 & samples_1) == 0)
        except Empty:
            context.join(0)
        
        while context.join(TIME_DATA_LOADING):
             pass

        self.assertTrue(is_not_intersect)
