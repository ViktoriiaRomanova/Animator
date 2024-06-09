import os
import torch.multiprocessing as mp

from animator.utils.preprocessing_data import PreprocessingData
from animator.utils.parameter_storages.params_holder import ParamsHolder
from train_eval.figure_extraction.worker import worker
from animator.figure_extraction.get_dataset import checker

HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'

if __name__ == '__main__':
    os.system('nvidia-smi')
    os.system('conda info --envs')

    params_holder = ParamsHolder(HYPERPARAMETERS, ptype = 'Extraction')
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params       

    pr_data = PreprocessingData(params.data.data_part, checker = checker)
    train_data, val_data = pr_data.get_data(os.path.join(base_param.dataset, 'images'),
                                              params.main.random_state,
                                              params.data.sub_part_data)      

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
