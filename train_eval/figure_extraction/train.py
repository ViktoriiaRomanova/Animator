import re

import torch.multiprocessing as mp

from animator.utils.preprocessing_data import PreprocessingData
from animator.utils.parameter_storages.params_holder import ParamsHolder
from . import worker

HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'

if __name__ == '__main__':

    params_holder = ParamsHolder(HYPERPARAMETERS, ptype = 'Extraction')
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params

    def checker(name_: str) -> bool:
        return name_.endswith('.jpg') and re.match('\d_+\d', name_) is not None            

    pr_data = PreprocessingData(params.data.data_part, checker = checker)
    train_data, val_data = pr_data.get_data(base_param.datasetX,
                                              params.main.random_state,
                                              params.data.data_part)        

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
