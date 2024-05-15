import torch.multiprocessing as mp

from animator.figure_extraction.processing_dataset import PreprocessingData
from animator.utils.parameter_storages.params_holder import ParamsHolder
from . import worker

HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'

if __name__ == '__main__':

    params_holder = ParamsHolder(HYPERPARAMETERS, ptype = 'Extraction')
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params

    pr_data = PreprocessingData(params.data.data_part)
    train_data, val_data = pr_data.get_data(base_param.datasetX,
                                              params.main.random_state,
                                              params.data.data_part)
        

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
