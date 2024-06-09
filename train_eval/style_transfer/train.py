import torch.multiprocessing as mp

from animator.utils.preprocessing_data import PreprocessingData
from animator.utils.parameter_storages.params_holder import ParamsHolder
from . import worker

HYPERPARAMETERS = 'train_eval/style_transfer/hyperparameters.yaml'

if __name__ == '__main__':

    params_holder = ParamsHolder(HYPERPARAMETERS)
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params

    pr_data = PreprocessingData(params.data.data_part)
    train_dataX, val_dataX = pr_data.get_data(base_param.datasetX,
                                              params.main.random_state,
                                              params.data.sub_part_data)
        
    train_dataY, val_dataY = pr_data.get_data(base_param.datasetY,
                                              params.main.random_state,
                                              params.data.sub_part_data)
    train_data = [train_dataX, train_dataY]
    val_data = [val_dataX, val_dataY]

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
