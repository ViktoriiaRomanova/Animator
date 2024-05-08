import torch.multiprocessing as mp

from animator.style_transfer.preprocessing_data import PreprocessingData
from animator.utils.params_holder import ParamsHolder
from . import worker

HYPERPARAMETERS = 'animator/train_eval/style_transfer/hyperparameters.yaml'

if __name__ == '__main__':

    params_holder = ParamsHolder(HYPERPARAMETERS)
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params

    pr_data = PreprocessingData(params.data.data_part)
    train_dataX, val_dataX = pr_data.get_data(base_param.datasetX,
                                              params.main.random_state,
                                              params.data.data_part)
        
    train_dataY, val_dataY = pr_data.get_data(base_param.datasetY,
                                              params.main.random_state,
                                              params.data.data_part)
    train_data = [train_dataX, train_dataY]
    val_data = [val_dataX, val_dataY]

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
