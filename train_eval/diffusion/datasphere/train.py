import os

import torch.multiprocessing as mp

from animator.utils.preprocessing_data import PreprocessingData
from animator.utils.parameter_storages.params_holder import ParamsHolder
from worker import worker


if __name__ == "__main__":
    
    params_holder = ParamsHolder("Diffusion")
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params

    pr_data = PreprocessingData(params.data.data_part, lambda x: True)

    datasetX = os.path.join(base_param.dataset, 'domainX/')
    datasetY = os.path.join(base_param.dataset, 'domainY/')
    
    train_dataX, val_dataX = pr_data.get_data(datasetX,
                                              params.main.random_state,
                                              params.data.sub_part_data)
        
    train_dataY, val_dataY = pr_data.get_data(datasetY,
                                              params.main.random_state,
                                              params.data.sub_part_data)

    train_data = [train_dataX, train_dataY]
    val_data = [val_dataX, val_dataY]

    # Set s3 storage folder to store cache
    os.environ['HF_HOME'] = base_param.st + '/cache/'
    os.environ['TORCH_HOME'] = base_param.st + '/cache/'

    mp.spawn(worker, args = (base_param, params, train_data, val_data),
                             nprocs = params.distributed.world_size)
