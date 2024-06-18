import os
import yaml

import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor

from animator.post_processing.prepare_data import PostProcessingDataset
from animator.figure_extraction.get_dataset import checker
from animator.utils.parameter_storages.extraction_parameters import ExtTrainingParams
from animator.utils.img_processing import ModelImgProcessing
from animator.figure_extraction.unet_model import UNet

DATA_PATH = '/home/viktoriia/Downloads/segmentation/images'
HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'
MODEL_WEIGHTS = 'train_eval/figure_extraction/train_checkpoints/2024_06_18_09_21_24/99.pt'

def get_data(data_path: str, checker = checker) -> list[str]:
        """Collect data."""
        filenames = []
        for name_ in os.listdir(data_path):
            if checker(name_):
                filenames.append(name_)

        return filenames


if __name__ == '__main__':
        with open(HYPERPARAMETERS, 'r') as file:
                data_transform = ExtTrainingParams(**yaml.safe_load(file)).data

        dataset = PostProcessingDataset(DATA_PATH, get_data(DATA_PATH), data_transform.size, data_transform.mean, data_transform.std)
        dataloader = DataLoader(dataset, batch_size = 8, shuffle = False, num_workers = 4)

        def img_transformation(img: Tensor) -> Tensor:
                img = img.permute((0, 2, 3, 1))
                img = img * Tensor(data_transform.std).unsqueeze(0) + Tensor(data_transform.mean).unsqueeze(0)
                return img
        
        model_based_img_processor = ModelImgProcessing(UNet(), MODEL_WEIGHTS,
                                                       mode = 'prune',
                                                       transform = img_transformation)