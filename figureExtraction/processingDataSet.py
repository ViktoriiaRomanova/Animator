import os
import random
import re
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io
from tqdm import tqdm
import time
import pandas as pd

__all__ = [
            'PreprocessingData',
            'MaskDataset',
            'get_not_RGB_pic',
            'conv_to_img'
]

DataType = Tuple[List[str], List[str]]


class PreprocessingData:
    """Collect file names, split into train and test."""

    def __init__(self, train_size: float = 0.95) -> None:
        self.train_size = train_size

    def get_data(self, data_path: str, random_state: int, part: float = 1.0) -> DataType:
        """
            Collect data.

            Splits into train and test/validation.
        """
        train_img, test_img = [], []  # list[image_name]
        images_path = os.path.join(data_path, 'input')
        filenames = []
        for name_ in os.listdir(images_path):
            if name_.endswith('.jpg') and re.match('\d_+\d', name_) is not None:
                filenames.append(name_)
           
        # From each class we take some of data to train and other to test
        train_img, test_img = train_test_split(filenames, train_size = self.train_size,
                                       shuffle = True, random_state = random_state)
        train_len, test_len = int(part * len(train_img)), int(part * len(test_img))
        return train_img[:train_len], test_img[:test_len]


class MaskDataset(Dataset):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: List[str],
                 transform: Union[nn.Module, transforms.Compose, None] = None, device = None) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        self.imgnames = data  # list[image_name]
        self.img_dir = img_dir
        self.transforms = transform
        self.to_resized_tensor = transforms.Compose([
            transforms.Resize([224, 224], antialias=True)])
        # The same parameters that were used for obtaining VGG16 weights (are used for the initial encoder parameters)
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.device = device

    def __len__(self) -> int:
        """Return number of pictures in dataset."""
        return len(self.imgnames)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """Return image/transformed image and it's mask by given index."""
        worker_info = torch.utils.data.get_worker_info()
        #with open('test_res.txt','a') as f:
            #f.write(str(worker_info.id))
        img_path = os.path.join(self.img_dir, os.path.join('input', self.imgnames[idx]))
        mask_path = os.path.join(self.img_dir, 
                                 os.path.join('Output', self.imgnames[idx].split('.')[0] + '.png')) # masks stored in 'png' format and images in 'jpg' format
        
        start = time.time()
        image = io.read_image(img_path)
        mask = io.read_image(mask_path)
        frame = pd.DataFrame({'device:': [self.device], 'time:': [time.time() - start], 'id: ': [worker_info.id]})
        frame.to_csv('open_time10.csv', mode = 'a')
        image = self.norm(self.to_resized_tensor(image).div(255))
        mask = self.to_resized_tensor(mask).div(255)

        if self.transforms:
            both = torch.cat((image, mask), dim = 0)
            image = self.transforms(both)
            image, mask = torch.tensor_split(both, [3], dim = 0)
        return image, mask


def get_not_RGB_pic(data: MaskDataset) -> Set[int]:
    """Get indexes of pictures which is not RGB."""
    indexes = set()
    for i in tqdm(range(len(data))):
        img_path = os.path.join(data.img_dir, os.path.join('input', data.imgnames[i]))
        img = Image.open(img_path)
        if not img.mode == 'RGB':
            indexes.add(i)
    return indexes


def conv_to_img(tensor: torch.tensor) -> np.array:
    """Convert image to display by pyplot."""
    img = tensor.to('cpu').clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    return img
