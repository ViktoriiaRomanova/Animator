import os
from typing import Callable

import torch.nn as nn
from torch import Tensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

from utils import _base_preprocessing_data as _bp


class PreprocessingData(_bp.BasePreprocessingData):
    """Collect file names, split into train and test."""

    def __init__(self, checker: Callable[[str], bool] | None = None, train_size: float = 0.95) -> None:
        super().__init__(train_size)
        self.checker = checker if checker is not None else self.check       

    def get_data(self, data_path: str, random_state: int, part: float = 1.0) -> _bp.DataType:
        """
            Collect data.

            Splits into train and test/validation.
        """
        train_img, test_img = [], []  # list[image_name]
        filenames = []
        for name_ in os.listdir(data_path):
            if self.checker(name_):
                filenames.append(name_)
           
        # From each class we take some of data to train and other to test
        train_img, test_img = train_test_split(filenames, train_size = self.train_size,
                                       shuffle = False, random_state = random_state)
        train_len, test_len = int(part * len(train_img)), int(part * len(test_img))
        return train_img[:train_len], test_img[:test_len]


class GetDataset(Dataset, _bp.BaseDataset):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: list[str],
                 transform: nn.Module | transforms.Compose | None = None,
                 size: list[int, int] = [224, 224],
                 mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image/transformed image and it's mask by given index."""
        img_path = os.path.join(self.img_dir, self.imgnames[idx])
        
        image = io.read_image(img_path)
        image = self.norm(self.to_resized_tensor(image).div(255))

        return image if self.transforms is None else self.transforms(image)