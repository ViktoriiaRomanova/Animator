import os
from random import randint

import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

from ..utils import _base_preprocessing_data as _bp

__all__ = ['GetDataset']

class GetDataset(Dataset, _bp.BaseDataset):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: list[list[str], list[str]],
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None
                ) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)
        self.to_resized_tensor = transforms.Compose([
            transforms.Resize(size[0] + 30,
                              interpolation = transforms.InterpolationMode.BICUBIC,
                              antialias = True),])
        self.img_dir_x = os.path.join(self.img_dir, 'domainX')
        self.img_dir_y = os.path.join(self.img_dir, 'domainY')

    def __len__(self,) -> int:
        """Return the number of pictures in the biggest dataset."""
        return max(len(self.imgnames[0]), len(self.imgnames[1]))

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image(transformed) by given index and random image from the smallest dataset."""
        if len(self.imgnames[0]) > len(self.imgnames[1]):
            ind_x = idx
            ind_y = randint(0, len(self.imgnames[1]) - 1)
        else:
            ind_x = randint(0, len(self.imgnames[0]) - 1)
            ind_y = idx
        img_x_path = os.path.join(self.img_dir_x, self.imgnames[0][ind_x])
        img_y_path = os.path.join(self.img_dir_y, self.imgnames[1][ind_y])
        
        image_x = io.read_image(img_x_path, mode = io.ImageReadMode.RGB)
        image_y = io.read_image(img_y_path, mode = io.ImageReadMode.RGB)
        image_x = self.crop(self.norm(self.to_resized_tensor(image_x).div(255)))
        image_y = self.crop(self.norm(self.to_resized_tensor(image_y).div(255)))

        return (image_x, image_y) if self.transforms is None \
               else (self.transforms(image_x), self.transforms(image_y))
