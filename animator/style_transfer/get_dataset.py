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

    def __init__(self, img_dir: str, data: list[str],
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None,
                 rand_mode: bool = False) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)
        self.rand_mode = rand_mode

    def __getitem__(self, idx: int) -> Tensor:
        """Return image/transformed image by given index."""
        if self.rand_mode:
            idx = (randint(0, len(self.imgnames) - 1)) % len(self.imgnames)
        img_path = os.path.join(self.img_dir, self.imgnames[idx])
        
        image = io.read_image(img_path, mode = io.ImageReadMode.RGB)
        #image = self.norm(self.to_resized_tensor(image).div(255))
        image = self.norm(image.div(255))

        return image if self.transforms is None else self.transforms(image)
