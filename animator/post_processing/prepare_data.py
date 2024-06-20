import os

import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

from ..utils import _base_preprocessing_data as _bp

__all__ = ['PostProcessingDataset']

class PostProcessingDataset(Dataset, _bp.BaseDataset):
    """Prepare data for DataLoader to load in trained model."""

    def __init__(self, img_dir: str, data: list[str],
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)

    def __getitem__(self, idx: int) -> tensor:
        """Return image/transformed image by given index."""
        img_path = os.path.join(self.img_dir, self.imgnames[idx])
        
        image = io.read_image(img_path)
        image = transforms.functional.center_crop(image, output_size = max(image.shape))
        image = self.norm(self.to_resized_tensor(image).div(255))

        return image if self.transforms is None else self.transforms(image)
