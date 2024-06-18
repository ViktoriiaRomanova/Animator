import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import io
from torchvision import transforms
from tqdm import tqdm

from ..utils import _base_preprocessing_data as _bp

__all__ = ['MaskDataset', 'get_not_rgb_pic']


class MaskDataset(Dataset, _bp.BaseDataset):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: list[str],
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None) -> None:
        """
            Store data path and all transformations.

            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image/transformed image and it's mask by given index."""
        img_path = os.path.join(self.img_dir, os.path.join('images', self.imgnames[idx]))
        mask_path = os.path.join(self.img_dir,
                                 os.path.join('masks', self.imgnames[idx]))

        image = io.read_image(img_path)
        mask = transforms.functional.rgb_to_grayscale(io.read_image(mask_path))
        image = self.norm(self.to_resized_tensor(image).div(255))
        mask = self.to_resized_tensor(mask).div(255)

        if self.transforms:
            both = torch.cat((image, mask), dim = 0)
            both = self.transforms(both)
            image, mask = torch.tensor_split(both, [3], dim = 0)
        return image, mask

def checker(name_: str) -> bool:
    forbidden = {'ds7_pexels-photo-842569.png', 'ds7_pexels-photo-724887.png'}
    return name_.endswith('.png') and name_ not in forbidden


def get_not_rgb_pic(data: MaskDataset) -> set[int]:
    """Get indexes of pictures which is not RGB."""
    indexes = set()
    for i in tqdm(range(len(data))):
        img_path = os.path.join(data.img_dir, os.path.join('input', data.imgnames[i]))
        img = Image.open(img_path)
        if not img.mode == 'RGB':
            indexes.add(i)
    return indexes
