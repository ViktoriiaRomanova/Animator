import os

import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
from torchvision import io
from torchvision import transforms

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
            Map-style Dataset.

            Args:
                img_dir - dataset directory,
                data - list of filenames,
                size - resulted size,
                mean - image mean,
                std - image standard deviation,
                transform - picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)

    def __getitem__(self, idx: int) -> tensor:
        """Return image/transformed image by given index."""
        img_path = os.path.join(self.img_dir, self.imgnames[idx])

        image = io.read_image(img_path, mode = io.ImageReadMode.RGB)
        image = transforms.functional.center_crop(image, output_size = max(image.shape))
        image = self.norm(self.to_resized_tensor(image).div(255))

        return image if self.transforms is None else self.transforms(image)


class PostProcessingVideoset(Dataset):
    """Prepare data for DataLoader to load in trained model."""

    def __init__(self, video_path: str,
                 start: int,
                 end: int,
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None) -> None:
        """
            Map-style Dataset.

            Args:
                video_dir - video directory,
                start - start,
                size - resulted size,
                mean - image mean,
                std - image standard deviation,
                transform - picture transformation.
        """
        self.to_resized_tensor = transforms.Resize(size, antialias = True)
        self.norm = transforms.Normalize(mean, std)
        self.crop = transforms.CenterCrop(size)
        self.transforms = transform
        # TODO: fix load of all video 
        self.frames, _, _ = io.read_video(video_path,
                                          start_pts=start,
                                          end_pts=end,
                                          pts_unit='sec',
                                          output_format='TCHW')
    def __len__(self,) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tensor:
        """Return image/transformed image by given index."""
        image = self.frames[idx]
        image = self.crop(self.norm(self.to_resized_tensor(image).div(255)))

        return image if self.transforms is None else self.transforms(image)
