import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

from ..utils import _base_preprocessing_data as _bp

__all__ = ['GetDataset']

class UnpairedDataset(Dataset, _bp.BaseDataset):
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
        # TODO add the rest part of the dataset preparation.