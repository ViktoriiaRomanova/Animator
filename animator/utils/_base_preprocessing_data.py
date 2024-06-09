from abc import ABC, abstractmethod

from torchvision import transforms
from torch import nn
from torch import Tensor

DataType = tuple[list[str], list[str]]

class BasePreprocessingData(ABC):
    """Collect file names, split into train and test."""

    def __init__(self, train_size: float = 0.95) -> None:
        self.train_size = train_size

    @classmethod
    def check(self, name: str) -> bool:
        return name.endswith('.jpg') or name.endswith('.png')
    
    @abstractmethod
    def get_data(self, data_path: str, random_state: int, part: float = 1.0) -> DataType:
        """
            Collect data.

            Splits into train and test/validation.
        """
        pass



class BaseDataset(ABC):
    """Prepare data for DataLoader."""

    def __init__(self, img_dir: str, data: list[str],
                 transform: nn.Module | transforms.Compose | None,
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float]) -> None:
        """
            Store data path and all transformations.

            Args:
                * dataset directory,
                * list of filenames,
                * picture transformation.
        """
        self.imgnames = data  # list[image_name]
        self.img_dir = img_dir
        self.transforms = transform
        self.to_resized_tensor = transforms.Compose([
            transforms.Resize(size, antialias = True)])
        self.norm = transforms.Normalize(mean, std)

    def __len__(self) -> int:
        """Return number of pictures in dataset."""
        return len(self.imgnames)

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | Tensor:
        """Return image/transformed image by given index."""
        pass
