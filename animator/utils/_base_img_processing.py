from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Callable

import torch

class BaseImgProcessing(ABC):
    def __init__(self, transform: Callable[[np.array], np.array] | None = None) -> None:
        self.transform = transform

    def _conv_to_img(self, img: torch.tensor) -> np.array:
        """Convert image to display."""
        img = img.numpy().squeeze()
        if self.transform is not None:
            img = self.transform(img)
        img = img.clip(0, 1)
        return img
    
    @abstractmethod
    def __call__(self,) -> None:
        """Convert image to show."""
        pass
