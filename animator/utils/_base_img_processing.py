from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Callable

import torch

class BaseImgProcessing(ABC):
    def __init__(self,) -> None:
        pass

    def _conv_to_img(self, images: torch.tensor,
                     transform: Callable[[np.array], np.array] | None = None
                     ) -> torch.tensor:
        """Convert image to display."""
        if transform is not None:
            images = self.transform(images)
        images.clip(0, 1)
        return images
    
    @abstractmethod
    def __call__(self,) -> None:
        """Convert image to show."""
        pass
