import numpy as np
import torch
from torch import nn
from ._base_img_processing import BaseImgProcessing
from typing import Callable

class ModelImgProcessing(BaseImgProcessing):
    def __init__(self, model: nn.Module , path: str, mode: str = 'mask', transform: Callable[[np.array], np.array] | None = None) -> None:
        super().__init__()
        self.model = model
        self.transform = transform
        state = torch.load(path, map_location = torch.device('cpu'))['model']
        self.model.load_state_dict(state)
        self.model.eval()
        self.mode = mode
    
    def __call__(self, img: torch.Tensor) -> np.array:
        img = img.unsqueeze(0)
        with torch.no_grad():
            if self.mode == 'mask':
                img = self._conv_to_img(self.model(img))
            elif self.mode == 'prune':
                img = self._conv_to_img((self.model(img) > 0.5).byte() * img, self.transform)
            else:
                raise NotImplementedError(
                    'Mode type "{}" is not found, available modes are "mask" and "prune"'.format(self.mode)
                    )
        return img

class ImgProcessing(BaseImgProcessing):
    def __init__(self, transform: Callable[[np.array], np.array] | None = None) -> None:
        super().__init__()
        self.transform = transform
    
    def __call__(self, img: torch.Tensor) -> np.array:
        return self._conv_to_img(img, self.transform)
