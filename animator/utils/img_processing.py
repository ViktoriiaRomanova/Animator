import torch
from torch import nn
from ._base_img_processing import BaseImgProcessing
from typing import Callable

class ModelImgProcessing(BaseImgProcessing):
    def __init__(self, model: nn.Module,
                 path: str,
                 strict: bool = True,
                 model_name: str | None = None,
                 mode: str = 'mask',
                 transform: Callable[[torch.tensor], torch.tensor] | None = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self.model = model.to(device)
        self.transform = transform
        state = torch.load(path, map_location = device, weights_only = True)
        if model_name is not None:
            state = state[model_name]
        self.model.load_state_dict(state, strict=strict)
        self.model.eval()
        self.mode = mode
    
    def __call__(self, img: torch.tensor) -> torch.tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            if self.mode == 'simple':
                img = self._conv_to_img(self.model(img), self.transform)
            elif self.mode == 'prune':
                mask = self.model(img).permute((0, 2, 3, 1)) # get mask in format HxWxC
                # Add mask as alpha channel 
                img = torch.cat((self._conv_to_img(img, self.transform),mask), dim=3)
            else:
                raise NotImplementedError(
                    'Mode type "{}" is not found, available modes are "simple" and "prune"'.format(self.mode)
                    )
        return img

class ImgProcessing(BaseImgProcessing):
    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.transform = transform
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self._conv_to_img(img, self.transform)
