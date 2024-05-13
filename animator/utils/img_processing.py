import numpy as np
import torch
from torch import nn
from ._base_img_processing import BaseImgProcessing

class ImgProcessing(BaseImgProcessing):
    def __init__(self, model: nn.Module , path: str) -> None:
        super().__init__()
        self.model = model
        state = torch.load(path, map_location = torch.device('cpu'))['model_state_dict']
        self.model.load_state_dict(state)
        self.model.eval()
    
    def __call__(self, img: torch.Tensor) -> np.array:
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = self._conv_to_img(self.model(img))
        return img
