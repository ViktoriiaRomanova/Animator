from warnings import warn

import torch
from torchvision.transforms import Normalize

from ..figure_extraction.unet_model import UNet


class SegmentCharacter:
    def __init__(
        self,
        path: str,
        model_type: str,
        device: torch.device,
        mean: tuple[float],
        std: tuple[float],
        warm_up: int,
    ) -> None:
        self.dtype = torch.float16
        self.modifier = UNet(model_type).to(device)
        state = torch.load(path, map_location=device, weights_only=True)["model"]
        self.modifier.load_state_dict(state)
        self.modifier.eval()
        self.modifier.requires_grad_(False)
        self.modifier.to(dtype=self.dtype)
        #self.modifier.compile()
        unet_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        unet_std = torch.tensor([0.229, 0.224, 0.225], device=device)

        cur_model_mean = torch.tensor(mean, device=device)
        cur_model_std = torch.tensor(std, device=device)

        inv_model_std = 1 / cur_model_std
        inv_model_mean = -cur_model_mean * inv_model_std

        self.norm = torch.nn.Sequential(
            Normalize(inv_model_mean, inv_model_std, inplace=False), Normalize(unet_mean, unet_std)
        )
        self.norm.to(dtype=self.dtype)
        self._warm_up = warm_up  # Better to be proportional to the number of class users.
        self.device = device

    def warm_up_update(self, delta: int) -> None:
        "Reduce/increase the amount of warm-up steps."
        self._warm_up = max(0, self._warm_up + delta)
        if self._warm_up == 0:
            warn("Segmentation is on.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._warm_up > 0:
            self.warm_up_update(-x.shape[0])
            return x
        input_device = x.device
        input_data_type = x.dtype
        x = x.to(self.device, dtype=self.dtype)
        mask = self.modifier(self.norm(x.detach().clone())) > 0
        return (mask * x).to(device=input_device, dtype=input_data_type)
