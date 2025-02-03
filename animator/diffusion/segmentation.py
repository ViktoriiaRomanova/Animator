import torch
from torchvision.transforms import Normalize

from ..figure_extraction.unet_model import UNet


class SegmentCharacter:
    def __init__(
        self, path: str, model_type: str, device: torch.device, mean: tuple[float], std: tuple[float]
    ) -> None:
        self.modifier = UNet(model_type).to(device)
        state = torch.load(path, map_location=device, weights_only=True)["model"]
        self.modifier.load_state_dict(state)
        self.modifier.eval()
        self.modifier.requires_grad_(False)
        unet_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        unet_std = torch.tensor([0.229, 0.224, 0.225], device=device)

        cur_model_mean = torch.tensor(mean, device=device)
        cur_model_std = torch.tensor(std, device=device)

        inv_model_std = 1 / cur_model_std
        inv_model_mean = -cur_model_mean * inv_model_std

        self.norm = torch.nn.Sequential(
            Normalize(inv_model_mean, inv_model_std, inplace=False), Normalize(unet_mean, unet_std)
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        data_type = x.dtype
        mask = (self.modifier(self.norm(x.detach().clone())) > 0).type(data_type)
        return mask * x
