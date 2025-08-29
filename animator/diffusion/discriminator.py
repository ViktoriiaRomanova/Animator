from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision.transforms import Normalize

from animator.diffusion.blurpool import BlurPool
from animator.utils.DiffAugment_pytorch import DiffAugment


class MultiLevelDViT(nn.Module):
    def __init__(self, level: int = 3, in_ch1: int = 768, in_ch2: int = 512, out_ch: int = 256) -> None:
        super().__init__()
        self.decoder = nn.ModuleList()
        self.level = level
        for _ in range(level - 1):
            self.decoder.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_ch1, out_ch, kernel_size=3, stride=1, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    BlurPool(out_ch, pad_type="zero", stride=1),
                    spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)),
                )
            )
        self.decoder.append(
            nn.Sequential(
                spectral_norm(nn.Linear(in_ch2, out_ch)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Linear(out_ch, 1)),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Discriminator(nn.Module):
    def __init__(
        self,
        cv_type: str,
        output_type: str = "conv_multi_level",
        loss_type: str | None = None,
        diffaug: bool = True,
        **kwargs
    ) -> None:

        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.decoder = MultiLevelDViT()

        self.image_processor.do_center_crop = False
        self.image_processor.do_convert_rgb = False
        self.image_processor.do_rescale = False
        self.diffaug_policy = "color,translation,cutout"

    def train(self, mode: bool = True):
        self.clip_model.train(False)
        self.decoder.train(mode)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.image_processor.preprocess(x)
        output = self.clip_model(x, output_hidden_states=True)
