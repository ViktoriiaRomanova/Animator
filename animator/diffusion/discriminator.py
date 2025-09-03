from transformers import CLIPModel, CLIPImageProcessor
import torch
from torch import nn
from torch.nn.utils import spectral_norm

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

    def forward(self, x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        if len(x) != len(self.decoder):
            raise RuntimeError(
                f"Incompatible interface: core model outputs ({len(x)}) "
                f"do not match the number of the decoder layers ({self.decoder})."
            )
        output = ()
        for block, input in zip(self.decoder, x):
            output = output + (block(input).squeeze(1),)
        return output


class Discriminator(nn.Module):
    def __init__(
        self,
        cv_type: str,
        output_type: str = "conv_multi_level",
        loss_type: str | None = None,
        diffaug: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()
        if cv_type.lower() == "clip":
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = model.vision_model
            self.visual_projection = model.visual_projection
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise NotImplementedError("Incorrect core model type, use 'clip'")
        self.clip_model.requires_grad_(False)
        self.image_mean = torch.tensor(processor.image_mean).reshape(1, 3, 1, 1)
        self.image_std = torch.tensor(processor.image_std).reshape(1, 3, 1, 1)
        self.image_size = tuple([processor.size["shortest_edge"]] * 2)
        if output_type.lower() == "conv_multi_level":
            self.decoder = MultiLevelDViT()
        else:
            raise NotImplementedError("Incorrect decoder model type, use 'conv_multi_level'")

        self.diffaug_policy = "color,translation,cutout" if diffaug else ""

    def train(self, mode: bool = True):
        self.clip_model.train(False)
        self.decoder.train(mode)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=self.image_size, mode="area")
        x = DiffAugment(x, policy=self.diffaug_policy)
        x = (x - self.image_mean) / self.image_std
        output = self.clip_model(x, output_hidden_states=True)
        x = (output.hidden_states[4][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7),
             output.hidden_states[8][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7),
             self.visual_projection(output.pooler_output))
        print([i.shape for i in x])
        x = self.decoder(x)
        print([i.shape for i in x])



if __name__ == "__main__":
    #from vision_aided_loss.cvmodel import CLIP
    #model = CLIP("conv_multi_level")
    model = Discriminator("clip")
    x = torch.rand(1, 3, 512, 512)
    model(x)
