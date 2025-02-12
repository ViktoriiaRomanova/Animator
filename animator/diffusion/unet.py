from diffusers import UNet2DConditionModel
from peft import get_peft_model, LoraConfig
from torch import nn


class LoRaUNet2DConditionModel(nn.Module):
    """Add LoRa to pretrained UNet."""

    def __init__(self, rank: int, *args, **kwargs) -> None:
        """Create UNet model with LoRa adapters and a trainable first layer."""
        super().__init__(*args, **kwargs)
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        target_modules = [
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv",
            "conv_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "proj_in",
            "proj_out",
        ]
        modules_to_save = ["conv_in"]
        lora_config = LoraConfig(
            r=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        self.unet = get_peft_model(self.unet, lora_config)

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)
