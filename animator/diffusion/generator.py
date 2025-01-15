import torch
from diffusers import DDPMScheduler
from torch import nn, Tensor
from transformers import AutoTokenizer, CLIPTextModel

from animator.diffusion import LoRaUNet2DConditionModel, SCAutoencoderKL
from animator.utils.parameter_storages.diffusion_parameters import GeneratorParams


class GANTurboGenerator(nn.Module):
    """Combine models to form Cycle GAN Turbo Generator."""

    def __init__(self, caption: str, params: GeneratorParams, device: str | torch.device = "cpu") -> None:
        """Load pretrained models and DDPMScheduler."""
        super().__init__()
        self.caption = caption
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        tokens = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        self.caption_enc = text_encoder(tokens)[0].detach().clone()

        self.noise_scheduler_1step = DDPMScheduler.from_pretrained(
            "stabilityai/sd-turbo", subfolder="scheduler"
        )
        # TODO trained model loading
        self.unet = LoRaUNet2DConditionModel(params.unet_lora_rank)
        self.vae = SCAutoencoderKL(rank=params.vae_lora_rank, gamma=params.gamma)
        self.device = device
        self.vae.eval()
        self.unet.eval()

    @torch.no_grad()
    def _random_fowrard(
        self,
    ) -> Tensor:
        """Forward method with random input(for pretrained model evaluation only)."""
        self.noise_scheduler_1step.set_timesteps(1, device=self.device)
        x = torch.randn(1, 4, 64, 64, device=self.device)
        for time in self.noise_scheduler_1step.timesteps:
            noise = self.unet(x, time, encoder_hidden_states=self.caption_enc).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample
        x = self.vae.decode(x, [])
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward with one step DDPMScheduler."""
        self.noise_scheduler_1step.set_timesteps(1, device=self.device)
        x, down_skip = self.vae.encode(x)
        for time in self.noise_scheduler_1step.timesteps:
            noise = self.unet(x, time, encoder_hidden_states=self.caption_enc).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample

        x = self.vae.decode(x, down_skip)
        return x
