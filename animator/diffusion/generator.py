from warnings import warn

import torch
from diffusers import DDPMScheduler
from torch import nn, Tensor
from transformers import AutoTokenizer, CLIPTextModel

from animator.diffusion import LoRaUNet2DConditionModel, SCAutoencoderKL

class GANTurboGenerator(nn.Module):
    def __init__(self, caption: str, device: str | torch.device = "cpu"):
        super().__init__()
        self.caption = caption
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        tokens = tokenizer(caption,
                                max_length=tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                                ).input_ids
        self.caption_enc = text_encoder(tokens)[0].detach().clone()
        
        self.noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
        # TODO add model parameters holder
        # TODO trained model loading
        self.unet = LoRaUNet2DConditionModel(8)
        self.vae = SCAutoencoderKL(rank=4, gamma=1)
        self.device = device
        self.vae.eval()
        self.unet.eval()

    @torch.no_grad()
    def _random_fowrard(self,) -> Tensor:
        self.noise_scheduler_1step.set_timesteps(1, device=self.device)
        x = torch.randn(1, 4, 64, 64, device=self.device)
        for time in self.noise_scheduler_1step.timesteps:
            noise = self.unet(x, time, encoder_hidden_states=self.caption_enc).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample
        x = self.vae.decode(x, [])
        return x
        
    def forward(self, x: Tensor) -> Tensor:
        self.noise_scheduler_1step.set_timesteps(1, device=self.device)
        x, down_skip = self.vae.encode(x)
        for time in self.noise_scheduler_1step.timesteps:
            noise = self.unet(x, time, encoder_hidden_states=self.caption_enc).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample
        
        x = self.vae.decode(x, down_skip)
        return x
