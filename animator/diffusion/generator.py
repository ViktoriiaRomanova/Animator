import torch
from diffusers import DDPMScheduler
from torch import nn, Tensor
from transformers import AutoTokenizer, CLIPTextModel

from animator.diffusion import LoRaUNet2DConditionModel, SCAutoencoderKL
from animator.utils.parameter_storages.diffusion_parameters import GeneratorParams


class GANTurboGenerator(nn.Module):
    """Combine models to form Cycle GAN Turbo Generator."""

    def __init__(self, caption: str, params: GeneratorParams) -> None:
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
        caption_enc = text_encoder(tokens)[0].detach().clone()
        self.register_buffer("caption_enc", caption_enc)

        self.noise_scheduler_1step = DDPMScheduler.from_pretrained(
            "stabilityai/sd-turbo", subfolder="scheduler"
        )

        self.unet = LoRaUNet2DConditionModel(params.unet_lora_rank)
        self.vae = SCAutoencoderKL(rank=params.vae_lora_rank, gamma=params.gamma)

    @torch.no_grad()
    def _random_fowrard(
        self,
    ) -> Tensor:
        """Forward method with random input(for pretrained model evaluation only)."""
        device = self.caption_enc.device
        self.noise_scheduler_1step.set_timesteps(1, device=device)
        x = torch.randn(2, 4, 64, 64, device=device)
        for time in self.noise_scheduler_1step.timesteps:
            batch_size = x.shape[0]
            hidden_states = self.caption_enc.expand(batch_size, -1, -1)
            noise = self.unet(x, time, encoder_hidden_states=hidden_states).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample
        x = self.vae.decode(x, [])
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward with one step DDPMScheduler."""
        device = self.caption_enc.device
        self.noise_scheduler_1step.set_timesteps(1, device=device)
        x = x.to(device=device)
        x, down_skip = self.vae.encode(x)
        for time in self.noise_scheduler_1step.timesteps:
            batch_size = x.shape[0]
            hidden_states = self.caption_enc.expand(batch_size, -1, -1)
            noise = self.unet(x, time, encoder_hidden_states=hidden_states).sample
            x = self.noise_scheduler_1step.step(noise, time, x, return_dict=True).prev_sample

        x = self.vae.decode(x, down_skip)
        return x


def get_trainable_params(model: nn.Module, print_num: bool = False) -> list:
    trainable_params = []
    count, tot_count = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
            count += param.numel()
        tot_count += param.numel()
    if print_num:
        print("Number of trainable parametes: {} of {}, {:.2%}".format(count, tot_count, count / tot_count))
    return trainable_params
