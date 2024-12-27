from diffusers import AutoencoderKL
from torch import nn, Generator, Tensor
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from huggingface_hub import PyTorchModelHubMixin
from warnings import warn


def encoder_forward(self, x: Tensor) -> Tensor:
    self.down_skip = []
    x = self.conv_in(x)
    for down_block in self.down_blocks:
        self.down_skip.append(x)
        x = down_block(x)
    x = self.mid_block(x)
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x


def decoder_forward(self, x: Tensor, latent_embeds: Tensor | None = None) -> Tensor:
    x = self.conv_in(x)
    x = self.mid_block(x)
    len_skip = len(self.incoming_skip) - 1
    if len_skip == -1:
        warn("Missing skip connection data")
        for ind, up_block in enumerate(self.up_blocks):
            x = up_block(x, latent_embeds)
    else:
        for ind, up_block in enumerate(self.up_blocks):
            up_skip = self.skip[ind](self.incoming_skip[len_skip - ind] * self.gamma)
            x = up_block(x + up_skip, latent_embeds)
    if latent_embeds is None:
        x = self.conv_norm_out(x)
    else:
        x = self.conv_norm_out(x, latent_embeds)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x


class SCAutoencoderKL(nn.Module, PyTorchModelHubMixin):

    def __init__(self, gamma: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        self.vae.encoder.forward = encoder_forward.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = decoder_forward.__get__(self.vae.decoder, self.vae.decoder.__class__)
        self.vae.encoder.down_skip = []
        self.vae.decoder.incoming_skip = []
        self.vae.decoder.gamma = gamma
        self.vae.decoder.skip = nn.ModuleList(
            [
                nn.Conv2d(512, 512, 1, 1, bias=False),
                nn.Conv2d(256, 512, 1, 1, bias=False),
                nn.Conv2d(128, 512, 1, 1, bias=False),
                nn.Conv2d(128, 256, 1, 1, bias=False),
            ]
        )

    def decode(
        self,
        x: Tensor, *args, **kwargs
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        return self.vae.decode(x, *args, **kwargs)

    def forward(
        self,
        x: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Generator | None = None,
    ) -> DecoderOutput | Tensor:

        posterior = self.encode(x).latent_dist
        self.decoder.incoming_skip = self.encoder.down_skip
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
