from warnings import warn

import deepspeed
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from peft import get_peft_model, LoraConfig
from torch import Generator, nn, Tensor, utils, cuda, amp


def check_memory(block, x, name, idx):
    cuda.reset_peak_memory_stats()
    mem_before = cuda.memory_allocated()
    print("input shape:", x.shape)
    x = utils.checkpoint.checkpoint(block, x, use_reentrant=False)
    peak = cuda.max_memory_allocated()
    mem_after = cuda.memory_allocated()
    print(f"[{name}{idx}] Forward mem delta: {(mem_after - mem_before)/1e6:.1f} MB, peak: {peak/1e6:.1f} MB")
    return x


def check_memory2(block, x, latent_embeds, name, idx):
    cuda.reset_peak_memory_stats()
    mem_before = cuda.memory_allocated()
    print("input shape:", x.shape)
    x = utils.checkpoint.checkpoint(block, x, latent_embeds, use_reentrant=False)
    peak = cuda.max_memory_allocated()
    mem_after = cuda.memory_allocated()
    print(f"[{name}{idx}] Forward mem delta: {(mem_after - mem_before)/1e6:.1f} MB, peak: {peak/1e6:.1f} MB")
    return x

def checkpoint_forward(module, *args):

    #def forward(*inputs):
        #return module(*inputs)
        
    #return utils.checkpoint.checkpoint(forward, *args, use_reentrant=False)
    return deepspeed.checkpointing.checkpoint(module, *args)

def encoder_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """Forward method for encoder(AutoencoderKL) with skip connections functionality."""
    down_skip = []
    x = self.conv_in(x)
    tot_size = 0
    for ind, down_block in enumerate(self.down_blocks):
        down_skip.append(x)
        print([sk.shape for sk in down_skip])
        size = x.nelement() * x.element_size() / 1024/ 1024
        tot_size += size
        print("Size Mb:", size)
        #x = utils.checkpoint.checkpoint(down_block, x, use_reentrant=False)
        x = checkpoint_forward(down_block, x)
        #x = check_memory(down_block, x, "down_block", str(ind))
        #x = down_block(x)
    #x = utils.checkpoint.checkpoint(self.mid_block, x, use_reentrant=False)
    #x = check_memory(self.mid_block, x, "mid_block_encoder", "")
    size = x.nelement() * x.element_size() / 1024/ 1024
    print("Size Mb:", size)
    tot_size += size
    x = checkpoint_forward(self.mid_block, x)
    print("Tot_size encoder: ", tot_size)
    #x = self.mid_block(x)
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x, down_skip


def decoder_forward(
    self, x: Tensor, incoming_skip: list[Tensor], latent_embeds: Tensor | None = None
) -> Tensor:
    """Forward method for decoder(AutoencoderKL) with skip connections functionality."""
    x = self.conv_in(x)
    tot_size = 0
    size = x.nelement() * x.element_size() / 1024/ 1024
    print("Size Mb:", size)
    tot_size += size
    #x = utils.checkpoint.checkpoint(self.mid_block, x, use_reentrant=False)
    x = checkpoint_forward(self.mid_block, x)
    #x = check_memory(self.mid_block, x, "mid_block_decoder", "")
    #x = self.mid_block(x)
    len_skip = len(incoming_skip) - 1
    if len_skip == -1:
        warn("Missing skip connection data")
        for up_block in self.up_blocks:
            x = utils.checkpoint.checkpoint(up_block, x, latent_embeds, use_reentrant=False)
    else:
        for ind, up_block in enumerate(self.up_blocks):
            up_skip = self.skip[ind](incoming_skip[len_skip - ind] * self.gamma)
            #x = utils.checkpoint.checkpoint(up_block, x + up_skip, latent_embeds, use_reentrant=False)
            size = x.nelement() * x.element_size() / 1024/ 1024
            print("Size Mb:", size)
            tot_size += size
            x = checkpoint_forward(up_block, x + up_skip, latent_embeds)
            #x = check_memory2(up_block, x + up_skip, latent_embeds, "up_block", str(ind))
            #x = up_block(x + up_skip, latent_embeds)
        print("Tot_size decoder: ", tot_size)
    if latent_embeds is None:
        x = self.conv_norm_out(x)
    else:
        x = self.conv_norm_out(x, latent_embeds)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x


class CheckpointedSubModule(nn.Module):
    def __init__(self, sub_module: nn.Module) -> None:
        super().__init__()
        self.sub_module = sub_module

    def forward(self, x: Tensor) -> Tensor:
        return deepspeed.checkpointing.checkpoint(self.sub_module, x)
    
    def __getattr__(self, name):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sub_module, name)

    def __getitem__(self, adapter: str) -> nn.Module:
        return self.sub_module[adapter]


class SCAutoencoderKL(nn.Module):
    """Customize pretrained AutoencoderKL."""

    def __init__(self, rank: int = 4, gamma: float = 1, *args, **kwargs) -> None:
        """
        Create castom AutoencoderKL.

        - load pretrained AutoencoderKL
        - change forward mathods
        - add skip connections
        - initialize skip connections weights
        - add LoRA adapters

        Parameters:
            rank - the inner dimension of the low-rank matrices
            gamma - grade of influence of skip connections

        """
        super().__init__(*args, **kwargs)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        self.vae.encoder.forward = encoder_forward.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = decoder_forward.__get__(self.vae.decoder, self.vae.decoder.__class__)
        self.vae.decoder.gamma = gamma
        self.vae.decoder.skip = nn.ModuleList(
            [
                nn.Conv2d(512, 512, 1, 1, bias=False),
                nn.Conv2d(256, 512, 1, 1, bias=False),
                nn.Conv2d(128, 512, 1, 1, bias=False),
                nn.Conv2d(128, 256, 1, 1, bias=False),
            ]
        )

        # Initialize skip connections with zeros
        def init_weights(sub_mod: nn.Module) -> None:
            if isinstance(sub_mod, nn.Conv2d):
                nn.init.constant_(sub_mod.weight, 1e-5)

        self.vae.decoder.skip.apply(init_weights)

        # Target modules names for Lora
        encoder_param_names = [
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
        module_names_to_keep = ["decoder.skip.0", "decoder.skip.1", "decoder.skip.2", "decoder.skip.3"]

        lora_config = LoraConfig(
            r=rank,
            init_lora_weights="gaussian",
            target_modules=encoder_param_names + module_names_to_keep,
            #modules_to_save=module_names_to_keep,
        )
        self.vae = get_peft_model(self.vae, lora_config)

        target_modules = encoder_param_names + module_names_to_keep

        """for name, module in self.vae.named_modules():
            for target_name in target_modules:
                if target_name in {"conv_in", "to_out.0", "conv_out"}: continue
                if name.endswith("." + target_name):
                    sub_module = CheckpointedSubModule(module)
                    self.vae.set_submodule(name, sub_module)
                    #sub_moduleA = CheckpointedSubModule(module.get_submodule("lora_A.default"))
                    #module.set_submodule("lora_A.default", sub_moduleA)
                    #print(name, module)

                    #sub_moduleB = CheckpointedSubModule(module.get_submodule("lora_B.default"))
                    #module.set_submodule("lora_B.default", sub_moduleB)
        print(self.vae)"""

    def decode(self, x: Tensor, incoming_skip: list[Tensor], *args, **kwargs) -> Tensor:
        """Decode rescale and sample images."""
        if self.vae.post_quant_conv is not None:
            x = self.vae.post_quant_conv(x)
        x = self.vae.decoder.forward(x / self.vae.config.scaling_factor, incoming_skip, *args, **kwargs)
        return x.clamp(-1, 1)

    def encode(
        self, x: Tensor, sample_posterior: bool = True, generator: Generator | None = None, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Encode images, sample and rescale tensor."""
        x, down_skip = self.vae.encoder.forward(x, *args, **kwargs)

        if self.vae.quant_conv is not None:
            x = self.vae.quant_conv(x)

        if sample_posterior:
            x = DiagonalGaussianDistribution(x).sample(generator=generator)
        else:
            x = DiagonalGaussianDistribution(x).mode(generator=generator)

        return x * self.vae.config.scaling_factor, down_skip

    def forward(
        self,
        x: Tensor,
        sample_posterior: bool = True,
        generator: Generator | None = None,
    ) -> Tensor:
        """Forward pass of autoencoder with skip."""
        x, down_skip = self.encode(x, sample_posterior, generator)
        x = self.decode(x, down_skip)
        return x


"""
 target_modules = encoder_param_names + module_names_to_keep

        for name, module in self.vae.named_modules():
            for target_name in target_modules:
                if target_name in {"conv_in", "to_out.0", "conv_out"}: continue
                if name.endswith("." + target_name):
                    sub_module = CheckpointedSubModule(module)
                    self.vae.set_submodule(name, sub_module)
                    sub_moduleA = CheckpointedSubModule(module.get_submodule("lora_A.default"))
                    module.set_submodule("lora_A.default", sub_moduleA)
                    #print(name, module)

                    sub_moduleB = CheckpointedSubModule(module.get_submodule("lora_B.default"))
                    module.set_submodule("lora_B.default", sub_moduleB)

"""