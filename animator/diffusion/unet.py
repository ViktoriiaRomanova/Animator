import deepspeed
from diffusers import UNet2DConditionModel
from peft import get_peft_model, LoraConfig
from torch import nn, Tensor, utils, cuda


class CheckpointedSubModule(nn.Module):
    def __init__(self, sub_module: nn.Module, name: str) -> None:
        super().__init__()
        self.sub_module = sub_module
        self.name = name

    def forward(self, *args, **kwargs) -> Tensor:
        #return self.check_memory(self.sub_module, *args, **kwargs)
        return utils.checkpoint.checkpoint(self.sub_module, *args, use_reentrant=False, **kwargs)

    def __getattr__(self, name):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sub_module, name)

    def check_memory(self, block, *args, **kwargs):
        cuda.reset_peak_memory_stats()
        mem_before = cuda.memory_allocated()
        # print("input shape_args:", [x.shape for x in args])
        # print("input shape_rwargs:", [(n, x.shape if type(x) == Tensor else x) for n, x in kwargs.items()])
        x = utils.checkpoint.checkpoint(block, *args, **kwargs, use_reentrant=False)
        peak = cuda.max_memory_allocated()
        mem_after = cuda.memory_allocated()
        print(
            f"[{self.name}] Forward mem delta: {(mem_after - mem_before)/1e6:.1f} MB, peak: {peak/1e6:.1f} MB"
        )
        return x


class LoRaUNet2DConditionModel(nn.Module):
    """Add LoRa to pretrained UNet."""

    def __init__(self, rank: int, *args, **kwargs) -> None:
        """Create UNet model with LoRa adapters and a trainable first layer."""
        super().__init__()
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
        
        #module = self.unet.get_submodule("base_model.model.mid_block")
        # sub_module = CheckpointedSubModule(module, "mid_block")
        # self.unet.set_submodule("base_model.model.mid_block", sub_module)

        #for ind, module in enumerate(self.unet.get_submodule("base_model.model.down_blocks")):
        #    sub_module = CheckpointedSubModule(module, f"down_block{ind}")
        #    self.unet.set_submodule("base_model.model.down_blocks.{}".format(ind), sub_module)
        #    if ind == 2: break
        #for ind, module in enumerate(self.unet.get_submodule("base_model.model.up_blocks")):
        #    sub_module = CheckpointedSubModule(module, f"up_block{ind}")
        #    self.unet.set_submodule("base_model.model.up_blocks.{}".format(ind), sub_module)
        #    if ind == 2: break
    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)
