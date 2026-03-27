from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
import matplotlib.pyplot as plt

#pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo") #, torch_dtype=torch.float16, variant="fp16")
#pipe.to("cuda")

#init_image = load_image("/home/viktoriia/Pictures/tmp/my_photo.jpg").resize((512, 512))
#prompt = "anime character, anime persona"

#image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.8, guidance_scale=0.0).images[0]

#plt.imshow(image)
#plt.show()
from animator.diffusion import SCAutoencoderKL, LoRaUNet2DConditionModel

vae = SCAutoencoderKL()
unet = LoRaUNet2DConditionModel(32)
print(unet.state_dict().keys())
#searched = {'vae.base_model.model.decoder.skip.0.base_layer.weight', 'vae.base_model.model.decoder.skip.0.lora_A.default.weight', 'vae.base_model.model.decoder.skip.0.lora_B.default.weight'}
#for name, par in vae.named_parameters():
#    if 'skip' in name:
#        print(name, par.requires_grad)

