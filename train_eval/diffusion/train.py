import torch
from matplotlib import pyplot as plt

from animator.diffusion.generator import GANTurboGenerator
from animator.utils.parameter_storages.params_holder import ParamsHolder


def img_transformation(img):
    img = img.permute((0, 2, 3, 1))
    img = img.squeeze()
    img = img * torch.tensor([0.5, 0.5, 0.5]) + torch.tensor([0.5, 0.5, 0.5])
    return img


if __name__ == "__main__":

    params_holder = ParamsHolder("Diffusion")
    base_param, params = params_holder.datasphere_params, params_holder.hyper_params
    caption = "anime style person"

    # Temporarily here
    gen = GANTurboGenerator(caption, params.models)
    x = gen._random_fowrard()
    x = img_transformation(x).detach().numpy()
    plt.imshow(x)
    plt.show()
