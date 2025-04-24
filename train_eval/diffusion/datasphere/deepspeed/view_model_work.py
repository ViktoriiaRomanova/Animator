import matplotlib.pyplot as plt
import yaml
from torch import tensor
from torch.utils.data import DataLoader

from animator.figure_extraction.get_dataset import get_data
from animator.diffusion.generator import GANTurboGenerator
from animator.post_processing.prepare_data import PostProcessingDataset
from animator.utils.img_processing import ImgProcessing, ModelImgProcessing
from animator.utils.parameter_storages.diffusion_parameters import DiffusionTrainingParams

HYPERPARAMETERS = "train_eval/diffusion/datasphere/deepspeed/hyperparameters.yaml"
MODEL1_WEIGHTS = "train_eval/diffusion/datasphere/deepspeed/train_checkpoints/2025_04_23_17_48/pytorch_model.bin"
IMG_PATH = "datasets/diffusion/domainX"


if __name__ == "__main__":
    with open(HYPERPARAMETERS, "r") as file:
        params = DiffusionTrainingParams(**yaml.safe_load(file))

    data_transform = params.data

    batch_size = 1  # keep batch_size == 1 for different size images
    # dataloader can't combine images of different size
    names = get_data(IMG_PATH)[20:25]
    imges = PostProcessingDataset(
        IMG_PATH, names, data_transform.size, data_transform.mean, data_transform.std
    )
    dataloader = DataLoader(
        imges, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False  # len(names),
    )

    def img_transformation(img: tensor) -> tensor:
        img = img.permute((0, 2, 3, 1))
        img = img * tensor(data_transform.std) + tensor(data_transform.mean)
        return img

    model_based_img_processor1 = ModelImgProcessing(
        GANTurboGenerator(params.main.caption_forward, params.generator),
        "genA",
        MODEL1_WEIGHTS,
        mode="simple",
        transform=img_transformation,
    )
    img_processor = ImgProcessing(img_transformation)

    for loaded_img in dataloader:
        fig, axs = plt.subplots(min(len(imges), batch_size), 2, squeeze=False)
        for ax, prev_im, res_im1 in zip(
            axs, img_processor(loaded_img), model_based_img_processor1(loaded_img)
        ):
            ax[0].axis("off")
            ax[1].axis("off")
            ax[0].imshow(prev_im)
            ax[1].imshow(res_im1)
            ax[1].set_title("Anime-styled person")
        plt.show()
