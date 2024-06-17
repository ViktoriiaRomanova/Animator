from animator.utils.img_processing import ModelImgProcessing, ImgProcessing
from animator.figure_extraction.unet_model import UNet
from animator.style_transfer.get_dataset import GetDataset
import matplotlib.pyplot as plt
import numpy as np
import yaml
from animator.utils.parameter_storages.extraction_parameters import ExtTrainingParams

HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'

MODEL_WEIGHTS = 'train_eval/figure_extraction/train_checkpoints/2024_06_17_12_47_12/29.pt'
IMG_PATH = 'tests/figure_extraction/test_img/images'

if __name__ == '__main__':
    with open(HYPERPARAMETERS, 'r') as file:
        data_transform = ExtTrainingParams(**yaml.safe_load(file)).data

    imges = GetDataset(IMG_PATH, ['0_00030.png', '0_00330.png', '0_00180.png', '0_00360.png', '0_00420.png', '1_00030.png','1_00150.png', '1_00240.png'], data_transform.size, data_transform.mean, data_transform.std)

    def img_transformation(img: np.array) -> np.array:
        img = img.transpose(1, 2, 0)
        img = img * np.array(data_transform.std) + np.array(data_transform.mean)
        return img

    model_based_img_processor = ModelImgProcessing(UNet(), MODEL_WEIGHTS)
    img_processor = ImgProcessing(img_transformation)

    fig, axs = plt.subplots(min(len(imges), 5), 2, squeeze = False)
    for ind, ax in enumerate(axs):
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(img_processor(imges[ind]))
        ax[1].imshow(model_based_img_processor(imges[ind]))
    plt.show()

