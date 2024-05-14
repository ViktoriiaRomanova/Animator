from animator.utils.img_processing import ModelImgProcessing, ImgProcessing
from animator.figure_extraction.UNet_model import UNet
from animator.style_transfer.preprocessing_data import GetDataset
import matplotlib.pyplot as plt
import numpy as np
import yaml
from animator.utils.parameter_storages import TrainingParams

HYPERPARAMETERS = 'train_eval/style_transfer/hyperparameters.yaml'

MODEL_WEIGHTS = 'animator/figure_extraction/train_checkpoints/2024_03_06_15_33_53/1.pt'
IMG_PATH = 'tests/style_transfer/test_img/'

if __name__ == '__main__':
    with open(HYPERPARAMETERS, 'r') as file:
        data_transform = TrainingParams(**yaml.safe_load(file)).data

    imges = GetDataset(IMG_PATH, ['my_photo.jpg', 'my_photo.jpg'])

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

