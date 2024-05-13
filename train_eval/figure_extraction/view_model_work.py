from animator.utils.img_processing import ImgProcessing
from animator.figure_extraction.UNet_model import UNet
from animator.style_transfer.preprocessing_data import GetDataset
import matplotlib.pyplot as plt
import numpy as np

MODEL_WEIGHTS = 'animator/figure_extraction/train_checkpoints/2024_03_06_15_33_53/1.pt'
IMG_PATH = 'tests/style_transfer/test_img/'

if __name__ == '__main__':
    img_processor = ImgProcessing(UNet(), MODEL_WEIGHTS)
    imges = GetDataset(IMG_PATH, ['my_photo.jpg'])
    fig, axs = plt.subplots(1, 2)
    img = imges[0].clone().numpy().transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    axs[0].imshow(img)
    axs[1].imshow(img_processor(imges[0]))
    plt.show()