"""
    Prepares dataset (for domain B) for style transfer.

    Based on the trained extraction model obtain cartoon character images,
    with removed background and resized to (512, 512) RGB.
"""

import os

import yaml
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import io, transforms
from tqdm import tqdm

from animator.figure_extraction.get_dataset import get_data
from animator.figure_extraction.unet_model import UNet
from animator.post_processing.prepare_data import PostProcessingDataset
from animator.utils.img_processing import ImgProcessing, ModelImgProcessing
from animator.utils.parameter_storages.extraction_parameters import ExtTrainingParams

DATA_PATH = '/home/viktoriia/Downloads/drawing/'
HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'
MODEL_WEIGHTS = 'train_eval/figure_extraction/train_checkpoints/2024_06_18_09_21_24/99.pt'
RESULT_PATH = '/home/viktoriia/Downloads/transform/drawing'

if __name__ == '__main__':
    with open(HYPERPARAMETERS, 'r') as file:
        data_transform = ExtTrainingParams(**yaml.safe_load(file)).data
    filenames = get_data(DATA_PATH)

    # Prepares transformation for UNet model (it takes 3x224x244 images)
    input_img_resize = transforms.Resize(size = data_transform.size)

    # Makes dataset prepare images of size 3x512x512
    data_transform.size = (512, 512)

    dataset = PostProcessingDataset(DATA_PATH, filenames,
                                    data_transform.size,
                                    data_transform.mean,
                                    data_transform.std)
    data_loader = DataLoader(dataset, batch_size = 32, shuffle = False,
                             num_workers = 2, drop_last = False)

    # To increase the size of the resulting mask
    mask_resize = transforms.Resize(size = (512, 512))

    def mask_reverse_tr(img: tensor) -> tensor:
        """Prepare a mask to transform an image."""
        img = mask_resize(img)
        img = (img > 0).byte()
        return img

    def img_reverse_tr(img: tensor) -> tensor:
        """Prepare an image for saving."""
        img = img.permute((0, 2, 3, 1))
        img = img * tensor(data_transform.std) + tensor(data_transform.mean)
        img = img.permute((0, 3, 1, 2))
        img = (img * 255).byte()
        return img

    # To get mask
    model_based_img_processor = ModelImgProcessing(UNet(), MODEL_WEIGHTS,
                                                   mode = 'mask',
                                                   transform = mask_reverse_tr)
    # To get image
    img_processor = ImgProcessing(img_reverse_tr)

    for i, batch in enumerate(tqdm(data_loader)):
        if len(batch.shape) != 4:
            print(i, filenames[8 * i: 8 * (i + 1)])
            continue
        for j, (mask, img) in enumerate(zip(model_based_img_processor(input_img_resize(batch)),
                                            img_processor(batch))):
            io.write_png(mask * img, os.path.join(RESULT_PATH, '{}.png'.format(8 * i + j)),
                         compression_level = 0)
