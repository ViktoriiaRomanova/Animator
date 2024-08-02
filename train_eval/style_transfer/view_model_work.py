import matplotlib.pyplot as plt
import yaml
from torch import tensor
from torch.utils.data import DataLoader

from animator.figure_extraction.get_dataset import get_data
from animator.style_transfer.cycle_gan_model import Generator
from animator.post_processing.prepare_data import PostProcessingDataset
from animator.utils.img_processing import ImgProcessing, ModelImgProcessing
from animator.utils.parameter_storages.transfer_parameters import TrainingParams

HYPERPARAMETERS = 'train_eval/style_transfer/hyperparameters.yaml'
MODEL_WEIGHTS = 'train_eval/local/train_checkpoints/199.pt'
#IMG_PATH = 'datasets/transform/domainX'
#IMG_PATH = '/home/viktoriia/Pictures/tmp/'
IMG_PATH = '/home/viktoriia/Downloads/transfer_test/domainX'


if __name__ == '__main__':
    with open(HYPERPARAMETERS, 'r') as file:
        data_transform = TrainingParams(**yaml.safe_load(file)).data
    names = get_data(IMG_PATH)[1:3]
    imges = PostProcessingDataset(IMG_PATH, names,
                                  data_transform.size,
                                  data_transform.mean,
                                  data_transform.std)
    dataloader = DataLoader(imges, batch_size = 5, #len(names),
                            shuffle = False, num_workers = 1,
                            drop_last = False)

    def img_transformation(img: tensor) -> tensor:
        img = img.permute((0, 2, 3, 1))
        img = img * tensor(data_transform.std) + tensor(data_transform.mean)
        return img

    model_based_img_processor = ModelImgProcessing(Generator(), 'genA', MODEL_WEIGHTS,
                                                   mode = 'simple',
                                                   transform = img_transformation)
    img_processor = ImgProcessing(img_transformation)

    for loaded_img in dataloader:
        fig, axs = plt.subplots(min(len(imges), 5), 2, squeeze = False)
        for ax, prev_im, res_im in zip(axs, img_processor(loaded_img), model_based_img_processor(loaded_img)):
            ax[0].axis('off')
            ax[1].axis('off')
            ax[0].imshow(prev_im)
            ax[1].imshow(res_im)
        plt.show()
