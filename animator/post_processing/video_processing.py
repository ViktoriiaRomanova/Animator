from argparse import ArgumentParser

import torch
import yaml

from animator.post_processing.prepare_data import PostProcessingVideo
from animator.style_transfer.cycle_gan_model import Generator
from animator.utils.img_processing import ModelImgProcessing
from animator.utils.parameter_storages.transfer_parameters import TrainingParams


def video_transform(video_path: str,
                    weights_path: str,
                    results_folder: str,
                    hyperparam_path: str,
                    start: float = 0.0,
                    length: float | None = None) -> None:
    """Transform video using trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(hyperparam_path, 'r') as file:
        data_transform = TrainingParams(**yaml.safe_load(file)).data

    def img_transformation(img: torch.Tensor) -> torch.Tensor:
        img = img.permute((0, 2, 3, 1))
        img = img * torch.tensor(data_transform.std) + torch.tensor(data_transform.mean)
        img = img.permute((0, 3, 1, 2))
        return img

    img_processor = ModelImgProcessing(Generator(),
                                       'genA',
                                       weights_path,
                                       mode = 'simple',
                                       transform = img_transformation,
                                       device = device)

    video_transformer = PostProcessingVideo(img_processor,
                                            data_transform.size[0],
                                            data_transform.mean,
                                            data_transform.std)

    video_transformer.apply(video_path,
                            results_folder, start, length)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pvideo', required=True)
    parser.add_argument('--pmodel', required=True)
    parser.add_argument('--pres', required=True)
    parser.add_argument('--hyperp', required=True)
    parser.add_argument('--start', default = 0.0, required = False)
    parser.add_argument('--length',  default = None, required = False)

    args = parser.parse_args()
    video_transform(args.pvideo, args.pmodel, args.pres, args.hyperp, args.start, args.length)
