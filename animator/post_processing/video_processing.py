import os
import yaml
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision import io
import matplotlib.pyplot as plt

from animator.style_transfer.cycle_gan_model import Generator
from animator.utils.img_processing import ModelImgProcessing
from animator.utils.parameter_storages.transfer_parameters import TrainingParams
from animator.post_processing.prepare_data import PostProcessingVideoset

def video_transform(video_path: str, weights_path: str, results_path, hyperparam_path: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(hyperparam_path, 'r') as file:
        data_transform = TrainingParams(**yaml.safe_load(file)).data

    def img_transformation(img: torch.Tensor) -> torch.Tensor:
        img = img.permute((0, 2, 3, 1))
        img = img * torch.tensor(data_transform.std) + torch.tensor(data_transform.mean)
        img = img.permute((0, 3, 1, 2))
        return img
    
    img_processor = ModelImgProcessing(Generator(), 'genA', weights_path,
                                                   mode = 'simple',
                                                   transform = img_transformation,
                                                   device = device)

    video_set = PostProcessingVideoset(video_path, 12, 14,
                                       data_transform.size[0],
                                       data_transform.mean,
                                       data_transform.std)
    data_loader = DataLoader(video_set, 8,False, drop_last=False)

    transformed_frames = []
    for batch in data_loader:
        batch = batch.to(device)
        transformed_frames.append((img_processor(batch)* 255).to(torch.uint8))
    res= torch.cat(transformed_frames)

    io.write_video(os.path.join(results_path, video_set.res_name),
                   res,
                   #video_codec='hevc',
                   audio_codec='aac',
                   audio_array=video_set.audio,
                   fps=video_set.metadata['video_fps'],
                   audio_fps=video_set.metadata['audio_fps'])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pvideo', required=True)
    parser.add_argument('--pmodel', required=True)
    parser.add_argument('--pres', required=True)
    parser.add_argument('--hyperp', required=True)
    
    args = parser.parse_args()
    video_transform(args.pvideo, args.pmodel, args.pres, args.hyperp)
