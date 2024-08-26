import yaml
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision import io

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
        img = (img * 255).to(torch.uint8)
        return img
    
    img_processor = ModelImgProcessing(Generator(), 'genA', weights_path,
                                                   mode = 'simple',
                                                   transform = img_transformation,
                                                   device = device)

    video_set = PostProcessingVideoset(video_path, 0, 2,
                                       data_transform.size,
                                       data_transform.mean,
                                       data_transform.std)
    data_loader = DataLoader(video_set, 4,False, drop_last=False)

    transformed_frames = []
    for batch in data_loader:
        batch = batch.to(device)
        transformed_frames.append(img_processor(batch))
    res= torch.cat(transformed_frames)
    print(res.shape)
    io.write_video(results_path,
                  res,
                  fps=24)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pvideo', required=True)
    parser.add_argument('--pmodel', required=True)
    parser.add_argument('--pres', required=True)
    parser.add_argument('--hyperp', required=True)
    
    args = parser.parse_args()
    video_transform(args.pvideo, args.pmodel, args.pres, args.hyperp)
