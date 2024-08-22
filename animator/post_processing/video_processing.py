import torch
from torchvision import io
from torchvision.transforms import CenterCrop, Normalize

from animator.style_transfer.cycle_gan_model import Generator

def video_transform(video_path: str, weights_path: str, results_path) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights_path, map_location=device)['GenA']
    model = Generator().load_state_dict(state).eval()
    frames, _, _ = io.read_video(video_path, end_pts= 14, output_format='TCHW')
    crop = CenterCrop(256, 256)
    norm = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    transformed_frames = []
    for batch in frames.split(7):
        batch = batch.to(device)
        transformed_frames.extend(model(batch))
    io.save_video(results_path,
                  torch.tensor(transformed_frames).to(memory_format=torch.channels_last),
                  fps=24)
            


