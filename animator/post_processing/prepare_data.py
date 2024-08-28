import os

import torch.nn as nn
from torch import Tensor, rot90
from torch.utils.data import Dataset
from torchaudio.io import StreamReader, StreamWriter
from torchvision import io
from torchvision import transforms

from ..utils import _base_preprocessing_data as _bp
from animator.utils.img_processing import ModelImgProcessing

__all__ = ['PostProcessingDataset']


class PostProcessingDataset(Dataset, _bp.BaseDataset):
    """Prepare data for DataLoader to load in trained model."""

    def __init__(self, img_dir: str, data: list[str],
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 transform: nn.Module | transforms.Compose | None = None) -> None:
        """
            Map-style Dataset.

            Args:
                img_dir - dataset directory,
                data - list of filenames,
                size - resulted size,
                mean - image mean,
                std - image standard deviation,
                transform - picture transformation.
        """
        super().__init__(img_dir, data, transform, size, mean, std)

    def __getitem__(self, idx: int) -> Tensor:
        """Return image/transformed image by given index."""
        img_path = os.path.join(self.img_dir, self.imgnames[idx])

        image = io.read_image(img_path, mode = io.ImageReadMode.RGB)
        image = transforms.functional.center_crop(image, output_size = max(image.shape))
        image = self.norm(self.to_resized_tensor(image).div(255))

        return image if self.transforms is None else self.transforms(image)


class PostProcessingVideo:
    """Prepare data for DataLoader to load in trained model."""

    def __init__(self, video_path: str,
                 results_folder: str,
                 processor: ModelImgProcessing, 
                 size: list[int, int],
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 start: int | None = None,
                 end: int | None = None,
                 rotation: int = 0,
                 batch_size: int = 8,
                 transform: nn.Module | transforms.Compose | None = None) -> None:
        """
            Map-style Dataset.

            Args:
                video_dir - video directory,
                start - start,
                size - resulted size,
                mean - image mean,
                std - image standard deviation,
                transform - frame transformation.
        """
        self.results_path = os.path.join(results_folder,
                                         os.path.join('res_', os.path.basename(video_path)))
        
        self.processor = processor

        self.to_resized_tensor = transforms.Resize(size, antialias = True)
        self.norm = transforms.Normalize(mean, std)
        self.rotation = rotation
 
        self.transforms = transform

        self.streamer = StreamReader(video_path)
        self.video_info = self.streamer.get_src_stream_info(0)
        self.audio_info = self.streamer.get_src_stream_info(1)
        self.streamer.add_video_stream(frames_per_chunk=batch_size, hw_accel='cuda')
        self.streamer.add_audio_stream(frames_per_chunk=batch_size)

        self.writer = None

    def apply(self,) -> None:
        for [video_chunk], [audio_chunk] in self.streamer.stream():
            video_chunk = self.processor(self._chunk_transform(video_chunk))
            if self.writer is None:
                self.writer = StreamWriter(self.results_path)
                self.writer.add_video_stream(self.video_info.frame_rate,
                                             video_chunk.shape[-2],
                                             video_chunk.shape[-1])



        # TODO: fix load of all video 
        self.frames, self.audio, self.metadata = io.read_video(video_path,
                                                               start_pts=start,
                                                               end_pts=end,
                                                               pts_unit='sec',
                                                               output_format='TCHW')


    def _chunk_transform(self, chunk: Tensor) -> Tensor:
        """Return image/transformed image by given index."""
        result = []
        for image in chunk:
            image = rot90(image, self.rotation, [1, 2])
            image = self.norm(self.to_resized_tensor(image).div(255))
            if self.transforms is not None:
                image = self.transforms(image)
            result.append(image)
        return torch.cat(result)
