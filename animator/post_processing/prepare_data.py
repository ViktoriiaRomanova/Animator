import os


import torch.nn as nn
from torch import Tensor, rot90, cuda, uint8, cat
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

    def __init__(self,
                 processor: ModelImgProcessing, 
                 size: int,
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 batch_size: int = 8,
                 ) -> None:
        """
            Args:
                size - resulted size,
                mean - image mean,
                std - image standard deviation,
                transform - frame transformation.
        """     
        self.processor = processor

        self.to_resized_tensor = transforms.Resize(size, antialias = True)
        self.norm = transforms.Normalize(mean, std)

        self.batch_size = batch_size
 
    def apply(self, video_path: str,
                    results_folder: str,
                    rotation: int = 0,
                    interval: list[float, float] = [0.0, float('inf')],
                    transform: nn.Module | transforms.Compose | None = None) -> None:
        results_path = os.path.join(results_folder,
                                    'res_' + os.path.basename(video_path))
        
        streamer = StreamReader(video_path)
        metadata = self.streamer.get_metadata()
        video_info = self.streamer.get_src_stream_info(0)
        audio_info = self.streamer.get_src_stream_info(1)
        streamer.add_basic_video_stream(frames_per_chunk=self.batch_size, hw_accel='cuda' if cuda.is_available() else None)
        streamer.add_basic_audio_stream(frames_per_chunk=self.audio_info.num_frames)
        streamer.seek(interval[0])

        iter_ = self.streamer.stream()
        video_chunk, audio_chunk = next(iter_)
        video_chunk = self._chunk_transform(video_chunk, rotation)

        writer = StreamWriter(self.results_path)
        writer.set_metadata(self.metadata)
        writer.add_video_stream(frame_rate=self.video_info.frame_rate,
                                height=video_chunk.shape[-2],
                                width=video_chunk.shape[-1],
                                hw_accel='cuda' if cuda.is_available() else None)
        writer.add_audio_stream(int(self.audio_info.sample_rate),
                                self.audio_info.num_channels,
                                encoder_format='fltp',
                                encoder=self.audio_info.codec)
        with writer.open():
            writer.write_video_chunk(0, (self.processor(video_chunk)* 255).to(uint8))
            writer.write_audio_chunk(1, audio_chunk)

            for video_chunk, _ in iter_:
                video_chunk = self.processor(self._chunk_transform(video_chunk))
                writer.write_video_chunk(0, (video_chunk * 255).to(uint8))


    def _chunk_transform(self, chunk: Tensor,
                         rotation: int = 0,
                         transform: nn.Module | transforms.Compose | None = None) -> Tensor:
        """Return image/transformed image by given index."""
        result = []
        for image in chunk:
            image = rot90(image.unsqueeze(0), rotation, [2, 3])
            image = self.norm(self.to_resized_tensor(image).div(255))
            size = image.shape
            image = transforms.functional.center_crop(image, )
            if transform is not None:
                image = self.transforms(image)
            result.append(image)
        return cat(result)
