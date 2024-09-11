import os
from warnings import warn

import ffmpeg
from torch import cat, cuda, nn, rot90, Tensor, uint8
from torch.utils.data import Dataset
from torchaudio.io import StreamReader, StreamWriter
from torchvision import io
from torchvision import transforms

from animator.utils.img_processing import ModelImgProcessing
from ..utils import _base_preprocessing_data as _bp

__all__ = ['PostProcessingDataset', 'PostProcessingVideo']


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
        self.to_resized_tensor = transforms.Resize(size[0], antialias = True)

    def __getitem__(self, idx: int) -> Tensor:
        """Return image/transformed image by given index."""
        img_path = os.path.join(self.img_dir, self.imgnames[idx])

        image = io.read_image(img_path, mode = io.ImageReadMode.RGB)
        #image = transforms.functional.center_crop(image, output_size = max(image.shape))
        image = self.norm(self.to_resized_tensor(image).div(255))

        return image if self.transforms is None else self.transforms(image)


class PostProcessingVideo:
    """Stream based video transformation."""

    def __init__(self,
                 processor: ModelImgProcessing,
                 size: int,
                 mean: tuple[float, float, float],
                 std: tuple[float, float, float],
                 batch_size: int = 9,
                 ) -> None:
        """
            Prepare processing units.

            Args:
                processor - image processor based on trained model
                size - smaller edge of the image will be matched to this number
                mean - mean which was used to train model, to renormalize resulted video frames
                std - standard deviation which was used to train model, to renormalize resulted video frames
                batch_size - batch size for video frame transformation
                             (preferable if video frame rate evenly divisible by the batch size)
        """
        self._processors = [processor, self._to_uint8]

        self.to_resized_tensor = transforms.Resize(size, antialias = True)
        self.norm = transforms.Normalize(mean, std)

        self.batch_size = batch_size

    def _to_uint8(self, images: Tensor) -> Tensor:
        """Transform final tensor for video encoder."""
        return (images * 255).to(uint8)

    def forward(self, video_chunk: Tensor) -> Tensor:
        """Apply list of model based transformations and other internal tensor modifications."""
        for modifier in self._processors:
            video_chunk = modifier(video_chunk)
        return video_chunk

    def apply(self,
              video_path: str,
              results_folder: str,
              start: float = 0.0,
              length: float | None = None,
              transformation: nn.Sequential | None = None
              ) -> None:
        """
        Stream initial video, apply transformations and save the result.

        Args:
            video_path - full path to initial video
            results_folder - folder where to save transformed video
            start - timestamp of the video to start from (in seconds)
            length - desirable duration in seconds
            transformation - arbitrary transformations to apply to a chunk of frames BEFORE model.
        """
        results_path = os.path.join(results_folder,
                                    'res_' + os.path.basename(video_path))

        streamer = StreamReader(video_path)
        ind_video_stream = streamer.default_video_stream
        ind_audio_stream = streamer.default_audio_stream
        if ind_video_stream is None:
            warn('Video stream is empty, no actions will be made')
            return

        metadata = ffmpeg.probe(video_path)
        try:
            rotation = int(metadata['streams'][ind_video_stream].get('side_data_list',
                                                                     [{}, {}])[1].get('rotation', 0) // 90)
        except IndexError:
            rotation = 0

        video_info = streamer.get_src_stream_info(ind_video_stream)
        audio_info = streamer.get_src_stream_info(ind_audio_stream)
        streamer.add_basic_video_stream(frames_per_chunk = self.batch_size,
                                        buffer_chunk_size = -1,
                                        frame_rate = video_info.frame_rate,
                                        hw_accel = 'cuda' if cuda.is_available() else None)

        # Calculate audio batch size to keep it synced with the video
        # details:https://pytorch.org/audio/stable/tutorials/streamreader_basic_tutorial.html#configuring-ouptut-streams # noqa: E501
        audio_batch_size = int(self.batch_size * audio_info.sample_rate / video_info.frame_rate)
        streamer.add_basic_audio_stream(frames_per_chunk = audio_batch_size,
                                        buffer_chunk_size = -1,
                                        sample_rate = audio_info.sample_rate)
        streamer.seek(start)
        # Recalculate length in seconds into audio/video frames count
        # if it's not provided, set a total count or a bit bigger
        video_length = round(length * video_info.frame_rate) if length is not None else video_info.num_frames
        audio_length = round(length * audio_info.sample_rate if length is not None
                             else (audio_info.num_frames + 1) * audio_info.sample_rate)

        iter_ = streamer.stream()
        video_chunk, audio_chunk = next(iter_)
        video_chunk = self.forward(self._chunk_transform(video_chunk, rotation, transformation))
        height, width = video_chunk.shape[-2:]

        # Adjust the resulted size
        # to avoid an error with encoding an odd length of a video side
        if height % 2 != 0 or width % 2 != 0:
            height -= 1 if height % 2 != 0 else 0
            width -= 1 if width % 2 != 0 else 0
            video_adjusting_transform = transforms.CenterCrop((height, width))
            video_chunk = video_adjusting_transform(video_chunk)
            self._processors.append(video_adjusting_transform)

        writer = StreamWriter(results_path)
        writer.set_metadata(streamer.get_metadata())
        writer.add_video_stream(video_info.frame_rate,
                                height = height,
                                width = width,
                                hw_accel = 'cuda' if cuda.is_available() else None)

        writer.add_audio_stream(int(audio_info.sample_rate),
                                audio_info.num_channels,
                                encoder_format = 'fltp',
                                encoder = audio_info.codec)
        with writer.open():
            video_end = min(video_length, video_chunk.shape[0])
            audio_end = min(audio_length, audio_chunk.shape[0])
            writer.write_video_chunk(0, video_chunk.narrow(0, 0, video_end))
            writer.write_audio_chunk(1, audio_chunk.narrow(0, 0, audio_end))

            for video_chunk, audio_chunk in iter_:
                video_length -= self.batch_size
                audio_length -= audio_batch_size
                if video_length <= 0:
                    break
                video_end = min(video_length, video_chunk.shape[0])
                audio_end = min(audio_length, audio_chunk.shape[0])
                video_chunk = self.forward(self._chunk_transform(video_chunk.narrow(0, 0, video_end),
                                                                 rotation,
                                                                 transformation))
                writer.write_video_chunk(0, video_chunk)
                writer.write_audio_chunk(1, audio_chunk.narrow(0, 0, audio_end))

    def _chunk_transform(self, chunk: Tensor,
                         rotation: int = 0,
                         transform: nn.Module | nn.Sequential | None = None) -> Tensor:
        """Return batches ready to be fed to the model."""
        result = []
        for image in chunk:
            image = rot90(image.unsqueeze(0), rotation, [2, 3])
            image = self.norm(self.to_resized_tensor(image).div(255))
            if transform is not None:
                image = self.transforms(image)
            result.append(image)
        return cat(result)
