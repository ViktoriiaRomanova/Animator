import os

from torch.utils.data import DataLoader, Dataset
from torch import tensor
from torchvision import transforms, io

from animator.figure_extraction.get_dataset import get_data

DATA_PATH = '/home/viktoriia/Downloads/segmentation/'
HYPERPARAMETERS = 'train_eval/figure_extraction/hyperparameters.yaml'
MODEL_WEIGHTS = 'train_eval/figure_extraction/train_checkpoints/2024_06_18_09_21_24/99.pt'
RESULT_PATH = '/home/viktoriia/Downloads/transfer/'

class ModifyDataset(Dataset):
    """Prepare data for DataLoader to load in trained model."""

    def __init__(self, img_dir: str, data: list[str]) -> None:
        """
            Args:
                * dataset directory,
                * list of filenames,
        """
        self.img_dir = img_dir
        self.imgnames = data
        self.size = (512, 512)

    def __len__(self,):
        return len(self.imgnames)


    def __getitem__(self, idx: int) -> tensor:
        """Return image/transformed image by given index."""
        img_path = os.path.join(os.path.join(self.img_dir, 'images'), self.imgnames[idx])
        mask_path = os.path.join(os.path.join(self.img_dir, 'masks'), self.imgnames[idx])
        image = io.read_image(img_path)
        mask = (io.read_image(mask_path, mode = io.ImageReadMode.GRAY) > 0).byte()

        image = transforms.functional.center_crop(mask * image, output_size = max(image.shape))
        image = transforms.functional.resize(image, size = self.size)

        return image


if __name__ == '__main__':
        filenames = get_data(os.path.join(DATA_PATH, 'images'))
        dataset = ModifyDataset(DATA_PATH, filenames)
        dataloader = DataLoader(dataset, batch_size = 8, shuffle = False, num_workers = 4, drop_last = False)

        for i, batch in enumerate(dataloader):
               for j, img in enumerate(batch):
                        io.write_png(img, os.path.join(RESULT_PATH, filenames[8 * i + j]), compression_level = 0)