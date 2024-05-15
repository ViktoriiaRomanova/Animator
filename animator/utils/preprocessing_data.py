import os
from typing import Callable

from sklearn.model_selection import train_test_split

from ..utils import _base_preprocessing_data as _bp

__all__ = ['PreprocessingData']

class PreprocessingData(_bp.BasePreprocessingData):
    """Collect file names, split into train and test."""

    def __init__(self, train_size: float = 0.95,
                 checker: Callable[[str], bool] | None = None) -> None:
        super().__init__(train_size)
        self.checker = checker if checker is not None else self.check       

    def get_data(self, data_path: str, random_state: int, part: float = 1.0) -> _bp.DataType:
        """
            Collect data.

            Splits into train and test/validation.
        """
        train_img, test_img = [], []  # list[image_name]
        filenames = []
        for name_ in os.listdir(data_path):
            if self.checker(name_):
                filenames.append(name_)
           
        # From each class we take some of data to train and other to test
        train_img, test_img = train_test_split(filenames, train_size = self.train_size,
                                       shuffle = False, random_state = random_state)
        train_len, test_len = int(part * len(train_img)), int(part * len(test_img))
        return train_img[:train_len], test_img[:test_len]
