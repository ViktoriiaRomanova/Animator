import unittest
import re

from animator.utils.preprocessing_data import PreprocessingData
from tests.utils import DATA_PATH, DATA_PATH_WITH_GARBAGE

def checker(name: str) -> bool:
    return name.endswith('.jpg') and re.match('\d_+\d', name) is not None


class PreprocessingTests(unittest.TestCase):

    def test_get_data_clean_splited(self) -> None:
        pr_data = PreprocessingData(train_size = 0.8)
        
        train, test = pr_data.get_data(DATA_PATH, 10)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)

    def test_get_data_splited(self) -> None:
        pr_data = PreprocessingData(checker = checker, train_size = 0.8)
        
        train, test = pr_data.get_data(DATA_PATH_WITH_GARBAGE, 10)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
