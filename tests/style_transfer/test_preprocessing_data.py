import unittest
import re
import os

import torch
from torchvision.transforms import (Compose,
                                    RandomHorizontalFlip,
                                    Resize,
                                    Normalize)
from torchvision import io

from animator.style_transfer.preprocessing_data import *

def checker(name: str) -> bool:
    return name.endswith('.jpg') and re.match('\d_+\d', name) is not None

IMG_PATH = os.path.join(os.getcwd(), 'tests/style_transfer/test_img')
IMG_PATH_WITH_GARBAGE = os.path.join(os.getcwd(), 'tests/style_transfer/test_img_to_clean')

class PreprocessingTests(unittest.TestCase):

    def test_get_data_clean_splited(self) -> None:
        pr_data = PreprocessingData(train_size = 0.8)
        
        train, test = pr_data.get_data(IMG_PATH, 10)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)

    def test_get_data_splited(self) -> None:
        pr_data = PreprocessingData(checker = checker, train_size = 0.8)
        
        train, test = pr_data.get_data(IMG_PATH_WITH_GARBAGE, 10)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)



class GetDataseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_shape = (3, 224, 224)
        pr_data = PreprocessingData(train_size = 0.8)
        cls.train_img, cls.test_img = pr_data.get_data(IMG_PATH, 10, 0.5)
        cls.mean = torch.tensor((0.485, 0.456, 0.406))
        cls.std = torch.tensor((0.229, 0.224, 0.225))

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.train_img, cls.test_img

    def test_getitem_right_size(self) -> None:
        dataset = GetDataset(IMG_PATH, self.train_img)

        im_size = dataset[0].shape

        self.assertEqual(im_size, self.img_shape)
    
    def test_getitem_norm(self) -> None:
        dataset = GetDataset(IMG_PATH, self.train_img)
        norm = Compose([Resize(self.img_shape[1:]), 
                       Normalize(self.mean, self.std)])
        original_img = norm(io.read_image(os.path.join(IMG_PATH, self.train_img[0])).div(255))
        
        img = dataset[0]
        mean, std = img.mean(dim = (1, 2)), img.std(dim = (1, 2))
        or_mean, or_std = original_img.mean((1,2)), original_img.std((1, 2))

        self.assertTrue(torch.allclose(mean, or_mean, rtol = 1e-3, atol = 1e-3) and
                        torch.allclose(std, or_std, rtol = 1e-3, atol = 1e-3))


    def test_getitem_transformedTrue(self) -> None:
        transform = RandomHorizontalFlip(1)
        tranformed_dataset = GetDataset(IMG_PATH, self.train_img, transform = transform)
        dataset = GetDataset(IMG_PATH, self.train_img)

        img = dataset[0]
        tr_img = tranformed_dataset[0]

        self.assertTrue(torch.allclose(transform(img), tr_img))
