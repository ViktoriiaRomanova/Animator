import unittest
import os

import torch
from torchvision.transforms import (Compose,
                                    RandomHorizontalFlip,
                                    Resize,
                                    Normalize)
from torchvision import io

from animator.utils.preprocessing_data import PreprocessingData
from animator.style_transfer.get_dataset import GetDataset
from tests import DATA_PATH


class GetDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_shape = (3, 224, 224)
        pr_data = PreprocessingData(train_size = 0.8)
        cls.path = os.path.join(DATA_PATH, 'domainX')
        cls.train_img, cls.test_img = pr_data.get_data(cls.path, 10, 0.5)
        cls.mean = torch.tensor((0.485, 0.456, 0.406))
        cls.std = torch.tensor((0.229, 0.224, 0.225))


    @classmethod
    def tearDownClass(cls) -> None:
        del cls.train_img, cls.test_img

    def test_getitem_right_size(self) -> None:
        dataset = GetDataset(self.path, self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)

        im_size = dataset[0].shape

        self.assertEqual(im_size, self.img_shape)
    
    def test_getitem_norm(self) -> None:
        dataset = GetDataset(self.path, self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)
        norm = Compose([Resize(self.img_shape[1:]), 
                       Normalize(self.mean, self.std)])
        original_img = norm(io.read_image(os.path.join(self.path, self.train_img[0])).div(255))
        
        img = dataset[0]
        mean, std = img.mean(dim = (1, 2)), img.std(dim = (1, 2))
        or_mean, or_std = original_img.mean((1,2)), original_img.std((1, 2))

        self.assertTrue(torch.allclose(mean, or_mean, rtol = 1e-3, atol = 1e-3) and
                        torch.allclose(std, or_std, rtol = 1e-3, atol = 1e-3))


    def test_getitem_transformedTrue(self) -> None:
        transform = RandomHorizontalFlip(1)
        tranformed_dataset = GetDataset(self.path, self.train_img, 
                                        size = self.img_shape[1:],
                                        mean = self.mean,
                                        std = self.std,
                                        transform = transform)
        dataset = GetDataset(self.path, self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)

        img = dataset[0]
        tr_img = tranformed_dataset[0]

        self.assertTrue(torch.allclose(transform(img), tr_img))
