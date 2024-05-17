import unittest
import os

import torch
from torchvision.transforms import (Compose,
                                    RandomHorizontalFlip,
                                    Resize,
                                    Normalize)
from torchvision import io

from animator.utils.preprocessing_data import PreprocessingData
from animator.figure_extraction.get_dataset import MaskDataset
from tests.figure_extraction import DATA_PATH

class GetDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_shape = (3, 224, 224)
        cls.mask_shape = (1, 224, 224)
        pr_data = PreprocessingData(train_size = 0.8)
        cls.train_img, cls.test_img = pr_data.get_data(os.path.join(DATA_PATH, 'input'), 10, 0.5)
        cls.mean = torch.tensor((0.485, 0.456, 0.406))
        cls.std = torch.tensor((0.229, 0.224, 0.225))


    @classmethod
    def tearDownClass(cls) -> None:
        del cls.train_img, cls.test_img

    def test_getitem_right_size(self) -> None:
        dataset = MaskDataset(DATA_PATH, self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)

        im_size, mask_size = map(lambda x: x.shape, dataset[0])

        self.assertEqual(im_size, self.img_shape)
        self.assertEqual(mask_size, self.mask_shape)
    
    def test_getitem_norm(self) -> None:
        dataset = MaskDataset(DATA_PATH, self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)
        norm = Compose([Resize(self.img_shape[1:]), 
                       Normalize(self.mean, self.std)])
        original_img = norm(io.read_image(os.path.join(os.path.join(DATA_PATH, 'input'),
                                                       self.train_img[0])).div(255))
        original_mask = Resize(self.img_shape[1:])(io.read_image(os.path.join(os.path.join(DATA_PATH, 'Output'),
                                                       self.train_img[0].split('.')[0] + '.png')).div(255))
        
        img, mask = dataset[0]
        mean_img, std_img = img.mean(dim = (1, 2)), img.std(dim = (1, 2))
        mean_mask, std_mask = mask.mean(dim = (1, 2)), mask.std(dim = (1, 2))

        or_mean_img, or_std_img = original_img.mean((1,2)), original_img.std((1, 2))
        or_mean_mask, or_std_mask = original_mask.mean((1,2)), original_mask.std((1, 2))

        self.assertTrue(torch.allclose(mean_img, or_mean_img, rtol = 1e-3, atol = 1e-3) and
                        torch.allclose(mean_mask, or_mean_mask, rtol = 1e-3, atol = 1e-3) and
                        torch.allclose(std_img, or_std_img, rtol = 1e-3, atol = 1e-3) and
                        torch.allclose(std_mask, or_std_mask, rtol = 1e-3, atol = 1e-3))


    def test_getitem_transformedTrue(self) -> None:
        transform = RandomHorizontalFlip(1)
        tranformed_dataset = MaskDataset(DATA_PATH, self.train_img, 
                                        size = self.img_shape[1:],
                                        mean = self.mean,
                                        std = self.std,
                                        transform = transform)
        dataset = MaskDataset(DATA_PATH,
                             self.train_img,
                             size = self.img_shape[1:],
                             mean = self.mean,
                             std = self.std)

        img = dataset[0]
        tr_img = tranformed_dataset[0]

        both = torch.cat(img, dim = 0)
        both = transform(both)
        img = torch.tensor_split(both, [3], dim = 0)

        self.assertTrue(torch.allclose(img[0], tr_img[0]))
        self.assertTrue(torch.allclose(img[1], tr_img[1]))
