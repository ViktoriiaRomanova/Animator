import unittest

import torch

from animator.figure_extraction.segnet_model import SegNet
from animator.figure_extraction.unet_model import UNet

class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.unet = {'A': UNet('A'), 'B': UNet('B'), 'C': UNet('C')}
        cls.segnet = {'A': SegNet('A'), 'B': SegNet('B'), 'C': SegNet('C')}
        cls.BATCH_SIZE = 5
        cls.INPUT_SIZE = (3, 224, 224)
        cls.OUTPUT_SIZE = (1, 224, 224)
        cls.input = torch.rand((cls.BATCH_SIZE, *cls.INPUT_SIZE))
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.unet, cls.segnet, cls.input

    def test_UNetA_forward_output_size(self) -> None:
        output_tensor = self.unet['A'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)
    
    def test_UNetB_forward_output_size(self) -> None:
        output_tensor = self.unet['B'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)
    
    def test_UNetC_forward_output_size(self) -> None:
        output_tensor = self.unet['C'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)

    def test_SegNetA_forward_output_size(self) -> None:
        output_tensor = self.segnet['A'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)
    
    def test_SegNetB_forward_output_size(self) -> None:
        output_tensor = self.segnet['B'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)
    
    def test_SegNetC_forward_output_size(self) -> None:
        output_tensor = self.segnet['C'](self.input)
        self.assertEqual(output_tensor.shape[1:], self.OUTPUT_SIZE)
    

if __name__ == '__main__':
    unittest.main()