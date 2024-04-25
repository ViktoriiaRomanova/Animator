import unittest

import torch

from animator.style_transfer.cycle_gan_model import *

class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = Generator()
        cls.disc = Discriminator()
        batch_size, in_ch, H, W = 5, 3, 224, 224
        cls.input = torch.rand((batch_size, in_ch, H, W))
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.gen, cls.disc, cls.input

    def test_generator_forward_output_size(self) -> None:
        output_tensor = self.gen(self.input)
        self.assertEqual(output_tensor.shape[1:], (3, 224, 224))
    

if __name__ == '__main__':
    unittest.main()