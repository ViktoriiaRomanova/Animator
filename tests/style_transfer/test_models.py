import unittest

import torch

from animator.style_transfer.cycle_gan_model import *

class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = Generator()
        cls.disc = Discriminator()
        cls.BATCH_SIZE = 5
        cls.INPUT_SIZE = (3, 224, 224)
        cls.input = torch.rand((cls.BATCH_SIZE, *cls.INPUT_SIZE))
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.gen, cls.disc, cls.input

    def test_generator_forward_output_size(self) -> None:
        output_tensor = self.gen(self.input)
        self.assertEqual(output_tensor.shape[1:], self.INPUT_SIZE)

    def test_discriminator_forward_output_size(self) -> None:
        output_tensor = self.disc(self.input)
        self.assertEqual(output_tensor.size(dim = 1), 1)
    

if __name__ == '__main__':
    unittest.main()