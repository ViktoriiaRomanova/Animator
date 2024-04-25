import unittest

import torch

from animator.style_transfer.cycle_gan_model import *
from animator.style_transfer.losses import *

class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.genA = Generator()
        cls.discA = Discriminator()
        cls.genB = Generator()
        cls.BATCH_SIZE = 1
        cls.INPUT_SIZE = (3, 224, 224)
        cls.input = torch.rand((cls.BATCH_SIZE, *cls.INPUT_SIZE))
        cls.cycle_loss = CycleLoss()
        cls.identity_loss = IdentityLoss()
    
    @classmethod
    def tearDownClass(cls) -> None:
        del cls.genA, cls.genB, cls.discA, cls.input

    def test_AdversarialLossMSE_call(self) -> None:
        output_tensor = self.discA(self.genA(self.input))
        adv_loss = AdversarialLoss('MSE')

        loss_true = float(adv_loss(output_tensor, True).detach())
        loss_false = float(adv_loss(output_tensor, False).detach())

        self.assertNotAlmostEqual(loss_true, loss_false)

    def test_AdversarialLossBCE_call(self) -> None:
        output_tensor = self.discA(self.genA(self.input))
        adv_loss = AdversarialLoss('BCE')

        loss_true = float(adv_loss(output_tensor, True).detach())
        loss_false = float(adv_loss(output_tensor, False).detach())

        self.assertNotAlmostEqual(loss_true, loss_false)

    def test_CycleLoss_call(self) -> None:
        obtained_X = self.genB(self.genA(self.input))
        obtained_Y = self.genA(self.genB(self.input))

        loss = self.cycle_loss(obtained_X, obtained_Y, self.input, self.input)

        self.assertGreater(loss, 0)
    
    def test_IdentityLoss_call(self) -> None:
        obtained_X = self.genB(self.genA(self.input))
        obtained_Y = self.genA(self.genB(self.input))

        loss = self.identity_loss(obtained_X, obtained_Y, self.input, self.input)

        self.assertGreater(loss, 0)

    

if __name__ == '__main__':
    unittest.main()