from torch import nn
import torch

__all__ = ['AdversarialLoss', 'CycleLoss', 'IdentityLoss']
# TO DO: add description everywhere 

class AdversarialLoss(nn.Module):
    def __init__(self, ltype: str = 'MSE',
                 real_val: float = 1.0, fake_val: float = 0.0,
                 device: torch.device | str = 'cpu') -> None:
        super().__init__()

        self.register_buffer('real', torch.tensor(real_val).to(device), persistent = False)
        self.register_buffer('fake', torch.tensor(fake_val).to(device), persistent = False)

        if ltype == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
        elif ltype == 'MSE':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('Adversarial loss type {} is not implemented'.format(ltype))

    def __make_target_tensor(self, ttype: bool, input: torch.Tensor) -> torch.Tensor:
        if ttype:
            return self.real.expand_as(input)
        else:
            return self.fake.expand_as(input)

    def __call__(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return self.loss(input, self.__make_target_tensor(target_is_real, input))
    

class CycleLoss(nn.Module):
    def __init__(self, ltype: str = 'L1', lambda_A: float = 10, lambda_B: float = 10) -> None:
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        if ltype == 'L1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError('Cycle loss type {} is not implemented'.format(ltype))
    
    def __call__(self, obtained_X: torch.Tensor, obtained_Y: torch.Tensor,
                 real_X: torch.Tensor, real_Y: torch.Tensor) -> torch.Tensor:
        return self.loss(obtained_X, real_X) * self.lambda_A + \
             + self.loss(obtained_Y, real_Y) * self.lambda_B

class IdentityLoss(nn.Module):
    def __init__(self, ltype: str = 'L1', lambda_idn: float = 0.5) -> None:
        super().__init__()
        self.lambda_idn = lambda_idn
        if ltype == 'L1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError('Identity loss type {} is not implemented'.format(ltype))

    def __call__(self, obtained_from_X: torch.Tensor, obtained_from_Y: torch.Tensor,
                 real_X: torch.Tensor, real_Y: torch.Tensor) -> torch.Tensor:
        return (self.loss(obtained_from_X, real_X) + self.loss(obtained_from_Y, real_Y)) * self.lambda_idn * 10
