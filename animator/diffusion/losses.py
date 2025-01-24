from torch import nn
import torch


class CycleLoss(nn.Module):
    def __init__(self, ltype: str = "L1", lambda_A: float = 10, lambda_B: float = 10) -> None:
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        if ltype == "L1":
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError("Cycle loss type {} is not implemented".format(ltype))

    def __call__(
        self, obtained_X: torch.Tensor, obtained_Y: torch.Tensor, real_X: torch.Tensor, real_Y: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(obtained_X, real_X) * self.lambda_A + self.loss(obtained_Y, real_Y) * self.lambda_B


class IdentityLoss(nn.Module):
    def __init__(self, ltype: str = "L1", lambda_idn: float = 0.5) -> None:
        super().__init__()
        self.lambda_idn = lambda_idn
        if ltype == "L1":
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError("Identity loss type {} is not implemented".format(ltype))

    def __call__(
        self,
        obtained_from_X: torch.Tensor,
        obtained_from_Y: torch.Tensor,
        real_X: torch.Tensor,
        real_Y: torch.Tensor,
    ) -> torch.Tensor:
        return (self.loss(obtained_from_X, real_X) + self.loss(obtained_from_Y, real_Y)) * self.lambda_idn
