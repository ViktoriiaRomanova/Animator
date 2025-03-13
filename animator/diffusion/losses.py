from torch import nn
import torch
from torchmetrics.metric import Metric


class CycleLoss(nn.Module):
    def __init__(
        self,
        lpips: Metric,
        ltype: str = "L1",
        lambda_A: float = 1,
        lambda_B: float = 1,
        lambda_lpips: float = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lpips = lpips
        self.lambda_lpips = lambda_lpips
        self.device = device
        if ltype == "L1":
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError("Cycle loss type {} is not implemented".format(ltype))

    def __call__(
        self, obtained_X: torch.Tensor, obtained_Y: torch.Tensor, real_X: torch.Tensor, real_Y: torch.Tensor
    ) -> torch.Tensor:
        obtained_X = obtained_X.to(self.device)
        obtained_Y = obtained_Y.to(self.device)
        real_X = real_X.to(self.device)
        real_Y = real_Y.to(self.device)
        return (
            self.loss(obtained_X, real_X) * self.lambda_A
            + self.lpips(obtained_X, real_X) * self.lambda_lpips
            + self.loss(obtained_Y, real_Y) * self.lambda_B
            + self.lpips(obtained_Y, real_Y) * self.lambda_lpips
        )


class IdentityLoss(nn.Module):
    def __init__(
        self,
        lpips: Metric,
        ltype: str = "L1",
        lambda_idn: float = 1,
        lambda_lpips: float = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.lambda_idn = lambda_idn
        self.lpips = lpips
        self.lambda_lpips = lambda_lpips
        self.device = device
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
        obtained_from_X = obtained_from_X.to(self.device)
        obtained_from_Y = obtained_from_Y.to(self.device)
        real_X = real_X.to(self.device)
        real_Y = real_Y.to(self.device)

        return (
            self.loss(obtained_from_X, real_X) * self.lambda_idn
            + self.lpips(obtained_from_X, real_X) * self.lambda_lpips
            + self.loss(obtained_from_Y, real_Y) * self.lambda_idn
            + self.lpips(obtained_from_Y, real_Y) * self.lambda_lpips
        )
