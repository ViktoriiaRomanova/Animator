from torch import nn, Tensor, tensor


class DiffRenorm(nn.Module):
    def __init__(self, mean: Tensor | list[float], std: Tensor | list[float]) -> None:
        super().__init__()
        if not isinstance(mean, Tensor):
            mean = tensor(mean)
        if not isinstance(std, Tensor):
            std = tensor(std)
        inv_std = 1 / std
        inv_mean = -mean * inv_std

        self.inv_std = inv_std.reshape(1, 3, 1, 1)
        self.inv_mean = inv_mean.reshape(1, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.inv_mean.to(device=x.device, dtype=x.dtype)) / self.inv_std.to(
            device=x.device, dtype=x.dtype
        )
        return x
