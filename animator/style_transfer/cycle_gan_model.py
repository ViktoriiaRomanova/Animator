import torch
import torch.nn as nn

__all__ = ['Generator', 'Discriminator']

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = self._make_block(in_ch)

    def _make_block(self, in_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect', bias = True),
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect', bias = True),
            nn.InstanceNorm2d(in_ch))    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, num_res_blocks: int = 9) -> None:
        super(Generator, self).__init__()
        layers: list[nn.Module] = []
        in_ch, out_ch = 3, 64
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect', bias = True),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)]
        
        in_ch, out_ch = out_ch, 2 * out_ch
        # Add downsampling
        for _ in range(2):
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1, bias = True),
                       nn.InstanceNorm2d(out_ch),
                       nn.ReLU(True)]
            in_ch, out_ch = out_ch, 2 * out_ch

        # Add residual blocks
        for _ in range(num_res_blocks):
            layers += [ResidualBlock(in_ch)]

        out_ch = in_ch // 2
        # Add upsampling
        for _ in range(2):
            layers += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                       nn.InstanceNorm2d(out_ch),
                       nn.ReLU(True)]
            in_ch, out_ch = out_ch, out_ch // 2

        out_ch = 3
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect', bias = True),
                   #nn.InstanceNorm2d(out_ch),
                   nn.Tanh()]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch, out_ch = 3, 64
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1, bias = True),
            nn.LeakyReLU(0.2, True)]

        for _ in range(2):
            in_ch, out_ch = out_ch, 2 * out_ch
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1, bias = True),
                       nn.InstanceNorm2d(out_ch),
                       nn.LeakyReLU(0.2, True)]
        
        in_ch, out_ch = out_ch, 2 * out_ch
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 1, padding = 1, bias = True),
                   nn.InstanceNorm2d(out_ch),
                   nn.LeakyReLU(0.2, True)]
        in_ch, out_ch = out_ch, 1
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 1, padding = 1, bias = True)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)        
