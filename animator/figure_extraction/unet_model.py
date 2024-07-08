"""Create a UNet model for several architectures."""
from typing import cast, Dict, List, Union

import torch
from torch import nn


ARC: Dict[str, List[Union[str, int]]] = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
          512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
          512, 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
    }


class UNet(nn.Module):
    """Make UNet architecture."""

    def __init__(self, arc: str = 'B') -> None:
        """Create model."""
        super().__init__()
        self.architecture = ARC[arc]
        self.encoder = nn.ModuleList(nn.Sequential(*block)
                                     for block in self._make_encoder(self.architecture))
        self.decoder = nn.ModuleList(nn.Sequential(*block)
                                     for block in self._make_decoder(self.architecture[::-1]))

        self.bottleneck = self._make_bottleneck(self.architecture[-2])

    def _make_encoder(self, arc: List[Union[str, int]]) -> List[nn.Module]:
        """Construct encoder layers."""
        layers: List[nn.Module] = [[]]
        in_ch = 3
        for l in arc:
            if l == 'M':
                layers.append([])
                layers[-1] += [nn.MaxPool2d(2, stride = 2)]
            else:
                l = cast(int, l)
                layers[-1] += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
        return layers

    def _make_bottleneck(self, in_ch: int) -> nn.Sequential:
        """Construct bottleneck layers."""
        layers: List[nn.Module] = []
        for scale_factor in [2, 1, 0.5]:
            out_ch = int(in_ch * scale_factor)
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride = 1, padding = 1, bias = False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace = True)]
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _make_decoder(self, arc: List[Union[str, int]]) -> List[nn.Module]:
        """Construct decoder layers."""
        layers: List[nn.Module] = [[]]
        in_ch = arc[1]
        for l in arc:
            if l == 'M':
                layers[-1] += [nn.Upsample(scale_factor = 2, mode = 'bilinear')]
                layers.append([])
            else:
                l = cast(int, l)
                if len(layers[-1]) == 0:
                    in_ch += l
                layers[-1] += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l

        layers[-1] += [nn.Conv2d(in_ch, 1, 3, stride = 1, padding = 1)]
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make forward path."""
        enc = []  # to store encoder outputs for the decoder
        for block in self.encoder:
            x = block(x)
            enc.append(x)

        x = self.bottleneck(x)
        x = self.decoder[0](x)

        for block, part_x in zip(self.decoder[1:], enc[-2::-1]):
            x = block(torch.cat((part_x, x), dim = 1))
        del enc
        return x
