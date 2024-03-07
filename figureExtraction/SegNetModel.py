from typing import cast, Dict, List, Union

import torch
from torch import nn


ARC: Dict[str, List[Union[str, int]]] = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M'],
    'C': [128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M'] 
    }


class SegNet(nn.Module):
    """Make SegNet architecture."""

    def __init__(self, arc: str = 'C') -> None:
        """Create model."""
        super().__init__()
        self.architecture = ARC[arc]
        self.encoder = self._make_encoder(self.architecture)
        self.decoder = self._make_decoder(self.architecture[::-1])
        self.bottleneck = self._make_bottleneck(self.architecture[-2])
        
    def _make_encoder(self, arc: List[Union[str, int]]) -> nn.Sequential:
        """Construct encoder layers."""
        layers: List[nn.Module] = []
        in_ch = 3
        for l in arc:
            if l == 'M':
                layers += [nn.MaxPool2d(2, stride = 2)]
            else:
                l = cast(int, l)
                layers += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1, bias = False),
                           nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
        return nn.Sequential(*layers)

    def _make_bottleneck(self, in_ch: int) -> nn.Sequential:
        """Construct bottleneck layers."""
        layers: List[nn.Module] = []
        for scale_factor in [2, 1, 0.5]:
            out_ch = int(in_ch * scale_factor)
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride = 1, padding = 1, bias = False),
                           nn.BatchNorm2d(out_ch), nn.ReLU(inplace = True)]
            in_ch = out_ch
        return nn.Sequential(*layers)
    
    def _make_decoder(self, arc: List[Union[str, int]]) -> nn.Sequential:
        """Construct decoder layers."""
        layers: List[nn.Module] = []
        in_ch = arc[1]
        for l in arc:
            if l == 'M':
                layers += [nn.Upsample(scale_factor = 2, mode = 'bilinear')]
            else:
                l = cast(int, l)
                layers += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1, bias = False),
                           nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
        layers.append(nn.Conv2d(in_ch, 1, 3, stride = 1, padding = 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make forward path."""
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x
