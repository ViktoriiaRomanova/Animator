from typing import cast, Dict, List, Union

import torch
from torch import nn

__all__ = [
            'vgg16',
]


ARC: Dict[str, List[Union[str, int]]] = {
    'V16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M']  # VGG16
    }


class SegNet(nn.Module):
    """Make SegNet architecture."""

    def __init__(self, architecture: List[Union[str, int]] = ARC['V16']) -> None:
        """Create model."""
        super().__init__()
        self.encoder = self._make_encoder(architecture)
        self.decoder = self._make_decoder(architecture[::-1])
        
    def _make_encoder(self, arc: List[Union[str, int]]) -> nn.Sequential:
        """Construct encoder layers."""
        layers: List[nn.Module] = []
        in_ch = 3
        for l in arc:
            if l == 'M':
                layers += [nn.MaxPool2d(2, stride = 2)]
            else:
                l = cast(int, l)
                layers += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1),
                           nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
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
                layers += [nn.Conv2d(in_ch, l, 3, stride = 1, padding = 1),
                           nn.BatchNorm2d(l), nn.ReLU(inplace = True)]
                in_ch = l
        layers.append(nn.Conv2d(in_ch, 1, 3, stride = 1, padding = 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make forward path."""
        x = self.encoder(x)
        x = self.decoder(x)

        return x
