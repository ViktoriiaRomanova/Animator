from typing import cast, Dict, List, Union

import torch
from torch import nn


ARC: Dict[str, List[Union[str, int]]] = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M'],
    }


class UNet(nn.Module):
    """Make SegNet architecture."""

    def __init__(self, architecture: List[Union[str, int]] = ARC['B']) -> None:
        """Create model."""
        super().__init__()
        self._encoder_ = self._make_encoder(architecture)
        self._decoder_ = self._make_decoder(architecture[::-1])
        
        self.enc1 = nn.Sequential(*self._encoder_[0]) # 64
        self.enc2 = nn.Sequential(*self._encoder_[1]) # 128
        self.enc3 = nn.Sequential(*self._encoder_[2]) # 256
        self.enc4 = nn.Sequential(*self._encoder_[3]) # 512
        self.enc5 = nn.Sequential(*self._encoder_[4]) # 512
        
        self.dec5 = nn.Sequential(*self._decoder_[0]) # 512
        self.dec4 = nn.Sequential(*self._decoder_[1]) # 512 + 512
        self.dec3 = nn.Sequential(*self._decoder_[2]) # 512 + 256 
        self.dec2 = nn.Sequential(*self._decoder_[3]) # 256 + 128
        self.dec1 = nn.Sequential(*self._decoder_[4]) # 128 + 64
        
        self.bottleneck = self._make_bottleneck(architecture[-2])
        
    def _make_encoder(self, arc: List[Union[str, int]]) -> List:
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
    
    def _make_decoder(self, arc: List[Union[str, int]]) -> List:
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
        enc1 = self.enc1(x) # 64
        enc2 = self.enc2(enc1) # 128
        enc3 = self.enc3(enc2) # 256
        enc4 = self.enc4(enc3) # 512
        x = self.enc5(enc4)
        
        x = self.bottleneck(x)
        x = self.dec5(x)
        
        x = self.dec4(torch.cat((enc4, x), dim = 1))
        x = self.dec3(torch.cat((enc3, x), dim = 1)) 
        x = self.dec2(torch.cat((enc2, x), dim = 1))
        x = self.dec1(torch.cat((enc1, x), dim = 1))

        return x
