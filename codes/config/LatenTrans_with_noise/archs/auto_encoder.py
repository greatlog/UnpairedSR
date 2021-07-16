import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.registry import ARCH_REGISTRY

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
    
    def forward(self, x):
        return torch.add(x, self.body(x))

@ARCH_REGISTRY.register()    
class Encoder(nn.Module):
    def __init__(self, nf, nb, in_nc=3, out_nc=3, scale_factor=4):
        super().__init__()

        self.headv2 = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=scale_factor, s2=scale_factor),
            nn.Conv2d(in_nc*scale_factor**2, nf, 3, 1, 1)
        )

        self.body = nn.Sequential(
            *[ResBlock(nf) for _ in range(nb)]
        )

        self.tail = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):

        x = self.headv2(x)
        x = self.body(x)
        x = self.tail(x)

        return x
        
@ARCH_REGISTRY.register() 
class Decoder(nn.Module):
    def __init__(self, nf, nb, in_nc=3, out_nc=3, scale_factor=4):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1)
        )

        self.body = nn.Sequential(
            *[ResBlock(nf) for _ in range(nb)]
        )

        if scale_factor == 4:  # x4
            self.tail = nn.Sequential(
                nn.Conv2d(nf, nf*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(nf, nf*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale_factor == 1:
            self.tail = nn.Sequential(
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

        else:  # x2, x3
            self.tail = nn.Sequential(
                nn.Conv2d(nf, nf*scale_factor**2, 3, 1, 1),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )


    def forward(self, x):

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x
