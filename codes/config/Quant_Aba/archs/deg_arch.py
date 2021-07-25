import torch
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY


class DPRB(nn.Module):
    def __init__(self, nf, ksize1=3, ksize2=3):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf, nf, ksize1, 1, ksize1//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, ksize1, 1, ksize1//2)
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf, nf, ksize2, 1, ksize1//2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, ksize2, 1, ksize1//2)
        )
    
    def forward(self, x):
        x1, x2 = x
        
        f1 = self.body1(x1)
        f2 = self.body2(x2)

        f1 = f1 * f2

        x1 = f1 + x1
        x2 = f2 + x2

        return [x1, x2]


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(self, nf, nb, nc=1, ksize=21, scale=4):
        super().__init__()

        self.ksize = ksize
        self.scale = scale

        self.head1 = nn.Conv2d(nc, nf, scale*2 + 1, scale, scale)
        self.head2 = nn.Conv2d(nc, nf, 3, 1, 1)

        self.body = nn.Sequential(
            *[
                DPRB(nf, ksize1 = 3, ksize2 = 3) for _ in range(nb)
                ],
        )

        self.tail = nn.Conv2d(nf, nc, 3, 1, 1)
        
    def forward(self, x, z):
        B, C, H, W = x.shape
        x = x.view(B*C, 1, *x.shape[2:])
        z = z.view(B*C, 1, *z.shape[2:])

        f1 = self.head1(x)
        f2 = self.head2(z)

        f1, f2 = self.body([f1, f2])

        x = self.tail(f1)

        return x.view(B, C, *x.shape[2:])