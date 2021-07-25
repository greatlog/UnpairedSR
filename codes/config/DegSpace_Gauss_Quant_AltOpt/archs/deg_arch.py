import torch
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .edsr import ResBlock, default_conv


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(self, nf, nb, in_nc=3, ksize=21, scale=4):
        super().__init__()

        self.ksize = ksize
        self.scale = scale

        deg_module = [
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            *[
                ResBlock(
                    conv=default_conv, n_feat=nf, kernel_size=3
                    ) for _ in range(nb)
                ],
            nn.Conv2d(nf, ksize**2 + 1, 1, 1, 0),
        ]
        self.deg_module = nn.Sequential(*deg_module)
        self.deg_module[-1].weight.data[-1] = 0

        self.pad = nn.ReflectionPad2d(self.ksize//2)
        
    def forward(self, x, z):
        B, C, H, W = x.shape

        kernel, noise_std = self.deg_module(z).split([self.ksize**2, 1], dim=1)
        kernel = kernel.view(B, 1, self.ksize**2, *z.shape[2:]).softmax(2)

        x = x.view(B*C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=self.ksize, stride=self.scale, padding=0
        ).view(B, C, self.ksize**2, *z.shape[2:])
        x = torch.mul(x, kernel).sum(2).view(B, C, *z.shape[2:])

        noise = noise_std * torch.randn_like(z)
        x = x + noise
        
        return x, kernel.view(B, self.ksize**2, *z.shape[2:])

