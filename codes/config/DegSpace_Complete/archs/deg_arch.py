import torch
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .edsr import ResBlock, default_conv
from kornia.color import yuv


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(
        self, in_nc=3, scale=4,
        nf_kernel=64, nb_kernel=8, ksize=21,
        jpeg=False, nf_jpeg=16, nb_jpeg=4,
        noise=False, nf_noise=16, nb_noise=4
    ):
        super().__init__()

        self.ksize = ksize
        self.scale = scale
        self.jpeg = jpeg
        self.noise = noise

        deg_kernel = [
            nn.Conv2d(in_nc, nf_kernel, 3, 1, 1),
            *[
                ResBlock(
                    conv=default_conv, n_feat=nf_kernel, kernel_size=3
                    ) for _ in range(nb_kernel)
                ],
            nn.Conv2d(nf_kernel, ksize ** 2, 1, 1, 0)
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)
        nn.init.constant_(self.deg_kernel[-1].weight, 0)
        nn.init.constant_(self.deg_kernel[-1].bias, 0)
        self.deg_kernel[-1].bias.data[ksize**2//2] = 1

        self.pad = nn.ReflectionPad2d(self.ksize//2)

        if self.noise:
            deg_noise = [
                nn.Conv2d(in_nc, nf_noise, 3, 1, 1),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_noise, kernel_size=3
                        ) for _ in range(nb_noise)
                    ],
                nn.Conv2d(nf_noise, 1, 1, 1, 0, bias=False),
            ]
            self.deg_noise = nn.Sequential(*deg_noise)
            nn.init.constant_(self.deg_noise[-1].weight, 0)
        
        if self.jpeg:
            deg_jpeg = [
                nn.Conv2d(in_nc, nf_jpeg, 3, 1, 1),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_jpeg, kernel_size=3
                        ) for _ in range(nb_jpeg)
                    ],
                nn.Conv2d(nf_jpeg, 1, 1, 1, 0, bias=False),
            ]
            self.deg_jpeg = nn.Sequential(*deg_jpeg)
            nn.init.constant_(self.deg_jpeg[-1].weight, 0)
        
    def forward(self, x, z):
        B, C, H, W = x.shape

        # kernel
        kernel = self.deg_kernel(z).view(
            B, 1, self.ksize**2, *z.shape[2:]
        )
        kernel = kernel / (kernel.sum(dim=2, keepdim=True) + 1e-8)

        x = x.view(B*C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=self.ksize, stride=self.scale, padding=0
        ).view(B, C, self.ksize**2, *z.shape[2:])
        x = torch.mul(x, kernel).sum(2).view(B, C, *z.shape[2:])
        kernel = kernel.view(B, self.ksize**2, *z.shape[2:])

        # noise
        if self.noise:
            noise = self.deg_noise(z)
            x = x + noise
        else:
            noise = None
        
        # jpeg
        if self.jpeg:
            jpeg = self.deg_jpeg(z)
            y, u, v = yuv.rgb_to_yuv(x).chunk(3, dim=1)
            y = torch.fft.fft2(y)
            y = y + jpeg
            y = torch.fft.ifft2(y).real
            x = yuv.yuv_to_rgb(torch.cat([y, u, v], dim=1))
        else:
            jpeg = None
        
        return x, kernel, noise, jpeg

