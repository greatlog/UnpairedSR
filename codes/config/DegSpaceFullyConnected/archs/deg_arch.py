import torch
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .edsr import ResBlock, default_conv
from kornia.color import yuv


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(
        self, in_nc=64, scale=4,
        kernel=True, nf_kernel=64, nb_kernel=8, ksize=21,
        noise=False, nf_noise=16, nb_noise=4
    ):
        super().__init__()

        self.in_nc = in_nc
        self.scale = scale

        self.kernel = kernel
        self.noise = noise

        if kernel:
            self.ksize = ksize
            deg_kernel = [
                nn.Conv2d(in_nc, nf_kernel, 1, 1, 0),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_kernel, kernel_size=1
                        ) for _ in range(nb_kernel)
                    ],
                nn.Conv2d(nf_kernel, ksize ** 2, 1, 1, 0),
                nn.Softmax(1)
            ]
            self.deg_kernel = nn.Sequential(*deg_kernel)
            # nn.init.constant_(self.deg_kernel[-2].weight, 0)
            # nn.init.constant_(self.deg_kernel[-2].bias, 0)
            # self.deg_kernel[-2].bias.data[ksize**2//2] = 1

            self.pad = nn.ReflectionPad2d(self.ksize//2)

        if self.noise:
            deg_noise = [
                nn.Conv2d(in_nc, nf_noise, 1, 1, 0),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_noise, kernel_size=1
                        ) for _ in range(nb_noise)
                    ],
                nn.Conv2d(nf_noise, 1, 1, 1, 0, bias=False),
                nn.Sigmoid()
            ]
            self.deg_noise = nn.Sequential(*deg_noise)
            nn.init.constant_(self.deg_noise[-2].weight, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        z = torch.randn(B, self.in_nc, H, W).to(x.device)

        # kernel
        if self.kernel:
            kernel = self.deg_kernel(z).view(
                B, 1, self.ksize**2, *z.shape[2:]
            )

            x = x.view(B*C, 1, H, W)
            x = F.unfold(
                self.pad(x), kernel_size=self.ksize, stride=self.scale, padding=0
            ).view(B, C, self.ksize**2, *z.shape[2:])

            x = torch.mul(x, kernel).sum(2).view(B, C, *z.shape[2:])
            kernel = kernel.view(B, self.ksize**2, *z.shape[2:])
        else:
            x = F.interpolate(x, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
            kernel = None

        # noise
        if self.noise:
            noise = self.deg_noise(z) * 2 - 1
            x = x + noise
        else:
            noise = None
        
        return x, kernel, noise

