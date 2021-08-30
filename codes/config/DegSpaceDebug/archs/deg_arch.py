import torch
from torch import nn
import torch.nn.functional as F

from utils.registry import ARCH_REGISTRY
from .edsr import ResBlock, default_conv
from kornia.color import yuv


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(
        self,  scale=4,
        kernel=True, nc_k=64, nf_k=64, nb_k=8, ksize=21,
        noise=False, nc_n=64, nf_n=16, nb_n=4
    ):
        super().__init__()

        self.scale = scale

        self.kernel = kernel
        self.noise = noise

        if kernel:
            self.nc_k = nc_k

            self.ksize = ksize
            deg_kernel = [
                nn.Conv2d(nc_k, nf_k, 3, 1, 1),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_k, kernel_size=3
                        ) for _ in range(nb_k)
                    ],
                nn.Conv2d(nf_k, ksize ** 2, 1, 1, 0),
                nn.Softmax(1)
            ]
            self.deg_kernel = nn.Sequential(*deg_kernel)
            nn.init.constant_(self.deg_kernel[-2].weight, 0)
            nn.init.constant_(self.deg_kernel[-2].bias, 0)
            self.deg_kernel[-2].bias.data[ksize**2//2] = 1

            self.pad = nn.ReflectionPad2d(self.ksize//2)

        if self.noise:
            self.nc_n = nc_n

            deg_noise = [
                nn.Conv2d(nc_n + 3, nf_n, 3, 1, 1),
                *[
                    ResBlock(
                        conv=default_conv, n_feat=nf_n, kernel_size=3
                        ) for _ in range(nb_n)
                    ],
                nn.Conv2d(nf_n, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ]
            self.deg_noise = nn.Sequential(*deg_noise)
            nn.init.constant_(self.deg_noise[-2].weight, 0)
        
    def forward(self, inp):
        B, C, H, W = inp.shape

        # kernel
        if self.kernel:
            zk = torch.randn(B, self.nc_k, H//self.scale, W//self.scale).to(inp.device)

            kernel = self.deg_kernel(zk).view(
                B, 1, self.ksize**2, *zk.shape[2:]
            )

            x = inp.view(B*C, 1, H, W)
            x = F.unfold(
                self.pad(x), kernel_size=self.ksize, stride=self.scale, padding=0
            ).view(B, C, self.ksize**2, *zk.shape[2:])

            x = torch.mul(x, kernel).sum(2).view(B, C, *zk.shape[2:])
            kernel = kernel.view(B, self.ksize**2, *zk.shape[2:])
        else:
            x = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
            kernel = None

        # noise
        if self.noise:
            zn = torch.randn(B, self.nc_n, H//self.scale, W//self.scale).to(inp.device)
            bic_inp = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
            noise_inp = torch.cat([bic_inp.detach(), zn], 1)
            noise = self.deg_noise(noise_inp) * 2 - 1
            x = x + noise
        else:
            noise = None
        
        return x, kernel, noise

