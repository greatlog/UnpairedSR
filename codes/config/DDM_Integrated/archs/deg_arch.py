import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils.registry import ARCH_REGISTRY
from kornia.color import yuv


class ResBlock(nn.Module):
    def __init__(self, nf, ksize, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        
        self.nf = nf
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            norm(nf), act(),
            nn.Conv2d(nf, nf, ksize, 1, ksize//2)
        )
    
    def forward(self, x):
        return torch.add(x, self.body(x))

class Quantization(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.n = n

    def forward(self, inp):
        out = inp * 255.0
        flag = -1
        for i in range(1, self.n + 1):
            out = out + flag / np.pi / i * torch.sin(2 * i * np.pi * inp * 255.0)
            flag = flag * (-1)
        return out / 255.0

@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(
        self,  scale=4, nc_img=3, kernel_opt=None, noise_opt=None
    ):
        super().__init__()

        self.scale = scale

        self.kernel_opt = kernel_opt
        self.noise_opt = noise_opt

        if kernel_opt is not None:
            nc, nf, nb = kernel_opt["nc"], kernel_opt["nf"], kernel_opt["nb"]
            ksize = kernel_opt["ksize"]
            mix = kernel_opt["mix"]
            in_nc = nc + nc_img if mix else nc

            spatial = kernel_opt["spatial"]
            if spatial:
                head_k = kernel_opt["head_k"]
                body_k = kernel_opt["body_k"]
            else:
                head_k = body_k = 1

            deg_kernel = [
                nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
                nn.BatchNorm2d(nf), nn.ReLU(),
                *[
                    ResBlock(nf=nf, ksize=body_k)
                    for _ in range(nb)
                    ],
                nn.Conv2d(nf, ksize ** 2, 1, 1, 0),
                nn.Softmax(1)
            ]
            self.deg_kernel = nn.Sequential(*deg_kernel)

            if kernel_opt["zero_init"]:
                nn.init.constant_(self.deg_kernel[-2].weight, 0)
                nn.init.constant_(self.deg_kernel[-2].bias, 0)
                self.deg_kernel[-2].bias.data[ksize**2//2] = 1

            self.pad = nn.ReflectionPad2d(ksize//2)

        if noise_opt is not None:
            nc, nf, nb = noise_opt["nc"], noise_opt["nf"], noise_opt["nb"]
            mix = noise_opt["mix"]
            in_nc = nc + nc_img if mix else nc

            head_k = noise_opt["head_k"]
            body_k = noise_opt["body_k"]

            deg_noise = [
                nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
                nn.BatchNorm2d(nf), nn.ReLU(),
                *[
                    ResBlock(nf=nf, ksize=body_k)
                    for _ in range(nb)
                    ],
                nn.Conv2d(nf, noise_opt["dim"], head_k, 1, head_k//2),
                # nn.Sigmoid()
            ]
            self.deg_noise = nn.Sequential(*deg_noise)
            if noise_opt["zero_init"]:
                nn.init.constant_(self.deg_noise[-1].weight, 0)
                nn.init.constant_(self.deg_noise[-1].bias, 0)
            else:
                nn.init.normal_(self.deg_noise[-1].weight, 0.001)
                nn.init.constant_(self.deg_noise[-1].bias, 0)
        else:
            self.quant = Quantization()
        
    def forward(self, inp):
        B, C, H, W = inp.shape
        h = H // self.scale
        w = W // self.scale

        # kernel
        if self.kernel_opt is not None:
            if self.kernel_opt["mix"]:
                inp_k = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
                if self.kernel_opt["nc"] > 0:
                    nc = self.kernel_opt["nc"]
                    if self.kernel_opt["spatial"]:
                        zk = torch.randn(B, nc, h, w).to(inp.device)
                    else:
                        zk = torch.randn(B, nc, 1, 1).to(inp.device)
                    inp_k = torch.cat([inp_k, zk], 1)
            else:
                nc = self.kernel_opt["nc"]
                if self.kernel_opt["spatial"]:
                    inp_k = torch.randn(B, nc, h, w).to(inp.device)
                else:
                    inp_k = torch.randn(B, nc, 1, 1).to(inp.device)
            
            ksize = self.kernel_opt["ksize"]
            kernel = self.deg_kernel(inp_k).view(B, 1, ksize**2, *inp_k.shape[2:])
            # kernel = kernel / (kernel.sum(2, keepdims=True) + 1e-6)

            x = inp.view(B*C, 1, H, W)
            x = F.unfold(
                self.pad(x), kernel_size=ksize, stride=self.scale, padding=0
            ).view(B, C, ksize**2, h, w)

            x = torch.mul(x, kernel).sum(2).view(B, C, h, w)
            kernel = kernel.view(B, ksize, ksize, *inp_k.shape[2:])
        else:
            x = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
            kernel = None

        # noise
        if self.noise_opt is not None:
            if self.noise_opt["mix"]:
                # inp_n = x.detach()
                inp_n = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
                if self.noise_opt["nc"] > 0:
                    nc = self.noise_opt["nc"]
                    zn = torch.randn(B, nc, h, w).to(inp.device)
                    inp_n = torch.cat([inp_n, zn], 1)
            else:
                nc = self.noise_opt["nc"]
                inp_n = torch.randn(B, nc, h, w).to(inp.device)

            noise = self.deg_noise(inp_n)

            x = x + noise

        else:
            noise = None
            x = self.quant(x)
        return x, kernel, noise

