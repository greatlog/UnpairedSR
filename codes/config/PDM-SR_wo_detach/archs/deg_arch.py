import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils.registry import ARCH_REGISTRY


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

class KernelModel(nn.Module):
    def __init__(self, opt, scale):
        super().__init__()

        self.opt = opt
        self.scale = scale

        nc, nf, nb = opt["nc"], opt["nf"], opt["nb"]
        ksize = opt["ksize"]

        head_k = opt["head_k"]
        body_k = opt["body_k"]
        
        if opt["mix"]:
            in_nc = 3 + nc
        else:
            in_nc = nc

        deg_kernel = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[
                ResBlock(nf=nf, ksize=body_k)
                for _ in range(nb)
                ],
            nn.Conv2d(nf, ksize ** 2, 1, 1, 0),
            nn.Softmax(1)
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)

        if opt["zero_init"]:
            nn.init.constant_(self.deg_kernel[-2].weight, 0)
            nn.init.constant_(self.deg_kernel[-2].bias, 0)
            self.deg_kernel[-2].bias.data[ksize**2//2] = 1

        self.pad = nn.ReflectionPad2d(ksize//2)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = H // self.scale
        w = W // self.scale

        if self.opt["nc"] > 0:
            if self.opt["spatial"]:
                zk = torch.randn(B, self.opt["nc"], H, W).to(x.device)
            else:
                zk = torch.randn(B, self.opt["nc"], 1, 1).to(x.device)
                if self.opt["mix"]:
                    zk = zk.repeat(1, 1, H, W)
        
        if self.opt["mix"]:
            if self.opt["nc"] > 0:
                inp = torch.cat([x, zk], 1)
            else:
                inp = x
        else:
            inp = zk          
        
        ksize = self.opt["ksize"]
        kernel = self.deg_kernel(inp).view(B, 1, ksize**2, *inp.shape[2:])

        x = x.view(B*C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=ksize, stride=self.scale, padding=0
        ).view(B, C, ksize**2, h, w)

        x = torch.mul(x, kernel).sum(2).view(B, C, h, w)
        kernel = kernel.view(B, ksize, ksize, *inp.shape[2:]).squeeze()

        return x, kernel

class NoiseModel(nn.Module):
    def __init__(self, opt, scale):
        super().__init__()

        self.scale = scale
        self.opt = opt

        nc, nf, nb = opt["nc"], opt["nf"], opt["nb"]

        head_k = opt["head_k"]
        body_k = opt["body_k"]
        
        if opt["mix"]:
            in_nc = 3 + nc
        else:
            in_nc = nc

        deg_noise = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k//2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[
                ResBlock(nf=nf, ksize=body_k)
                for _ in range(nb)
                ],
            nn.Conv2d(nf, opt["dim"], 1, 1, 0),
        ]
        self.deg_noise = nn.Sequential(*deg_noise)

        if opt["zero_init"]:
            nn.init.constant_(self.deg_noise[-1].weight, 0)
            nn.init.constant_(self.deg_noise[-1].bias, 0)
        else:
            nn.init.normal_(self.deg_noise[-1].weight, 0.001)
            nn.init.constant_(self.deg_noise[-1].bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape

        if self.opt["nc"] > 0:     
            if self.opt["spatial"]:
                zn = torch.randn(x.shape[0], self.opt["nc"], H, W).to(x.device)
            else:
                zn = torch.randn(x.shape[0], self.opt["nc"], 1, 1).to(x.device)
                if self.opt["mix"]:
                    zn = zn.repeat(1, 1, H, W)
        
        if self.opt["mix"]:
            if self.opt["nc"] > 0:
                inp = torch.cat([x, zn], 1)
            else:
                inp = x
        else:
            inp = zn
            
        noise = self.deg_noise(inp)

        return noise

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
            self.deg_kernel = KernelModel(kernel_opt, scale)
        
        if noise_opt is not None:
           self.deg_noise = NoiseModel(noise_opt, scale)

        else:
            self.quant = Quantization()
        
    def forward(self, inp):
        B, C, H, W = inp.shape
        h = H // self.scale
        w = W // self.scale

        # kernel
        if self.kernel_opt is not None:
            x, kernel = self.deg_kernel(inp)
        else:
            x = F.interpolate(inp, scale_factor=1/self.scale, mode="bicubic", align_corners=False)
            kernel = None

        # noise
        if self.noise_opt is not None:
            noise = self.deg_noise(x)
            x = x + noise
        else:
            noise = None
            x = self.quant(x)
        return x, kernel, noise

