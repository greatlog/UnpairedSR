import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.registry import ARCH_REGISTRY
from .edsr import default_conv, BasicBlock, ResBlock, Upsampler


@ARCH_REGISTRY.register()
class DegModel(nn.Module):
    def __init__(self, nb, nf, scale=4, zero_tail=False, conv=default_conv):
        super().__init__()

        self.scale = scale

        # define head module
        m_head = [nn.Conv2d(4, nf, kernel_size=2 * scale + 1, stride=scale, padding=scale)]
        n_head = [nn.Linear(64, 128**2)]

        # define body module
        m_body = [
            ResBlock(conv, nf, 3, act=nn.ReLU(True), res_scale=1) for _ in range(nb)
        ]
        m_body.append(conv(nf, nf, 3))

        # define tail module
        m_tail = [
            conv(nf, 3, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.head_noise = nn.Sequential(*n_head)

        if zero_tail:
            nn.init.constant_(self.tail[-1].weight, 0)
            nn.init.constant_(self.tail[-1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        noise = torch.randn(B, 64).to(x.device)
        noise = self.head_noise(noise).view(B, 1, H, W)

        f = self.head(torch.cat([x, noise], 1))
        f = self.body(f)
        f = self.tail(f)

        if self.scale == 1:
            x = f + x
        else:
            x = f + F.interpolate(x, scale_factor=1 / self.scale)
        return x
