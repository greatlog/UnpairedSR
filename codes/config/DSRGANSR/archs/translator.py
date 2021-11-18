import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.registry import ARCH_REGISTRY
from .edsr import default_conv, BasicBlock, ResBlock, Upsampler


@ARCH_REGISTRY.register()
class Translator(nn.Module):
    def __init__(self, nb, nf, scale=4, zero_tail=False, conv=default_conv):
        super().__init__()

        self.scale = scale
        # define head module
        if scale >= 1:
            m_head = [conv(3, nf, 3)]
        else:
            s = int(1 / scale)
            m_head = [nn.Conv2d(3, nf, kernel_size=2 * s + 1, stride=s, padding=s)]

        # define body module
        m_body = [
            ResBlock(conv, nf, 3, act=nn.ReLU(True), res_scale=1) for _ in range(nb)
        ]
        m_body.append(conv(nf, nf, 3))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, nf, act=False) if scale > 1 else nn.Identity(),
            conv(nf, 3, 3),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if zero_tail:
            nn.init.constant_(self.tail[-1].weight, 0)
            nn.init.constant_(self.tail[-1].bias, 0)

    def forward(self, x):

        f = self.head(x)
        f = self.body(f)
        f = self.tail(f)

        if self.scale == 1:
            x = f + x
        else:
            x = f + F.interpolate(x, scale_factor=self.scale)
        
        return x
