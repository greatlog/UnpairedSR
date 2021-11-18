import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.registry import ARCH_REGISTRY
from utils.resize_utils import imresize


@ARCH_REGISTRY.register()
class BicuBic(nn.Module):
    def __init__(self, upscale=4):
        super().__init__()

        self.empty = nn.Parameter(torch.FlaotTensor([0.0]))
        self.upscale = upscale

    def forward(self, x):
        y  = imresize(x, self.upscale)
        return y
