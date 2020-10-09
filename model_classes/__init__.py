import torch
from torch import nn as nn
from utils_dct import dct_flip


class TransformLayer(torch.nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return x.sub(self.mean).div(self.std)


class TransformFlippedLayer(nn.Module):

    def __init__(self, mean, std, shape, device):
        super().__init__()
        self.mean = nn.Parameter(dct_flip(torch.ones(shape).to(device))[None, :, :, :] * mean,
                                 requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return x.sub(self.mean).div(self.std)
