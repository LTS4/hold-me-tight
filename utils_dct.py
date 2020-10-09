import torch
import torch_dct


def dct_flip(x):
    return torch_dct.idct_2d(torch.flip(torch_dct.dct_2d(x, norm='ortho'), [-2, -1]), norm='ortho')


def dct_low_pass(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.zeros_like(x)
    mask[:, :bandwidth, :bandwidth] = 1
    return torch_dct.idct_2d(torch_dct.dct_2d(x, norm='ortho') * mask, norm='ortho').squeeze_()


def dct_high_pass(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.zeros_like(x)
    mask[:, -bandwidth:, -bandwidth:] = 1
    return torch_dct.idct_2d(torch_dct.dct_2d(x, norm='ortho') * mask, norm='ortho').squeeze_()


def dct_cutoff_low(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.ones_like(x)
    mask[:, :bandwidth, :bandwidth] = 0
    return torch_dct.idct_2d(torch_dct.dct_2d(x, norm='ortho') * mask, norm='ortho').squeeze_()
