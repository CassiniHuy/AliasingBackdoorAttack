from typing import Dict
import torch
from aliasing import PreProcess
from torch import Tensor, nn
from numpy import prod


def get_dist(size: tuple):
    if len(size) == 2:
        m, n = size
        i_ind = torch.tile(torch.tensor([[[i]] for i in range(m)]),
                           dims=[1, n, 1]).float()  # [m, n, 1]
        j_ind = torch.tile(torch.tensor([[[i] for i in range(n)]]),
                           dims=[m, 1, 1]).float()  # [m, n, 1]
        ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
        x = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
        y = torch.tile(ij_ind.reshape([1, -1, 2]),
                       dims=[m * n, 1, 1])  # [m * n, m * n, 2]
        dist_xy = torch.cdist(y, x, p=2).squeeze(dim=-1)  # [m * n, m * n]
    elif len(size) == 1:
        n = size[0]
        i_ind = torch.tensor([[i] for i in range(n)]).float()  # [n, 1]
        x = i_ind.reshape([n, 1, 1])  # [n, 1, 1]
        y = torch.tile(i_ind.unsqueeze(0), dims=[n, 1, 1])  # [n, n, 1]
        dist_xy = torch.cdist(y, x, p=2).squeeze(dim=-1)  # [n, n]
    else:
        raise ValueError(f'Unsupported size={str(size)}')
    return dist_xy


def get_mask(size, radius):
    mask = get_dist(size).clone()
    mask[mask > radius] = -1
    mask[mask != -1] = 1
    mask[mask == -1] = 0
    return mask


class Adaptive(PreProcess):
    name = 'Adaptive'

    def __init__(self, size: tuple, channels: int, radius: float = 1):
        super().__init__(size=size, channels=channels)
        self.interp_w = torch.tile(torch.zeros(
            (prod(size), ) * 2).float().unsqueeze(0),
                                   dims=[channels, 1, 1])
        self.interp_w = nn.Parameter(self.interp_w)
        self.radius = radius
        self.mask = get_mask(size, radius)

    def matrix(self) -> Tensor:
        P_exp = torch.exp(self.interp_w)
        P_msk = torch.mul(P_exp, self.mask.to(P_exp.device))
        matrix = P_msk / torch.sum(P_msk, dim=-1, keepdim=True)
        return matrix

    def updatable(self) -> Dict[str, torch.Tensor]:
        return dict(interp_w=self.interp_w)

    def __str__(self) -> str:
        return f'Adaptive(size={str(self.size)}, channels={self.channels}, radius={self.radius})'
