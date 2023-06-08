import torch
from aliasing import PreProcess
from torch import Tensor, nn


class Downsample(PreProcess):
    name = 'downsample'

    def __init__(self,
                 size: tuple,
                 channels: int,
                 ratio: float = 2,
                 mode: str = 'nearest'):
        self.ratio = ratio
        self.mode = mode
        super().__init__(size=size, channels=channels)

    def _upscale(self, x: Tensor, size: tuple) -> Tensor:
        x_up = nn.functional.interpolate(x,
                                         size=size,
                                         mode='bicubic',
                                         align_corners=False)
        return x_up

    def _downscale(self, x: Tensor) -> Tensor:
        size = [int(l * (1 / self.ratio)) for l in x.shape[-2:]]
        x_down = nn.functional.interpolate(x, size=size, mode=self.mode)
        return x_down

    def downsample(self, x: Tensor) -> Tensor:
        hw = x.shape[-2:]
        x_padded = nn.functional.pad(x, pad=(0, hw[1] % 2, 0, hw[0] %
                                             2))  # to force a uniform sampling
        x_down = self._downscale(x_padded)
        x_up = self._upscale(x_down, size=hw)
        return x_up

    def matrix(self) -> Tensor:
        n = self.size[0] * self.size[1]
        e = torch.eye(n)
        y = self.downsample(e.reshape((n, 1, *self.size)))
        # return y.reshape((n, n))  # when y=xA
        return y.reshape((n, n)).permute(1, 0)  # when y=Ax

    def __str__(self) -> str:
        return 'Downsample(size={2}, downscale={0}, mode={1})'.format(
            str(self.downscale), self.mode, self.size)
