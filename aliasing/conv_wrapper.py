from typing import Union
import torch
from aliasing import load_trans
from torch import nn, Tensor


class ConvWrapper(nn.Module):
    def __init__(self, layer, preprocess: str, **kwargs):
        super().__init__()
        if not (isinstance(layer, nn.Conv2d)
                or isinstance(layer, nn.Conv1d)):
            raise ValueError('Unsupported layer type:', type(layer))
        self.layer = layer
        self.preprocess = load_trans(preprocess)(size=self.layer.kernel_size,
                                              channels=self.layer.out_channels,
                                              **kwargs)
        for k, v in self.preprocess.updatable().items():
            self.register_parameter(k, v)
    
    def new_weight(self) -> Tensor:
        # assume y = WMx
        M = self.preprocess.matrix()
        W = self.layer.weight
        # [C_out, C_in, 1, N] * [C_out, 1, N, N]
        new_weight = torch.matmul(W.reshape([*W.shape[:2], 1, -1]),
                                  M.reshape([-1, 1, *M.shape[-2:]]))
        return new_weight.reshape(W.shape)

    def forward(self, x):
        W = self.new_weight()
        if W.dim() == 3:
            f = nn.functional.conv1d
        else:
            f = nn.functional.conv2d
        y = f(x,
              weight=W,
              bias=self.layer.bias,
              stride=self.layer.stride,
              padding=self.layer.padding)
        return y
    
    def unwrap(self) -> Union[nn.Conv1d, nn.Conv2d]:
        self.layer.weight.data = self.new_weight()
        return self.layer

    def __str__(self):
        return f'ConvWrapper(layer={str(self.layer)}, preprocess={str(self.preprocess.name)})'
