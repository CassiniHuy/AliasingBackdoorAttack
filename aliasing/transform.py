from abc import ABC
from abc import abstractmethod
from typing import Dict
from torch import Tensor


class PreProcess(ABC):
    name = 'preprocess'

    def __init__(self, size: tuple, channels: int):
        self.size = size
        self.channels = channels

    @abstractmethod
    def matrix(self) -> Tensor:
        pass  # y=Mx

    def __str__(self) -> str:
        return f"PreProcess(size={self.size}, channels={self.channels}))"

    def updatable(self) -> Dict[str, Tensor]:
        return dict()
