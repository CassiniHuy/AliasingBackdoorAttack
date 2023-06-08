from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class FinetuneSetting:
        # * Model
        layer_name: str = None
        pretrained: Union[bool, str] = True
        freeze_norm: bool = True
        input_size: Union[Tuple[int], int] = None
        # * Dataset
        n_class: int = -1
        restore_from: str = None
        # * Train
        epoch: int = 500
        save_best_on_val: bool = True
        early_stop: int = 5 # negative for disable
        lr: float = 5e-3
        weight_decay: float = 1e-4
        momentum: float = 0.9
        gamma: float = 0.9
        save_path: str = 'logs'
        save_every_epoch: int = -1
        batch: int = 256
        sub_batch: int = None
