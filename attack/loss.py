from typing import Callable, Dict, Tuple, Union
import torch
import inspect, sys
from torch import Tensor


class Mels():
    name = 'mels'
    def __init__(self,
                 alpha=0.7,
                 sr=16000,
                 n_features=80,
                 winlen=0.025,
                 winstep=0.010,
                 **kwargs):
        import torchaudio
        self.alpha = alpha
        self._n_features = n_features
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sr,
            n_fft=int(winlen * sr),
            hop_length=int(winstep * sr),
            n_mels=n_features)
        super().__init__(**kwargs)
    
    def to(self, device):
        self.melspec.to(device=device)
        return self

    def __call__(self, x):
        mels = self.melspec(x)
        valid_n = int(self.alpha * self._n_features)
        return mels[:, :valid_n, :]


class BackdoorLoss():
    name = 'backdoorloss'
    def __init__(
        self, 
        f: Callable[[Tensor], Tensor], transform: Callable[[Tensor], Tensor], x_tgt: Tensor, beta1: float, 
        phi: Callable[[Tensor], Tensor], x_src: Tensor, beta2: float,
        unit: Tensor) -> None:
        super().__init__()
        self.f_tgt = f(transform(x_tgt)).detach()
        self.phi_src = phi(x_src).detach()
        self.unit = unit.clone()
        self.phi = phi
        self.transform = transform
        self.beta1 = beta1
        self.beta2 = beta2
    
    def __call__(
        self, 
        fq: Callable[[Tensor], Tensor], 
        x: Tensor, 
        Pk: Union[Tensor, Callable[[None], Tensor]]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        l_tgt = torch.norm(fq(self.transform(x)) - self.f_tgt, p=2)
        l_src = torch.norm(self.phi(x) - self.phi_src, p=2)
        l_wgt = torch.norm(Pk if isinstance(Pk, Tensor) else Pk() - self.unit).float()
        l_sum = l_tgt + self.beta1 * l_src + self.beta2 * l_wgt
        return l_sum, dict(srcl=l_src, tgtl=l_tgt, wgtl=l_wgt)


class TriggerLoss():
    name = 'triggerloss'
    def __init__(
        self, 
        fq: Callable[[Tensor], Tensor], transform: Callable[[Tensor], Tensor], x_tgt: Tensor,
        phi: Callable[[Tensor], Tensor], x_src: Tensor, lambda_: float,) -> None:
        super().__init__()
        self.fq_tgt = fq(transform(x_tgt)).detach()
        self.phi_src = phi(x_src).detach()
        self.fq = fq
        self.phi = phi
        self.transform = transform
        self.lambda_ = lambda_
    
    def __call__(self, x: Tensor,) -> Tuple[Tensor, Dict[str, Tensor]]:
        l_tgt = torch.norm(self.fq(self.transform(x)) - self.fq_tgt, p=2)
        l_src = torch.norm(self.phi(x) - self.phi_src, p=2)
        l_sum = l_tgt + self.lambda_ * l_src
        return l_sum, dict(srcl=l_src, tgtl=l_tgt)


def load_loss(name: str):
    for _, c in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if c.name == name:
            return c
    raise ValueError(f'Loss name can not found: {name}')
