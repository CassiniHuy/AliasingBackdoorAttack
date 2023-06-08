from typing import Dict, List, Tuple
import torch
from attack.loss import load_loss
from tqdm import tqdm
from torch import Tensor


class Optimizer():
    def __init__(self, loss: str, **kwargs) -> None:
        self.loss = load_loss(loss)(**kwargs)

    def set_params(self, params: List[Tensor]) -> List[Tensor]:
        """set optimizable parameters

        Args:
            params (List[Tensor]): parameters to be optimized

        Returns:
            List[Tensor]: optimizable parameters
        """        
        for i in range(len(params)):
            params[i].requires_grad_(True)
        self.params = params
        return self.params
        
    def optimize(self,
                 loss_params: List[Tensor],
                 lr: float = 1e-3,
                 x_adv: torch.Tensor = None,
                 clamp_min: float = None,
                 clamp_max: float = None,
                 iterations: int = 5000,
                 optimizer: str ='Adam',
                 quiet: bool = False,
                 ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        updater = getattr(torch.optim, optimizer)(self.params, lr=lr)
        progress = range(iterations) if quiet else tqdm(range(iterations))
        for _ in progress:
            updater.zero_grad()
            loss, state = self.loss(*loss_params)
            loss.backward()
            if quiet is False:
                progress.set_description(
                    f"loss: {loss.item():.4f} " +
                    ' '.join([f'{k}: {v.item():.4f}' for k, v in state.items()]))
            updater.step()
            if (clamp_max is not None) and (clamp_min is not None):
                # x_adv.data.clamp_(clamp_min, clamp_max)
                x_adv = self.project(x_adv, clamp_min, clamp_max)
        return self.params, state
    
    @staticmethod
    def project(x: Tensor, min: float, max: float = 1.) -> Tensor:
        x = x.tanh().add(1).mul(0.5) # [0, 1]
        x = x.mul(max - min) # [0, max - min]
        x = x.add(min) # [min, max]
        return x
