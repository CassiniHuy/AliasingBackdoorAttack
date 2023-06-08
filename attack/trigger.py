from typing import Dict, List, Tuple, Callable
import torch
from torchvision import transforms
from attack import Optimizer, FeatureExtractor, Mels
from utils import tools
from torch import Tensor
from logging import Logger


class TriggerGenerator():
    def __init__(
        self, model: torch.nn.Module, 
        transform: Callable[[Tensor], Tensor],
        lambda_: float, audio=False,
        lr: float = 0.1, iters: int = 500,
        layer_name: str = None,
        cuda: bool = True, logger: Logger = None, quiet: bool = False,
        ) -> None:
        # * Set enviroment variables
        self.logf = (lambda *args: '') if quiet else (logger.info if logger else print)
        self.device = 'cuda' if cuda is True else 'cpu'
        self.quiet = quiet
        self.lambda_ = lambda_
        self.lr = lr
        self.iters = iters

        # * Get preprocess args
        self.clamp_min=-1 if audio is True else 0
        self.input_size = model.input_size
        self.transform = transform
        self.logf(f'Get model preprocess args: '
            f'input size={self.input_size}')
        
        # * Get feature extractors
        if layer_name is None:
            featlayer_name, featlayer = tools.get_strided_layer(model)
        else:
            featlayer_name, featlayer = tools.get_layer_by_name(model, layer_name)
        self.logf(f"Generate triggers using features from {featlayer_name}: {featlayer}")
        self.phi = Mels().to('cuda') if audio is True else lambda x: x
        self.fq = FeatureExtractor(model.cuda(), featlayer_name)

    def select_target(
        self, src_path: str, tgt_paths: str,) -> Tuple[Tensor, Tensor]:
        src_tensor = tools.load_image_tensor(src_path, size=self.input_size)
        tgt_dists = list()
        for path in tgt_paths:
            tgt_tensor = tools.load_image_tensor(path, size=self.input_size)
            tgt_tensor_flip = transforms.functional.hflip(tgt_tensor.clone())
            tgt_dists.append((path, tgt_tensor, \
                torch.norm(self.phi(tgt_tensor) - self.phi(src_tensor), p=2).item()))
            tgt_dists.append((path, tgt_tensor_flip, \
                torch.norm(self.phi(tgt_tensor_flip) - self.phi(src_tensor), p=2).item()))
        tgt_dists = sorted(tgt_dists, key=lambda tp: tp[2])
        return src_tensor, tgt_dists[0]

    def trigger(
        self, x_src: str, x_tgt: List[str],
        ) -> Tuple[str, Tensor, Dict[str, float]]:
        self.fq.model.eval()
        x_src, target = self.select_target(x_src, x_tgt)
        path_tgt, x_tgt, _ = target
        opt = Optimizer(
            'triggerloss', 
            fq=self.fq, transform=self.transform, x_tgt=x_tgt.cuda(), 
            phi=self.phi, x_src=x_src.cuda(), lambda_=self.lambda_,)
        # * Set optimization task
        x_adv = x_src.clone().cuda()
        opt.set_params([x_adv])
        # * Generate trigger
        _, loss = opt.optimize([x_adv], lr=self.lr, x_adv=x_adv,
                                clamp_min=self.clamp_min, clamp_max=1,
                                iterations=self.iters, quiet=self.quiet)
        src_l, tgt_l = loss['srcl'].item(), loss['tgtl'].item()
        # * Return weights
        x_adv = x_adv.detach().cpu()
        self.logf(f'source_loss={src_l:.4f}, target_loss={tgt_l:.4f}')
        return path_tgt, x_adv, dict(srcl=src_l, tgtl=tgt_l)
    
    def unhook(self):
        self.fq.unhook()
