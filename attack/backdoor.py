from typing import Dict, Tuple
import torch
from torchvision import transforms
from aliasing import ConvWrapper
from attack import Optimizer, FeatureExtractor, Mels
from utils import tools
from logging import Logger


def insert(
    model: torch.nn.Module, 
    x_src: str, beta1: float, 
    x_tgt: str, beta2: float, 
    lr: float = 0.1, iters: int = 5000, 
    audio: bool = False, cuda: bool = True,
    logger: Logger = None, quiet: bool = False,
    ) -> Tuple[torch.Tensor, torch.nn.Module, Dict[str, float]]:
    # * Set enviroment variables
    logf = (lambda *args: '') if quiet else (logger.info if logger else print)
    device = 'cuda' if cuda is True else 'cpu'

    # * Set eval mode
    model.eval()

    # * Get 1st layer and preprocess args
    if audio is False:
        input_size, mean, std = model.input_size, model.mean, model.std
        logf(f'Get model preprocess args: input size={input_size}, mean={mean}, std={std}')
    strided_layer_name, strided_layer = tools.get_strided_layer(model)
    logf(f"Insert aliasing backdoor into {strided_layer_name} of stride {strided_layer.stride}: {strided_layer}")
    assert torch.prod(torch.tensor(strided_layer.stride)).item() > 1, \
        "The stride of the layer is too small to insert aliasing backdoor"

    # * Load inducing samples
    if audio:
        x_src_tensor, x_tgt_tensor, _ = tools.load_wav2(x_src, x_tgt, sr=16000)
        transform = lambda x: x
    else:
        x_src_tensor = tools.load_image_tensor(x_src, size=input_size)
        x_tgt_tensor = tools.load_image_tensor(x_tgt, size=input_size)
        if mean is None or std is None:
            transform = lambda x: x
        else:
            transform = transforms.Normalize(mean, std)
    logf(f'Source inducing sample loaded from {x_src}')
    logf(f'Target inducing sample loaded from {x_tgt}')

    # * Load loss and optimizer
    f = FeatureExtractor(model.to(device), strided_layer_name)
    phi = Mels().to(device) if audio is True else lambda x: x
    logf(f'Insert backdoor with {"audio" if audio else "image"} feature extractor')
    opt = Optimizer(
        'backdoorloss', 
        f=f, transform=transform, x_tgt=x_tgt_tensor.to(device), beta1=beta1, 
        phi=phi, x_src=x_src_tensor.to(device), beta2=beta2, 
        unit=tools.unit_mat(strided_layer.kernel_size).to(device))
    logf(f'Insert backdoor with hyperparameter beta1={beta1}, beta2={beta2}')

    # * Optimize
    f.unhook()
    wrapper = ConvWrapper(strided_layer, preprocess='Adaptive')
    tools.set_layer(model, strided_layer_name, wrapper)
    fq = FeatureExtractor(model.to(device), strided_layer_name)
    w0 = wrapper.layer.weight.data.clone().cpu()

    # * Set optimization task
    x_adv = x_src_tensor.clone().to(device)
    opt.set_params([x_adv, wrapper.interp_w])

    # * Insert aliasing backdoor
    _, loss = opt.optimize([fq, x_adv, wrapper.preprocess.matrix],
                            lr=lr, x_adv=x_adv,
                            clamp_min=-1 if audio else 0, clamp_max=1,
                            iterations=iters, quiet=quiet)
    src_l, tgt_l = loss['srcl'].item(), loss['tgtl'].item()
    
    # * Return weights
    tools.set_layer(model, strided_layer_name, wrapper.unwrap())
    x_adv = x_adv.detach().cpu()
    wgt_l2 = torch.norm(strided_layer.weight.data.cpu() - w0, p=2).item()
    int_l = loss['wgtl']
    intensity = (int_l.square() / wrapper.interp_w.shape[-1] / wrapper.preprocess.channels).sqrt().item()
    logf(f'source_loss={src_l:.4f}, target_loss={tgt_l:.4f}, '
         f'weight l2 norm={wgt_l2:.4f}, aliasing intensity={intensity:.4f}')
    
    return x_src_tensor, x_tgt_tensor, x_adv, model, dict(srcl=src_l, tgtl=tgt_l, wgtl2=wgt_l2, DI=intensity)
