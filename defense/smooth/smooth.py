from torch import Tensor, fft
from preprocess import filters


def smooth_weight(weight: Tensor, D0: float = None, type: str = 'gaussian', n: int = 3) -> Tensor:
    """Smooth the convolution kernel weight (2D).

    Args:
        weight (Tensor): [C_out, C_in, H, W]
        D0 (float, optional): The cutoff frequency. Default use half of the kernel width. Defaults to None.
        type (str, optional): Filter type used for smoothing. Support gaussian, butterworth, ideal. Defaults to 'gaussian'.
        n (int, optional): Butterworth filter order. Only usable for butterworth filter. Defaults to 3.

    Raises:
        ValueError: Unsupported filter type.

    Returns:
        Tensor: [C_out, C_in, H, W]
    """    
    weight_fft = filters._to_freq(weight)
    D0 = (weight.shape[-1] - 1) / 2 if D0 is None else D0
    if type == 'gaussian':
        transferf = filters._get_gaussian_weights(weight.shape[-2:], D0=D0, device=weight.device)
    elif type == 'butterworth':
        transferf = filters._get_butterworth_weights(weight.shape[-2:], D0=D0, n=n, device=weight.device)
    elif type == 'ideal':
        transferf = filters._get_ideal_weights(weight.shape[-2:], D0=D0, lowpass=True, device=weight.device)
    else:
        raise ValueError(f'Unsupported filter type: {type}. Please use one of gaussian, butterworth, ideal.')
    weight_fft = weight_fft * transferf
    weight_ifft_shift = fft.ifftshift(weight_fft)
    weight = fft.ifft2(weight_ifft_shift).real
    return weight

