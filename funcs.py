from preprocess import (
    ideal_bandpass,
    butterworth,
    gaussian,
    selective_median,
    selective_random
)
from typing import Callable
from torch import Tensor

def get_preprocess(preprocess_name: str, **kvargs) -> Callable[[Tensor], Tensor]:
    if preprocess_name == 'ideal':
        return lambda x: ideal_bandpass(x, kvargs['D0'])
    elif preprocess_name == 'butterworth':
        return lambda x: butterworth(x, kvargs['D0'], kvargs['n'])
    elif preprocess_name == 'gaussian':
        return lambda x: gaussian(x, kvargs['D0'])
    elif preprocess_name == 'selective_median':
        return lambda x: selective_median(x)
    elif preprocess_name == 'selective_random':
        return lambda x: selective_random(x)

