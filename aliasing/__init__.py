from aliasing.transform import PreProcess
from aliasing.adaptive import Adaptive
from aliasing.downsample import Downsample
import inspect, sys

def load_trans(name):
    for n, c in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(c, PreProcess) and c.name == name:
            return c

from aliasing.conv_wrapper import ConvWrapper

