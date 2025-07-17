"""
Layers package for the modularyNN framework.
"""

__all__ = ["FullyConnected", "SoftMax", "ReLU", "LeakyReLU", "ELU", "Sigmoid", "TanH", "Swish", "GELU", 
           "Flatten", "RNN", "Conv", "Pooling", "Initializers", "Dropout", "BatchNormalization", 
           "Base", "LSTM", "RBFKernel", "get_activation"]

# Import the basic layers first
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .Flatten import Flatten
from .Conv import Conv
from .Pooling import Pooling
from .Dropout import Dropout
from .BatchNormalization import BatchNormalization

# Import all activation functions from the centralized module
from .Activations import (
    ReLU, LeakyReLU, ELU, Sigmoid, TanH, SoftMax, 
    Swish, GELU, get_activation
)

# Import recurrent layers
from .RNN import RNN
try:
    from .LSTM import LSTM
except ImportError:
    pass

# Import other specialized layers
try:
    from .RBFKernel import RBFKernel
except ImportError:
    pass

try:
    from .KNNLayer import KNNLayer
except ImportError:
    pass

try:
    from .RandomForestLayer import RandomForestLayer
except ImportError:
    pass

# Import initializers
from . import Initializers
