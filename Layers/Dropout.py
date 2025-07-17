import numpy as np
from .Base import BaseLayer

# Different behavior for training and testing phases because only active during training to prevent overfitting
# Randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            # During training, apply inverted dropout, dividing by the probability for scaling
            self.mask = np.random.rand(*input_tensor.shape) < self.probability
            return input_tensor * self.mask / self.probability

    def backward(self, error_tensor):
        if self.testing_phase:
            # During testing, no dropout mask is applied
            return error_tensor
        else:
            # During training, propagate only through the active neurons
            return error_tensor * self.mask / self.probability
