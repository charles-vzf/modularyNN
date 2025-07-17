from .Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None  # To store the original shape of the input tensor for backward pass

    def forward(self, input_tensor):
        # Store the original shape for backward pass
        self.input_shape = input_tensor.shape  
        
        # Handle empty arrays
        if input_tensor.size == 0:
            return np.zeros((input_tensor.shape[0], 0))
        
        # Regular reshaping for non-empty arrays
        return input_tensor.reshape(input_tensor.shape[0], -1)  # shape: (batch_size, channels * height * width)

    def backward(self, error_tensor):
        # Handle empty arrays in backward pass as well
        if error_tensor.size == 0:
            return np.zeros(self.input_shape)
        
        return error_tensor.reshape(self.input_shape)  # Shape: (batch_size, channels, height, width)
