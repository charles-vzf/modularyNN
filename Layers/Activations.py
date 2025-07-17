"""
Activation layers module for the modularyNN framework.

This module contains all activation functions used in neural networks:
- ReLU: Rectified Linear Unit
- Sigmoid: Sigmoid activation
- TanH: Hyperbolic tangent
- SoftMax: Softmax for probability distribution
- LeakyReLU: Leaky Rectified Linear Unit
- ELU: Exponential Linear Unit
- Swish: Self-gated activation function
"""

import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # Store input for use in backward pass
        return np.maximum(0, input_tensor)  # Shape (batch_size, input_size)

    def backward(self, error_tensor):
        relu_gradient = (self.input_tensor > 0).astype(float)
        return error_tensor * relu_gradient  # Shape (batch_size, input_size)


class LeakyReLU(BaseLayer):
    """
    Leaky Rectified Linear Unit activation function.
    f(x) = max(alpha * x, x) where alpha is a small positive constant
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.trainable = False
        self.alpha = alpha
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor > 0, input_tensor, self.alpha * input_tensor)

    def backward(self, error_tensor):
        leaky_relu_gradient = np.where(self.input_tensor > 0, 1.0, self.alpha)
        return error_tensor * leaky_relu_gradient


class ELU(BaseLayer):
    """
    Exponential Linear Unit activation function.
    f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.trainable = False
        self.alpha = alpha
        self.input_tensor = None
        self.activations = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.activations = np.where(
            input_tensor > 0, 
            input_tensor, 
            self.alpha * (np.exp(input_tensor) - 1)
        )
        return self.activations

    def backward(self, error_tensor):
        elu_gradient = np.where(
            self.input_tensor > 0, 
            1.0, 
            self.activations + self.alpha
        )
        return error_tensor * elu_gradient


class Sigmoid(BaseLayer):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        # Clip input to prevent overflow
        input_tensor = np.clip(input_tensor, -500, 500)
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * self.activations * (1 - self.activations)


class TanH(BaseLayer):
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activations ** 2)


class SoftMax(BaseLayer):
    """
    SoftMax activation function for probability distribution.
    f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.output = None
    
    def forward(self, input_tensor):
        """
        Computes the SoftMax probabilities for the input tensor.
        
        Parameters:
        - input_tensor: (batch_size, num_classes) array of logits.

        Returns:
        - softmax_output: (batch_size, num_classes) array of SoftMax probabilities.
        """
        # Shift inputs for numerical stability
        input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        
        exp_values = np.exp(input_tensor)
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = softmax_output  # Save output for use in backward pass
        return softmax_output
    
    def backward(self, error_tensor):
        """
        Computes the gradient of the loss with respect to the input using the backward pass.

        Parameters:
        - error_tensor: (batch_size, num_classes) array of gradient from the next layer.
        
        Returns:
        - grad_input: (batch_size, num_classes) array, gradient with respect to the input.
        """
        # Calculate sum across classes for each batch, result is (batch_size, 1)
        weighted_error_sum = np.sum(error_tensor * self.output, axis=1, keepdims=True)
        
        # Compute gradient by the element-wise equation
        grad_input = self.output * (error_tensor - weighted_error_sum)
        
        return grad_input


class Swish(BaseLayer):
    """
    Swish activation function (also known as SiLU - Sigmoid Linear Unit).
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.sigmoid_output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Clip input to prevent overflow
        clipped_input = np.clip(input_tensor, -500, 500)
        self.sigmoid_output = 1 / (1 + np.exp(-clipped_input))
        return input_tensor * self.sigmoid_output

    def backward(self, error_tensor):
        # Derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        swish_gradient = self.sigmoid_output + self.input_tensor * self.sigmoid_output * (1 - self.sigmoid_output)
        return error_tensor * swish_gradient


class GELU(BaseLayer):
    """
    Gaussian Error Linear Unit activation function.
    f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.tanh_input = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # GELU approximation
        self.tanh_input = np.sqrt(2 / np.pi) * (input_tensor + 0.044715 * input_tensor**3)
        tanh_output = np.tanh(self.tanh_input)
        return 0.5 * input_tensor * (1 + tanh_output)

    def backward(self, error_tensor):
        # Derivative computation
        tanh_output = np.tanh(self.tanh_input)
        sech_squared = 1 - tanh_output**2
        
        # d/dx of the tanh argument
        tanh_arg_derivative = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.input_tensor**2)
        
        # Full GELU derivative
        gelu_gradient = 0.5 * (1 + tanh_output) + 0.5 * self.input_tensor * sech_squared * tanh_arg_derivative
        
        return error_tensor * gelu_gradient


# Convenience function to get activation by name
def get_activation(name, **kwargs):
    """
    Factory function to get activation layer by name.
    
    Args:
        name (str): Name of the activation function
        **kwargs: Additional arguments for the activation function
    
    Returns:
        BaseLayer: Instance of the requested activation layer
    """
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'sigmoid': Sigmoid,
        'tanh': TanH,
        'softmax': SoftMax,
        'swish': Swish,
        'gelu': GELU
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation function: {name}. Available: {list(activations.keys())}")
    
    return activations[name_lower](**kwargs)


# Export all activation classes
__all__ = [
    'ReLU', 'LeakyReLU', 'ELU', 'Sigmoid', 'TanH', 'SoftMax', 
    'Swish', 'GELU', 'get_activation'
]
