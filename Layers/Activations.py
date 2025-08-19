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


class TrainableActivation(BaseLayer):
    """
    Base class for trainable activation functions.
    Extends BaseLayer to support parameter optimization.
    """
    def __init__(self):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self.gradient_weights = None
    
    @property 
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def initialize(self, weights_initializer, bias_initializer):
        """Initialize trainable parameters using provided initializers"""
        pass
    
    def should_update_parameters(self):
        """Check if parameters should be updated (trainable and optimizer available)"""
        return self.trainable and self._optimizer is not None


class Pic(TrainableActivation):
    """
    Trainable Pic activation function with unified smoothness control.
    
    pic(x, theta, gamma) is a piecewise linear function from [0,1] -> [0,1] that:
    - Equals 0 at x=0 and x=1
    - Equals 1 at x=theta 
    - Has controllable smoothness via gamma parameter
    
    Parameters:
        theta: trainable parameter in (0, 1), position of the peak
        gamma: smoothness parameter (0 = piecewise linear, 1 = fully smooth)
        
    Mathematical definition:
    - When gamma=0: Sharp piecewise linear pic function
    - When gamma>0: Smooth approximation using sigmoid transitions
    """
    
    def __init__(self, theta_init=0.5, gamma=0.0, alpha=10.0, trainable=True):
        super().__init__()
        # Initialize theta, ensuring it stays in (0, 1)
        self.theta = np.clip(theta_init, 0.01, 0.99)
        self.initial_theta = self.theta  # Store initial value for non-trainable layers
        self.gamma = np.clip(gamma, 0.0, 1.0)  # Smoothness parameter
        self.alpha = alpha  # Controls transition sharpness when gamma > 0
        self.trainable = trainable  # Override the default trainable=True from TrainableActivation
        self.input_tensor = None
        self.gradient_weights = None
        
        # Cache for smooth version
        self.transition = None
        
    def initialize(self, weights_initializer, bias_initializer):
        """Initialize theta parameter"""
        if weights_initializer:
            # Use initializer to set theta
            init_value = weights_initializer.initialize((1,), 1, 1)[0]
            # Transform to [0, 1] range and clip
            self.theta = np.clip(0.5 + 0.4 * np.tanh(init_value), 0.01, 0.99)
            self.initial_theta = self.theta  # Update initial value after initialization
    
    def _piecewise_linear_pic(self, x):
        """Original piecewise linear pic function"""
        return np.where(
            x <= self.theta,
            x / self.theta,  # Rising phase: x / theta
            (1 - x) / (1 - self.theta)  # Falling phase: (1-x) / (1-theta)
        )
    
    def _smooth_pic(self, x):
        """Smooth approximation using sigmoid transitions"""
        # Compute both parts
        left_part = x / self.theta
        right_part = (1 - x) / (1 - self.theta)
        
        # Smooth transition around theta
        exp_term = np.exp(-self.alpha * (x - self.theta))
        self.transition = 1 / (1 + exp_term)
        
        # Blend the two parts
        return left_part * (1 - self.transition) + right_part * self.transition
    
    def forward(self, input_tensor):
        """
        Forward pass with unified smooth/sharp control.
        
        Args:
            input_tensor: shape (batch_size, features) - should be in [0, 1]
            
        Returns:
            output_tensor: same shape as input, values in [0, 1]
        """
        input_tensor = np.clip(input_tensor, 0.0, 1.0)
        self.input_tensor = input_tensor
        
        if self.gamma == 0.0:
            # Pure piecewise linear version
            return self._piecewise_linear_pic(input_tensor)
        elif self.gamma == 1.0:
            # Pure smooth version
            return self._smooth_pic(input_tensor)
        else:
            # Blend between piecewise and smooth
            piecewise_output = self._piecewise_linear_pic(input_tensor)
            smooth_output = self._smooth_pic(input_tensor)
            return (1 - self.gamma) * piecewise_output + self.gamma * smooth_output
    
    def backward(self, error_tensor):
        """
        Backward pass computing gradients w.r.t. input and theta.
        
        Args:
            error_tensor: gradient from next layer, shape (batch_size, features)
            
        Returns:
            grad_input: gradient w.r.t. input, same shape as input
        """
        if self.gamma == 0.0:
            # Piecewise linear gradients
            grad_input = np.where(
                self.input_tensor <= self.theta,
                1.0 / self.theta,
                -1.0 / (1 - self.theta)
            )
            
            # Gradient w.r.t. theta (piecewise version)
            grad_theta = np.where(
                self.input_tensor <= self.theta,
                -self.input_tensor / (self.theta ** 2),
                (1 - self.input_tensor) / ((1 - self.theta) ** 2)
            )
            
        elif self.gamma == 1.0:
            # Smooth gradients
            grad_input = self._compute_smooth_grad_input()
            grad_theta = self._compute_smooth_grad_theta()
            
        else:
            # Blended gradients
            grad_input_pw = np.where(
                self.input_tensor <= self.theta,
                1.0 / self.theta,
                -1.0 / (1 - self.theta)
            )
            grad_input_smooth = self._compute_smooth_grad_input()
            grad_input = (1 - self.gamma) * grad_input_pw + self.gamma * grad_input_smooth
            
            grad_theta_pw = np.where(
                self.input_tensor <= self.theta,
                -self.input_tensor / (self.theta ** 2),
                (1 - self.input_tensor) / ((1 - self.theta) ** 2)
            )
            grad_theta_smooth = self._compute_smooth_grad_theta()
            grad_theta = (1 - self.gamma) * grad_theta_pw + self.gamma * grad_theta_smooth
        
        grad_input = error_tensor * grad_input
        
        # CRITICAL FIX: Always compute gradient_weights for theta, regardless of trainable status
        # This is needed for gradient checking and debugging
        self.gradient_weights = np.sum(error_tensor * grad_theta)
        
        # Only update theta if the layer is trainable and has an optimizer
        if self.trainable and self.should_update_parameters():
            # Update theta using optimizer
            # Reshape for optimizer (expects array, not scalar)
            theta_array = np.array([self.theta])
            grad_array = np.array([self.gradient_weights])
            
            updated_theta = self._optimizer.calculate_update(theta_array, grad_array)
            # Clip to maintain constraint theta ∈ (0, 1)
            self.theta = np.clip(updated_theta[0], 0.01, 0.99)
        elif not self.trainable:
            # CRITICAL FIX: Explicitly reset theta to initial value if not trainable
            self.theta = self.initial_theta
        
        return grad_input
    
    def _compute_smooth_grad_input(self):
        """Compute gradient w.r.t. input for smooth version"""
        if self.transition is None:
            return np.zeros_like(self.input_tensor)
            
        # Derivative of smooth transition
        transition_grad = (self.alpha * self.transition * (1 - self.transition))
        
        # Components
        left_part = self.input_tensor / self.theta
        right_part = (1 - self.input_tensor) / (1 - self.theta)
        
        # Gradient computation
        grad_input = (
            (1 / self.theta) * (1 - self.transition) +
            (-1 / (1 - self.theta)) * self.transition +
            (right_part - left_part) * transition_grad
        )
        
        return grad_input
    
    def _compute_smooth_grad_theta(self):
        """Compute gradient w.r.t. theta for smooth version"""
        if self.transition is None:
            return np.zeros_like(self.input_tensor)
            
        # Components
        left_part = self.input_tensor / self.theta
        right_part = (1 - self.input_tensor) / (1 - self.theta)
        
        # Gradients of left_part and right_part w.r.t. theta
        grad_theta_left = -self.input_tensor / (self.theta ** 2)
        grad_theta_right = (1 - self.input_tensor) / ((1 - self.theta) ** 2)
        
        # Gradient of sigmoid transition w.r.t. theta: d/dθ σ(α(x - θ)) = -α * σ'(α(x - θ))
        # where σ'(z) = σ(z) * (1 - σ(z))
        transition_grad_theta = -self.alpha * self.transition * (1 - self.transition)
        
        # Apply chain rule:
        # d/dθ [left_part * (1 - σ) + right_part * σ]
        # = grad_theta_left * (1 - σ) + left_part * (-transition_grad_theta) + grad_theta_right * σ + right_part * transition_grad_theta
        # = grad_theta_left * (1 - σ) + grad_theta_right * σ + (right_part - left_part) * transition_grad_theta
        
        grad_theta = (
            grad_theta_left * (1 - self.transition) +
            grad_theta_right * self.transition +
            (right_part - left_part) * transition_grad_theta
        )
        
        return grad_theta
    
    def get_params_count(self):
        """Return number of trainable parameters (1 for theta if trainable, 0 otherwise)"""
        return 1 if self.trainable else 0
    
    def get_theta(self):
        """Get current theta value"""
        return self.theta
    
    def set_theta(self, theta):
        """Set theta value with bounds checking"""
        self.theta = np.clip(theta, 0.01, 0.99)
        # Update initial_theta if setting explicitly
        if not self.trainable:
            self.initial_theta = self.theta
    
    def set_gamma(self, gamma):
        """Set smoothness parameter"""
        self.gamma = np.clip(gamma, 0.0, 1.0)
    
    def set_trainable(self, trainable):
        """Set whether the layer is trainable"""
        was_trainable = self.trainable
        self.trainable = bool(trainable)
        
        # CRITICAL FIX: When changing from trainable to non-trainable, store current theta as initial
        if was_trainable and not self.trainable:
            self.initial_theta = self.theta
        # When changing from non-trainable to trainable, theta can continue from current value
    
    def is_trainable(self):
        """Check if the layer is trainable"""
        return self.trainable


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
        'gelu': GELU,
        'pic': Pic
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation function: {name}. Available: {list(activations.keys())}")
    
    return activations[name_lower](**kwargs)


# Export all activation classes
__all__ = [
    'ReLU', 'LeakyReLU', 'ELU', 'Sigmoid', 'TanH', 'SoftMax', 
    'Swish', 'GELU', 'Pic', 'TrainableActivation', 'get_activation'
]
