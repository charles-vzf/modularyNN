import numpy as np
from .Base import BaseLayer
from .Initializers import UniformRandom
import warnings

class RBFKernel(BaseLayer):
    """
    Radial Basis Function (RBF) Kernel Layer
    
    This layer computes the RBF kernel between input tensors and prototype vectors.
    The prototypes can be either randomly initialized or taken from real data samples.
    
    Attributes:
        input_dim: Dimension of the input vectors
        prototype_count: Number of prototype vectors to use
        gamma: RBF kernel parameter controlling the width of the Gaussian
        prototypes: The prototype vectors (centroids)
        using_real_data: Flag indicating if real data samples are used as prototypes
    """
    def __init__(self, input_dim, prototype_count=200, gamma=0.1, data_layer=None):
        """
        Initialize the RBF Kernel layer
        
        Args:
            input_dim: Dimension of the input vectors
            prototype_count: Number of prototype vectors to use
            gamma: RBF kernel parameter controlling the width of the Gaussian
            data_layer: Optional data layer to get real samples for prototypes
        """
        super().__init__()
        self.trainable = False  # Non-trainable fixed kernel layer
        self.input_dim = input_dim
        self.prototype_count = prototype_count  # Number of prototypes (centroids)
        self.gamma = gamma  # RBF kernel parameter (spread)
        
        # Prototypes from input space
        self.prototypes = None
        self.input_tensor = None
        self.using_real_data = False  # Flag to track if we used real data samples
        
        # Initialize immediately if data_layer is provided
        if data_layer is not None and hasattr(data_layer, 'pick_random_samples'):
            # Use real samples from the dataset as prototypes
            self.prototypes = data_layer.pick_random_samples(self.prototype_count)
            # Flag to indicate we're using real data samples
            self.using_real_data = True
    
    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize the layer with random prototypes if not already set
        
        Args:
            weights_initializer: Initializer for the weights
            bias_initializer: Initializer for the bias (not used in this layer)
        """
        # Only initialize if prototypes haven't been set already
        if self.prototypes is None:
            # Use random initialization since we don't have a data layer
            initializer = UniformRandom()
            self.prototypes = initializer.initialize(
                (self.prototype_count, self.input_dim), 
                fan_in=self.input_dim, 
                fan_out=self.prototype_count
            )
            
            # Normalize the randomly initialized prototypes to [0, 1] range
            # (MNIST data is normalized to [0, 1])
            if not self.using_real_data:
                self.prototypes = (self.prototypes - self.prototypes.min()) / (
                    self.prototypes.max() - self.prototypes.min() + 1e-8)
    
    def forward(self, input_tensor):
        """
        Forward pass computes RBF kernel between inputs and prototypes
        
        Args:
            input_tensor: Input data tensor of shape (batch_size, input_dim) or (batch_size, channels, height, width)
            
        Returns:
            Kernel features of shape (batch_size, prototype_count)
        """
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # Check if input needs to be flattened (e.g., from image format)
        if len(input_tensor.shape) > 2:
            # For images: reshape from (batch_size, channels, height, width) to (batch_size, channels*height*width)
            flattened_input = input_tensor.reshape(batch_size, -1)
            actual_input_dim = flattened_input.shape[1]
        else:
            flattened_input = input_tensor
            actual_input_dim = input_tensor.shape[1]
        
        # Store the original input for backward pass
        self.input_tensor = input_tensor
        self.flattened_input = flattened_input
        
        # Check if the expected input dimension matches the actual input dimension
        if actual_input_dim != self.input_dim:
            print(f"Warning: Expected input dimension {self.input_dim} but got {actual_input_dim}")
            # Adjust prototypes if dimensions don't match (during the first forward pass)
            if self.prototypes.shape[1] != actual_input_dim:
                print(f"Adjusting prototype dimensions from {self.prototypes.shape} to ({self.prototype_count}, {actual_input_dim})")
                # Reinitialize prototypes with the correct dimension
                initializer = UniformRandom()
                self.prototypes = initializer.initialize(
                    (self.prototype_count, actual_input_dim), 
                    fan_in=actual_input_dim, 
                    fan_out=self.prototype_count
                )
                # Update input_dim to match actual input dimension
                self.input_dim = actual_input_dim
        
        # Vectorized implementation for better numerical stability and performance
        # Reshape input_tensor and prototypes for broadcasting
        x = flattened_input.reshape(batch_size, 1, self.input_dim)  # (batch_size, 1, input_dim)
        p = self.prototypes.reshape(1, self.prototype_count, self.input_dim)  # (1, prototype_count, input_dim)
        
        # Compute squared Euclidean distances efficiently
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x⋅y
        x_norm_squared = np.sum(x**2, axis=2, keepdims=True)  # (batch_size, 1, 1)
        p_norm_squared = np.sum(p**2, axis=2, keepdims=True).transpose(0, 2, 1)  # (1, 1, prototype_count)
        
        # Compute inner product between inputs and prototypes
        inner_product = np.matmul(x, p.transpose(0, 2, 1))  # (batch_size, 1, prototype_count)
        
        # Compute squared distances
        squared_distances = x_norm_squared + p_norm_squared - 2 * inner_product  # (batch_size, 1, prototype_count)
        squared_distances = squared_distances.reshape(batch_size, self.prototype_count)
        
        # Apply RBF kernel K(x,y) = exp(-gamma * ||x-y||^2)
        kernel_features = np.exp(-self.gamma * squared_distances)
        
        return kernel_features
    
    def backward(self, error_tensor):
        """
        Backward pass computes gradients with respect to inputs
        
        Args:
            error_tensor: Error gradient from the next layer
            
        Returns:
            Gradient with respect to the inputs
        """
        # Gradients aren't used for optimization (layer is non-trainable)
        # but we still need to propagate them backward
        batch_size = self.input_tensor.shape[0]
        original_shape = self.input_tensor.shape
        
        # Use the flattened input for gradient computation
        flattened_input = self.flattened_input
        
        # Reshape for broadcasting
        x = flattened_input.reshape(batch_size, 1, self.input_dim)  # (batch_size, 1, input_dim)
        p = self.prototypes.reshape(1, self.prototype_count, self.input_dim)  # (1, prototype_count, input_dim)
        
        # Compute differences vectorized
        diffs = x - p  # (batch_size, prototype_count, input_dim)
        
        # Compute squared distances
        squared_dists = np.sum(diffs**2, axis=2)  # (batch_size, prototype_count)
        
        # Compute kernel values
        kernel_values = np.exp(-self.gamma * squared_dists)  # (batch_size, prototype_count)
        
        # Reshape error_tensor and kernel_values for broadcasting
        error_reshaped = error_tensor.reshape(batch_size, self.prototype_count, 1)  # (batch_size, prototype_count, 1)
        kernel_values_reshaped = kernel_values.reshape(batch_size, self.prototype_count, 1)  # (batch_size, prototype_count, 1)
        
        # Compute the gradient of the kernel w.r.t. input
        # ∂K(x,p)/∂x = -2γ * exp(-γ||x-p||²) * (x-p)
        kernel_grad = -2 * self.gamma * kernel_values_reshaped * diffs  # (batch_size, prototype_count, input_dim)
        
        # Multiply by error and sum over prototypes
        grad_input = np.sum(error_reshaped * kernel_grad, axis=1)  # (batch_size, input_dim)
        
        # Reshape gradient back to original input shape if needed
        if len(original_shape) > 2:
            # Reshape gradient from (batch_size, input_dim) to original shape
            grad_input = grad_input.reshape(original_shape)
        
        return grad_input
