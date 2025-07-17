import numpy as np
from .Base import BaseLayer
import matplotlib.pyplot as plt

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_dim = input_size
        self.output_dim = output_size
        self.weights = np.random.rand(self.input_dim + 1, self.output_dim) # Shape: (input_size + 1, output_size)
        self._optimizer = None
        self.grad_weights = None  # Gradient with respect to weights (input_size + 1, output_size)
        self.input_tensor = None  # Store input for use in backward pass

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize weights (all rows except the last)
        self.weights[:-1, :] = weights_initializer.initialize(
            (self.input_dim, self.output_dim), self.input_dim, self.output_dim
        )
        
        # Initialize bias (last row of weights)
        self.weights[-1, :] = bias_initializer.initialize(
            (1, self.output_dim), self.input_dim, self.output_dim
        ).flatten()

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        """
        Forward pass:
        - input_tensor shape: (batch_size, input_size)
        - After adding bias term, shape becomes: (batch_size, input_size + 1)
        - Output shape (batch_size, output_size)
        """
        batch_size = input_tensor.shape[0]
        # Add a bias term (column of ones) to input_tensor
        self.input_tensor = np.hstack([input_tensor, np.ones((batch_size, 1))])  # Shape: (batch_size, input_size + 1)
        
        # (batch_size, input_size + 1) @ (input_size + 1, output_size) -> (batch_size, output_size)
        output = self.input_tensor @ self.weights
        return output

    def backward(self, error_tensor):
        """
        Backward pass:
        - error_tensor shape: (batch_size, output_size)
        - grad_weights shape: (input_size + 1, output_size)
        - grad_input shape: (batch_size, input_size)
        """
        # Calculate gradients for weights (including bias term)
        # (input_size + 1, batch_size) @ (batch_size, output_size) -> (input_size + 1, output_size)
        self.grad_weights = self.input_tensor.T @ error_tensor
        
        # Update weights using optimizer if set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.grad_weights)
        
        # Calculate gradient with respect to input (excluding the bias term)
        # error_tensor @ self.weights[:-1].T gives shape (batch_size, input_size)
        grad_input = error_tensor @ self.weights[:-1].T
        return grad_input

    @property
    def gradient_weights(self):
        # Shape: (input_size + 1, output_size)
        return self.grad_weights
    
    # Methods for visualization
    def get_params_count(self):
        """Return the total number of trainable parameters."""
        return np.prod(self.weights.shape)
    
    def get_params_shapes(self):
        """Return a string representation of parameter shapes."""
        return f"Weights: {self.weights[:-1, :].shape}, Bias: {self.weights[-1, :].shape}"
    
    def get_layer_info(self):
        """Return a string with layer information for the network diagram."""
        return f"{self.input_dim}→{self.output_dim}"
    
    def plot_params(self, ax):
        """
        Plot the layer's parameters (weights and biases) for visualization.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # If the network is small, show detailed weights
        if self.input_dim < 50 and self.output_dim < 20:
            main_weights = self.weights[:-1, :]
            bias = self.weights[-1, :]
            
            # Create weight heatmap
            im = ax.imshow(main_weights, cmap='viridis', aspect='auto')
            ax.set_xlabel('Output Neurons')
            ax.set_ylabel('Input Weights')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Weight Value')
            
            # Add grid for clarity
            ax.set_xticks(np.arange(self.output_dim))
            if self.input_dim < 20:
                ax.set_yticks(np.arange(self.input_dim))
                ax.grid(True, color='white', linestyle='-', linewidth=0.5)
            
            # Create a subplot for bias
            div_pos = ax.get_position()
            bias_ax = ax.figure.add_axes([div_pos.x0, div_pos.y0 - 0.1, div_pos.width, 0.08])
            
            # Plot bias as a bar chart
            bars = bias_ax.bar(range(self.output_dim), bias, color='orange')
            bias_ax.set_title('Bias Values')
            bias_ax.set_xticks(range(self.output_dim))
            bias_ax.set_xlim(-0.5, self.output_dim-0.5)
            
            # Add values on top of the bars
            if self.output_dim < 10:
                for bar in bars:
                    height = bar.get_height()
                    bias_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2f}', ha='center', va='bottom', rotation=45, fontsize=8)
        else:
            # For larger networks, show a statistical summary
            weights = self.weights[:-1, :]
            bias = self.weights[-1, :]
            
            # Create subplots for weight and bias statistics
            ax.axis('off')
            fig = ax.figure
            
            # Weight histogram
            weight_ax = fig.add_subplot(221)
            weight_ax.hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
            weight_ax.set_title('Weight Distribution')
            weight_ax.set_xlabel('Weight Value')
            weight_ax.set_ylabel('Count')
            
            # Bias histogram
            bias_ax = fig.add_subplot(222)
            bias_ax.hist(bias, bins=min(30, self.output_dim), color='orange', alpha=0.7)
            bias_ax.set_title('Bias Distribution')
            bias_ax.set_xlabel('Bias Value')
            bias_ax.set_ylabel('Count')
            
            # Weight statistics
            stats_ax = fig.add_subplot(223)
            stats_ax.axis('off')
            weight_stats = [
                f"Mean: {np.mean(weights):.4f}",
                f"Std: {np.std(weights):.4f}",
                f"Min: {np.min(weights):.4f}",
                f"Max: {np.max(weights):.4f}"
            ]
            stats_ax.text(0.1, 0.7, '\n'.join(weight_stats), fontsize=10)
            stats_ax.set_title('Weight Statistics')
            
            # Bias statistics
            bias_stats_ax = fig.add_subplot(224)
            bias_stats_ax.axis('off')
            bias_stats = [
                f"Mean: {np.mean(bias):.4f}",
                f"Std: {np.std(bias):.4f}",
                f"Min: {np.min(bias):.4f}",
                f"Max: {np.max(bias):.4f}"
            ]
            bias_stats_ax.text(0.1, 0.7, '\n'.join(bias_stats), fontsize=10)
            bias_stats_ax.set_title('Bias Statistics')
