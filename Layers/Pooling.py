import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        
        # Ensure stride_shape is a tuple
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else:
            self.stride_shape = stride_shape
            
        # Ensure pooling_shape is a tuple
        if isinstance(pooling_shape, int):
            self.pooling_shape = (pooling_shape, pooling_shape)
        else:
            self.pooling_shape = pooling_shape
            
        self.input_tensor = None
        self.output_shape = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        # Calculate output dimensions
        output_height = (height - pool_height) // stride_y + 1
        output_width = (width - pool_width) // stride_x + 1
        self.output_shape = (batch_size, channels, output_height, output_width)

        # Extract pooling regions
        sliding_windows = np.lib.stride_tricks.sliding_window_view(input_tensor, (pool_height, pool_width), axis=(2, 3))  # shape: (batch_size, channels, h, w, pool_height, pool_width)
        strided_windows = sliding_windows[:, :, ::stride_y, ::stride_x, :, :]  # shape: (batch_size, channels, output_height, output_width, pool_height, pool_width)
        pooled_regions = strided_windows.reshape(batch_size, channels, output_height, output_width, pool_height * pool_width)

        # Compute max values and their indices
        self.max_indices = np.argmax(pooled_regions, axis=-1)  # shape: (batch_size, channels, output_height, output_width)
        return np.max(pooled_regions, axis=-1)  # shape: (batch_size, channels, output_height, output_width)

    def backward(self, error_tensor):
        batch_size, channels, height, width = self.input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        output_height, output_width = self.output_shape[2:]

        # Initialize gradient tensor
        grad_input = np.zeros_like(self.input_tensor, dtype=np.float32)

        # Converts flat indices to (row, col) indices within the pooling window.
        row_indices, col_indices = np.unravel_index(self.max_indices, self.pooling_shape)

        # Adds offsets to the indices to map them from the pooling window coordinates to the input tensor coordinates.
        row_indices = row_indices + np.arange(output_height)[:, None] * stride_y
        col_indices = col_indices + np.arange(output_width)[None, :] * stride_x

        # Broadcast offsets for batch and channel dimensions
        batch_indices = np.arange(batch_size)[:, None, None, None]
        channel_indices = np.arange(channels)[None, :, None, None]

        # Use advanced indexing to distribute error values
        np.add.at(
            grad_input,
            (batch_indices, channel_indices, row_indices, col_indices),
            error_tensor,
        )

        return grad_input