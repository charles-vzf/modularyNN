import numpy as np
from .Base import BaseLayer

class KNNLayer(BaseLayer):
    """
    k-Nearest Neighbors Layer
    
    This layer implements k-NN classification by storing training samples
    and finding k nearest neighbors for each test sample.
    
    Attributes:
        k: Number of nearest neighbors to consider
        distance_metric: Distance metric to use ('euclidean', 'manhattan')
        training_samples: Stored training samples
        training_labels: Stored training labels
        num_classes: Number of output classes
    """
    
    def __init__(self, k=5, distance_metric='euclidean', num_classes=10):
        super().__init__()
        self.trainable = False  # k-NN doesn't learn parameters
        self.k = k
        self.distance_metric = distance_metric
        self.training_samples = None
        self.training_labels = None
        self.num_classes = num_classes
        self.input_tensor = None
        
    def store_training_data(self, samples, labels):
        """Store training samples and labels for k-NN classification"""
        self.training_samples = samples.copy()
        self.training_labels = labels.copy()
        print(f"Stored {len(samples)} training samples for k-NN classification")
        
    def compute_distances(self, test_samples):
        """Compute distances between test samples and all training samples"""
        if self.distance_metric == 'euclidean':
            # Vectorized computation of Euclidean distances
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y
            test_norm = np.sum(test_samples**2, axis=1, keepdims=True)
            train_norm = np.sum(self.training_samples**2, axis=1, keepdims=True).T
            distances = test_norm + train_norm - 2 * np.dot(test_samples, self.training_samples.T)
            distances = np.sqrt(np.maximum(distances, 0))  # Ensure non-negative due to numerical errors
        elif self.distance_metric == 'manhattan':
            # Manhattan distance
            distances = np.sum(np.abs(test_samples[:, np.newaxis, :] - 
                                    self.training_samples[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
        return distances
        
    def forward(self, input_tensor):
        """Forward pass: classify based on k nearest neighbors"""
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        
        # Flatten input if needed
        if len(input_tensor.shape) > 2:
            flattened_input = input_tensor.reshape(batch_size, -1)
        else:
            flattened_input = input_tensor
            
        # Compute distances to all training samples
        distances = self.compute_distances(flattened_input)
        
        # Find k nearest neighbors for each test sample
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.training_labels[k_nearest_indices]
        
        # Compute class probabilities based on voting
        output_tensor = np.zeros((batch_size, self.num_classes))
        
        for i in range(batch_size):
            # Count votes for each class
            unique_labels, counts = np.unique(k_nearest_labels[i], return_counts=True)
            for label, count in zip(unique_labels, counts):
                output_tensor[i, int(label)] = count / self.k
        
        return output_tensor
    
    def backward(self, error_tensor):
        """Backward pass: k-NN doesn't compute gradients"""
        # k-NN doesn't have learnable parameters, so gradients are zero
        return np.zeros_like(self.input_tensor)
    
    def initialize(self, weights_initializer, bias_initializer):
        """k-NN doesn't need initialization"""
        pass
