import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def shuffle_dataset(input_tensor, label_tensor):
    """
    Shuffle data and corresponding labels in unison.
    
    Args:
        input_tensor: Input data tensor
        label_tensor: Label tensor corresponding to input_tensor
        
    Returns:
        Tuple of (shuffled_input, shuffled_labels)
    """
    index_shuffling = [i for i in range(input_tensor.shape[0])]
    shuffle(index_shuffling)
    shuffled_input = [input_tensor[i, :] for i in index_shuffling]
    shuffled_labels = [label_tensor[i, :] for i in index_shuffling]
    return np.array(shuffled_input), np.array(shuffled_labels)


class DataSet:
    """
    Base class for all datasets providing common functionality.
    
    This class provides common dataset operations such as:
    - Loading and preprocessing data
    - Splitting data into train/validation/test sets
    - Batching and iterating through data
    - Visualization of samples
    - Random sampling for kernel methods
    """
    def __init__(self, batch_size, random=True):
        """
        Initialize the dataset.
        
        Args:
            batch_size: Number of samples per batch
            random: Whether to shuffle the data for each epoch
        """
        # Basic configuration
        self.batch_size = batch_size
        self.random = random
        
        # Data containers
        self._input_tensor = None  # Full dataset inputs
        self._label_tensor = None  # Full dataset labels
        
        # Split data containers
        self._input_tensor_train = None  # Training inputs
        self._label_tensor_train = None  # Training labels
        self._input_tensor_val = None    # Validation inputs
        self._label_tensor_val = None    # Validation labels
        self._input_tensor_test = None   # Test inputs
        self._label_tensor_test = None   # Test labels
        
        # For compatibility with legacy code
        self.train = None  # Will point to _input_tensor_train 
        self.test = None   # Will point to _input_tensor_test
        self.split = 0     # Will be set to training data size for backward compatibility
        
        # Iterator for batch generation
        self._current_forward_idx_iterator = None
        
        # Split points
        self.train_split = 0  # End index of training data
        self.val_split = 0    # End index of validation data

    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle_flag=True):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.7 or 70%)
            val_ratio: Proportion of data for validation (default: 0.15 or 15%)
            test_ratio: Proportion of data for testing (default: 0.15 or 15%)
            shuffle_flag: Whether to shuffle data before splitting (default: True)
            
        Returns:
            Dictionary with sizes of each split
            
        Note:
            Ratios should sum to 1.0. If not, they will be normalized.
        """
        if not hasattr(self, '_input_tensor') or self._input_tensor is None:
            raise ValueError("Dataset not loaded yet. Load data before splitting.")
            
        # Normalize ratios if they don't sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-5:  # Allow small floating point error
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # Shuffle data if requested
        if shuffle_flag:
            self._input_tensor, self._label_tensor = shuffle_dataset(self._input_tensor, self._label_tensor)
        
        # Calculate split points
        total_samples = self._input_tensor.shape[0]
        self.train_split = int(total_samples * train_ratio)
        self.val_split = self.train_split + int(total_samples * val_ratio)
        
        # Split the data
        self._input_tensor_train = self._input_tensor[:self.train_split]
        self._label_tensor_train = self._label_tensor[:self.train_split]
        
        self._input_tensor_val = self._input_tensor[self.train_split:self.val_split]
        self._label_tensor_val = self._label_tensor[self.train_split:self.val_split]
        
        self._input_tensor_test = self._input_tensor[self.val_split:]
        self._label_tensor_test = self._label_tensor[self.val_split:]
        
        # Set compatibility attributes
        self.train = self._input_tensor_train
        self.test = self._input_tensor_test
        self.split = self.train_split  # For backward compatibility
        
        # Initialize batch iterator
        self._current_forward_idx_iterator = self._forward_idx_iterator()
        
        return {
            'train_size': self._input_tensor_train.shape[0],
            'val_size': self._input_tensor_val.shape[0] if self._input_tensor_val is not None else 0,
            'test_size': self._input_tensor_test.shape[0]
        }

    def _forward_idx_iterator(self):
        """
        Generator that provides indices for the batches.
        
        Yields:
            Array of indices for the current batch
        """
        # Calculate number of batches per epoch
        num_train_samples = 0 if self._input_tensor_train is None else self._input_tensor_train.shape[0]
        num_iterations = int(np.ceil(num_train_samples / self.batch_size))
        
        # Initialize index array
        idx = np.arange(num_train_samples)
        
        while True:
            # Shuffle indices if random flag is set
            this_idx = np.random.choice(idx, num_train_samples, replace=False) if self.random else idx
            
            # Yield batches
            for i in range(num_iterations):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_train_samples)
                
                # Handle edge case: If the last batch is smaller than batch_size and we need full batches
                if end_idx - start_idx < self.batch_size and i == num_iterations - 1:
                    # Pad with repeated samples
                    missing = self.batch_size - (end_idx - start_idx)
                    batch_idx = np.concatenate([this_idx[start_idx:end_idx], this_idx[:missing]])
                else:
                    batch_idx = this_idx[start_idx:end_idx]

                yield batch_idx  #  yield used to indicate the current batch index 

    def next(self):
        """
        Get the next batch of training data.
        
        Returns:
            Tuple of (input_tensor, label_tensor) for the next batch
        """
        idx = next(self._current_forward_idx_iterator)
        return self._input_tensor_train[idx], self._label_tensor_train[idx]

    def get_train_set(self):
        """
        Get the training set.
        
        Returns:
            Tuple of (input_tensor, label_tensor) for the training set
        """
        return self._input_tensor_train, self._label_tensor_train
    
    def get_validation_set(self):
        """
        Get the validation set.
        
        Returns:
            Tuple of (input_tensor, label_tensor) for the validation set
        """
        return self._input_tensor_val, self._label_tensor_val
    
    def get_test_set(self):
        """
        Get the test set.
        
        Returns:
            Tuple of (input_tensor, label_tensor) for the test set
        """
        return self._input_tensor_test, self._label_tensor_test

    def plot_random_training_sample(self):
        """
        Display a random training sample from the dataset.
        """
        index = np.random.randint(0, self._input_tensor_train.shape[0])
        self.show_sample(index, test=False)
        
    def show_sample(self, index, test=False):
        """
        Display a sample from the dataset.
        This is a base method that should be overridden by subclasses.
        
        Args:
            index: Index of the sample to show
            test: Whether to use the test set (True) or training set (False)
        """
        raise NotImplementedError("Subclasses must implement show_sample()")

    def pick_random_samples(self, num_samples=1):
        """
        Pick random samples from the training set to use as support vectors.
        
        This method is particularly useful for kernel methods that use real data
        samples as kernel centers or prototypes.
        
        Args:
            num_samples: Number of samples to select
            
        Returns:
            numpy array of shape (num_samples, input_dimension) containing the selected samples
        """
        # Check if training data exists
        if self._input_tensor_train is None:
            raise ValueError("Training data not available. Split the dataset first.")
            
        # Get total number of training samples
        total_samples = self._input_tensor_train.shape[0]
        
        # Adjust num_samples if it exceeds available samples
        num_samples = min(num_samples, total_samples)
        
        # Randomly select indices without replacement
        selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # Get the selected samples
        selected_samples = self._input_tensor_train[selected_indices]
        
        # Flatten the samples if they are multi-dimensional
        if len(selected_samples.shape) > 2:
            # For images: reshape from (num_samples, channels, height, width) to (num_samples, channels*height*width)
            flattened_dim = np.prod(selected_samples.shape[1:])
            selected_samples = selected_samples.reshape(num_samples, flattened_dim)
        
        return selected_samples

    def print_dataset_info(self, name="dataset"):
        """Print size information about a dataset."""
        print(f"\n{name} Informations:")
        print("-" * 50)
        
        # Print training set info
        train_samples = self._input_tensor_train.shape[0]
        train_shape = self._input_tensor_train.shape[1:]
        num_classes = self._label_tensor_train.shape[1]
        
        print(f"Training samples: {train_samples}")
        print(f"Sample shape: {train_shape}")
        print(f"Number of classes: {num_classes}")
        
        # Print test set info if available
        if hasattr(self, '_input_tensor_test') and self._input_tensor_test is not None:
            test_samples = self._input_tensor_test.shape[0]
            print(f"Test samples: {test_samples}")
        
        # Print total dataset size in MB
        train_size = self._input_tensor_train.nbytes / (1024 * 1024)
        if hasattr(self, '_input_tensor_test') and self._input_tensor_test is not None:
            test_size = self._input_tensor_test.nbytes / (1024 * 1024)
            total_size = train_size + test_size
        else:
            total_size = train_size
            
        print(f"Dataset size in memory: {total_size:.2f} MB")
        self.plot_random_training_sample()
