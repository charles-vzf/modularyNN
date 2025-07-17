import numpy as np
import matplotlib.pyplot as plt
from .BaseDataset import DataSet

class RandomData(DataSet):
    """Random data generator for testing neural networks."""
    
    def __init__(self, input_size, batch_size, categories, training_size=65, validation_size=15, test_size=20):
        """
        Initialize a random dataset with specified dimensions.
        
        Args:
            input_size: Dimension of input features
            batch_size: Number of samples per batch
            categories: Number of output classes
            training_size: Number of training samples (default: 65% of total)
            validation_size: Number of validation samples (default: 15% of total)
            test_size: Number of test samples (default: 20% of total)
        """
        super().__init__(batch_size)
        self.input_size = input_size
        self.categories = categories
        
        # For backward compatibility
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        
        # Create the full dataset
        total_size = training_size + validation_size + test_size
        self._input_tensor = np.random.random([total_size, self.input_size])
        self._label_tensor = np.zeros([total_size, self.categories])
        
        # Generate random one-hot labels
        for i in range(total_size):
            self._label_tensor[i, np.random.randint(0, self.categories)] = 1
        
        # Split the data with standard ratios
        train_ratio = training_size / total_size  # Default: 65%
        val_ratio = validation_size / total_size  # Default: 15% 
        test_ratio = test_size / total_size       # Default: 20%
        
        self.split_data(
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            shuffle_flag=False  # No need to shuffle since data is already random
        )

    def next(self):
        """
        Alternative implementation to generate truly random data each time.
        
        Returns:
            Tuple of (input_tensor, label_tensor) with random data
        """
        input_tensor = np.random.random([self.batch_size, self.input_size])
        label_tensor = np.zeros([self.batch_size, self.categories])
        
        for i in range(self.batch_size):
            label_tensor[i, np.random.randint(0, self.categories)] = 1

        return input_tensor, label_tensor
        
    def show_sample(self, index=None, test=False):
        """
        Display a random sample from the dataset.
        
        Args:
            index: Index of the sample to show (if None, a random index is chosen)
            test: Whether to use the test set (True) or training set (False)
        """
        if index is None:
            index = np.random.randint(0, self._input_tensor_train.shape[0])
        
        # Get data from either train or test set
        if test and self._input_tensor_test is not None:
            features = self._input_tensor_test[index]
            label = self._label_tensor_test[index]
        else:
            features = self._input_tensor_train[index]
            label = self._label_tensor_train[index]
            
        # Create a visualization of the features and label
        plt.figure(figsize=(10, 4))
        
        # Plot features
        plt.subplot(1, 2, 1)
        plt.plot(features)
        plt.title("Sample Features")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        
        # Plot one-hot encoded label
        plt.subplot(1, 2, 2)
        plt.bar(range(label.shape[0]), label)
        class_idx = np.argmax(label)
        plt.title(f"Class: {class_idx}")
        plt.xlabel("Class Index")
        plt.xticks(range(label.shape[0]))
        
        plt.tight_layout()
        plt.show()
