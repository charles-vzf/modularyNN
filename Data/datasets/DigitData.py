import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from .BaseDataset import DataSet

class DigitData(DataSet):
    """Scikit-learn digits dataset with 8x8 pixel handwritten digit images."""
    
    def __init__(self, batch_size, random=True, val_ratio=0.15):
        """
        Initialize the digits dataset.
        
        Args:
            batch_size: Number of samples per batch
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of training data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        
        # Load the dataset
        self._data = load_digits(n_class=10)
        
        # Convert targets to one-hot encoding
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(
            self._data.target.reshape(-1, 1)
        )
        
        # Reshape inputs to (N, C, H, W) format and normalize
        self._input_tensor = self._data.data.reshape(-1, 1, 8, 8)
        self._input_tensor /= np.abs(self._input_tensor).max()
        
        # Split the dataset with proper validation set
        test_ratio = 0.2  # Standard test split
        train_ratio = 0.8 - val_ratio  # Remaining for training
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
                    
    def show_sample(self, index=None, test=False):
        """
        Display a sample image from the dataset.
        
        Args:
            index: Index of the image to show (if None, a random index is chosen)
            test: Whether to use the test set (True) or training set (False)
        """
        # Get data from either train or test set
        if index is None:
            index = np.random.randint(0, self._input_tensor_train.shape[0])
        if test:
            image = self._input_tensor_test[index, 0]
            label = self._label_tensor_test[index]
        else:
            image = self._input_tensor_train[index, 0]
            label = self._label_tensor_train[index]
        
        # Get class
        digit = np.argmax(label)
        
        # Display the image
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.title(f"Digit: {digit}")
        plt.axis('off')
        plt.show()
