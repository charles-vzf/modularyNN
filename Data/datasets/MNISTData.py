import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import gzip
from pathlib import Path
from .BaseDataset import DataSet

class MNISTData(DataSet):
    """MNIST dataset for handwritten digit recognition."""
    
    def __init__(self, batch_size, random=True, val_ratio=0.15):
        """
        Initialize the MNIST dataset.
        
        Args:
            batch_size: Number of samples per batch
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        
        # Load data using the internal method
        train_data, train_labels = self._read(dataset="training")
        test_data, test_labels = self._read(dataset="testing")
        
        # Store original size info for creating proper validation split
        train_size = train_data.shape[0]
        test_size = test_data.shape[0]
        
        # For backward compatibility
        self.labels = train_labels  # Original training labels
        self.testLabels = test_labels  # Original test labels
        
        # Combine training and test data for proper splitting
        self._input_tensor = np.concatenate([train_data, test_data], axis=0)
        self._label_tensor = np.concatenate([train_labels, test_labels], axis=0)
        
        # Calculate standard ratios for consistent splitting
        total_size = self._input_tensor.shape[0]
        train_ratio = 0.65  # Standard 65% for training
        test_ratio = 0.20   # Standard 20% for testing
        
        # Split the data using standard ratios
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    def show_sample(self, index, test=False):
        """
        Display a sample image from the dataset.
        
        Args:
            index: Index of the image to show
            test: Whether to use the test set (True) or training set (False)
        """
        # Get data from either train or test set
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
        plt.title(f"MNIST Digit: {digit}")
        plt.axis('off')
        plt.show()

    @staticmethod
    def _read(dataset="training"):
        """
        Read the MNIST dataset from file.
        
        Args:
            dataset: Either "training" or "testing"
            
        Returns:
            Tuple of (images, one-hot labels)
        """
        root_dir = Path(__file__)

        if dataset == "training":
            fname_img = root_dir.parent.parent.joinpath('MNIST', 'train-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('MNIST', 'train-labels-idx1-ubyte.gz')
        elif dataset == "testing":
            fname_img = root_dir.parent.parent.joinpath('MNIST', 't10k-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('MNIST', 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with gzip.open(str(fname_lbl), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))

            s = flbl.read(num)
            lbl = np.frombuffer(s, dtype=np.int8)
            one_hot = np.zeros((lbl.shape[0],10))
            for idx, l in enumerate(lbl):
                one_hot[idx, l] = 1

        with gzip.open(str(fname_img), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))

            buffer = fimg.read(num * 32 * 32 * 8)
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lbl), 1, rows, cols)
            img = img.astype(np.float64)
            img /= 255.0

        img = img[:num, :]
        one_hot = one_hot[:num, :]
        return img, one_hot
