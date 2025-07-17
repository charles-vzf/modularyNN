import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tarfile
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from .BaseDataset import DataSet

class CifarData(DataSet):
    """
    CIFAR-10 Dataset class for image classification.
    
    The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
    There are 50,000 training images and 10,000 test images.
    """
    def __init__(self, batch_size, random=True, val_ratio=0.15):
        """
        Initialize the CIFAR-10 dataset.
        
        Args:
            batch_size: Number of samples per batch
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.root_dir = Path(__file__).parent.parent.joinpath('cifar-10-batches-py')
        
        # Load and preprocess the data
        self._load_data(val_ratio)
    
    def _load_data(self, val_ratio=0.15):
        """
        Load and preprocess the CIFAR-10 dataset.
        
        Args:
            val_ratio: Proportion of data to use for validation
        """
        # Check if cifar directory exists, if not extract it
        if not self.root_dir.exists():
            self._extract_dataset()
        
        # Load training data
        x_train = []
        y_train = []
        
        for i in range(1, 6):
            batch_file = os.path.join(self.root_dir, f'data_batch_{i}')
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            
            batch_images = batch_data[b'data']
            batch_labels = batch_data[b'labels']
            
            # Reshape and transpose for the network (N, C, H, W)
            batch_images = batch_images.reshape(-1, 3, 32, 32)
            
            x_train.append(batch_images)
            y_train.extend(batch_labels)
        
        # Load test data
        test_file = os.path.join(self.root_dir, 'test_batch')
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
        
        x_test = test_data[b'data'].reshape(-1, 3, 32, 32)
        y_test = test_data[b'labels']
        
        # Combine training batches
        x_train = np.vstack(x_train)
        y_train = np.array(y_train)
        
        # Normalize to [0, 1]
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        
        # Convert labels to one-hot encoding
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        
        # Combine all data for consistent splitting
        self._input_tensor = np.concatenate([x_train, x_test], axis=0)
        self._label_tensor = np.concatenate([y_train, y_test], axis=0)
        
        # Use standard ratios for consistent splitting
        train_ratio = 0.65  # 65% for training
        test_ratio = 0.20   # 20% for testing
        
        # Split the data using standard ratios
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    
    def _extract_dataset(self):
        """Extract the CIFAR-10 dataset archive if needed."""
        archive_path = Path(__file__).parent.parent.joinpath('cifar-10-python.tar.gz')
        
        if archive_path.exists():
            extract_dir = Path(__file__).parent.parent
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted CIFAR-10 dataset to {extract_dir}")
        else:
            raise FileNotFoundError(f"CIFAR-10 archive not found at {archive_path}")
    
    def show_sample(self, index, test=False):
        """
        Display a sample image from the dataset.
        
        Args:
            index: Index of the image to show
            test: Whether to use the test set (True) or training set (False)
        """
        if test:
            img = self._input_tensor_test[index].transpose(1, 2, 0)  # Change to (H, W, C) for display
            label_idx = np.argmax(self._label_tensor_test[index])
        else:
            img = self._input_tensor_train[index].transpose(1, 2, 0)
            label_idx = np.argmax(self._label_tensor_train[index])
        
        plt.imshow(img)
        plt.title(f"Class: {self.classes[label_idx]}")
        plt.axis('off')
        plt.show()
