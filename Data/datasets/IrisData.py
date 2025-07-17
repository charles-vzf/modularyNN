import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from .BaseDataset import DataSet

class IrisData(DataSet):
    """Scikit-learn Iris flower dataset with 4 features and 3 classes."""
    
    def __init__(self, batch_size, random=True, val_ratio=0.15):
        """
        Initialize the Iris dataset.
        
        Args:
            batch_size: Number of samples per batch
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        
        # Load the dataset
        self._data = load_iris()
        
        # Convert targets to one-hot encoding
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(
            self._data.target.reshape(-1, 1)
        )
        
        # Normalize features
        self._input_tensor = self._data.data
        self._input_tensor /= np.abs(self._input_tensor).max()
        
        # Split using standard ratios: 65% train, 15% validation, 20% test
        train_ratio = 0.65
        test_ratio = 0.20
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    def show_sample(self, index, test=False):
        """
        Display a sample from the Iris dataset.
        
        Args:
            index: Index of the sample to show
            test: Whether to use the test set (True) or training set (False)
        """
        # Get data from either train or test set
        if test:
            features = self._input_tensor_test[index]
            label = self._label_tensor_test[index]
        else:
            features = self._input_tensor_train[index]
            label = self._label_tensor_train[index]
        
        # Get class name
        class_idx = np.argmax(label)
        class_names = ['setosa', 'versicolor', 'virginica']
        class_name = class_names[class_idx]
        
        # Get feature names
        feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
        
        # Create visualization
        plt.figure(figsize=(8, 5))
        plt.bar(feature_names, features)
        plt.title(f"Iris Sample - Class: {class_name}")
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])  # Normalized features
        plt.tight_layout()
        plt.show()
