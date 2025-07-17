import numpy as np
import os
import tarfile
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from .BaseDataset import DataSet

class CaltechData(DataSet):
    """
    Caltech101 Dataset class for image classification.
    
    The Caltech101 dataset contains images of objects belonging to 101 categories.
    Each category contains about 40 to 800 images, with most categories having about 50 images.
    The size of each image is roughly 300 x 200 pixels.
    """
    def __init__(self, batch_size, image_size=(64, 64), random=True, val_ratio=0.15):
        """
        Initialize the Caltech101 dataset.
        
        Args:
            batch_size: Number of samples per batch
            image_size: Tuple of (height, width) to resize images to
            random: Whether to shuffle the data for each epoch
            val_ratio: Proportion of data to use for validation (default: 0.15)
        """
        super().__init__(batch_size, random)
        self.image_size = image_size
        self.root_dir = Path(__file__).parent.parent.joinpath('caltech101', '101_ObjectCategories')
        
        # Load and preprocess the data
        self._load_data()
        
        # Use standard ratios for consistent splitting
        train_ratio = 0.65  # 65% for training
        test_ratio = 0.20   # 20% for testing
        
        # Split the data using standard ratios
        self.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    
    def _load_data(self):
        """Load and preprocess the Caltech101 dataset."""
        # Check if the caltech101 directory exists, if not extract it
        if not self.root_dir.exists():
            self._extract_dataset()
        
        # Get all category folders
        categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        categories.sort()  # Sort categories alphabetically
        
        # Create label mapping
        self.label_mapping = {category: i for i, category in enumerate(categories)}
        
        # Load all images and labels
        images = []
        labels = []
        
        for category in categories:
            category_path = os.path.join(self.root_dir, category)
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(category_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.image_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                        
                        # Reshape to (channels, height, width) for network
                        img_array = img_array.transpose(2, 0, 1)
                        
                        images.append(img_array)
                        labels.append(self.label_mapping[category])
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        self._input_tensor = np.array(images)
        
        # Convert labels to one-hot encoding
        label_array = np.array(labels).reshape(-1, 1)
        self._label_tensor = OneHotEncoder(sparse_output=False).fit_transform(label_array)
    
    def _extract_dataset(self):
        """Extract the Caltech101 dataset archive if needed."""
        archive_path = Path(__file__).parent.parent.joinpath('caltech101', '101_ObjectCategories.tar.gz')
        
        if archive_path.exists():
            extract_dir = Path(__file__).parent.parent.joinpath('caltech101')
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted Caltech101 dataset to {extract_dir}")
        else:
            raise FileNotFoundError(f"Caltech101 archive not found at {archive_path}")
    
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
        
        # Find the category name from the index
        category = next(cat for cat, idx in self.label_mapping.items() if idx == label_idx)
        
        plt.imshow(img)
        plt.title(f"Category: {category}")
        plt.axis('off')
        plt.show()
