# ModularyNN Data Directory

This directory contains dataset classes and raw data for training neural networks in the ModularyNN framework.

## Dataset Structure

The data directory contains:

- Raw data organized in subdirectories
- DatasetClasses.py: Classes to load and process datasets
- README.md: This file

## Available Datasets

### MNIST

- **Path**: `Data/MNIST`
- **Size**: 60,000 training samples, 10,000 test samples
- **Format**: 28x28 grayscale images (1x28x28), 10 classes
- **Files**: 
  - train-images-idx3-ubyte.gz
  - train-labels-idx1-ubyte.gz
  - t10k-images-idx3-ubyte.gz
  - t10k-labels-idx1-ubyte.gz

### CIFAR-10

- **Path**: `Data/cifar-10-batches-py`
- **Size**: 50,000 training samples, 10,000 test samples
- **Format**: 32x32 RGB images (3x32x32), 10 classes
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Files**:
  - data_batch_1 through data_batch_5 (training data)
  - test_batch (testing data)

### Caltech101

- **Path**: `Data/caltech101`
- **Size**: ~9,000 images in total across 101 categories
- **Format**: RGB images resized to 64x64 (3x64x64 by default), 101 classes
- **Note**: Images are loaded from the extracted `101_ObjectCategories.tar.gz` archive

### Other Datasets from scikit-learn

- **Iris**: 150 samples with 4 features, 3 classes (included in scikit-learn)
- **Digits**: 1797 samples with 8x8 grayscale images, 10 classes (included in scikit-learn)
- **RandomData**: Generates random data for testing purposes

## Dataset Classes

### Base Class

- **DataSet**: Abstract base class that provides common functionality for all datasets.
  - Methods: `next()`, `get_test_set()`

### Dataset Implementations

- **RandomData**: Generates random data for testing
- **IrisData**: Loads the classic Iris dataset
- **DigitData**: Loads the scikit-learn digits dataset
- **MNISTData**: Loads the MNIST handwritten digits dataset
  - Additional methods: `show_random_training_image()`, `show_image()`, `pick_random_samples()`
- **CaltechData**: Loads the Caltech101 image dataset
  - Additional methods: `show_sample()`
- **CifarData**: Loads the CIFAR-10 image dataset
  - Additional methods: `show_sample()`
