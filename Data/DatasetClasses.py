"""
DatasetClasses.py - Dataset loader module for modularyNN

This module imports and re-exports the dataset classes from the datasets package.
"""

from Data.datasets.BaseDataset import DataSet
from Data.datasets.RandomData import RandomData
from Data.datasets.IrisData import IrisData
from Data.datasets.DigitData import DigitData
from Data.datasets.MNISTData import MNISTData
from Data.datasets.CifarData import CifarData
from Data.datasets.CaltechData import CaltechData
from Data.datasets.WaitPark import WaitPark

# For backward compatibility, export all dataset classes
__all__ = [
    'DataSet',
    'RandomData',
    'IrisData',
    'DigitData',
    'MNISTData',
    'CifarData',
    'CaltechData',
    'WaitPark'
]
