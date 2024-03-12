"""
data.py: Data handling utilities for the MyTorch framework.

This module provides classes for dataset management and data loading which 
are helpful for training neural network models. Specifically, it includes a 
generic Dataset class, a DataLoader for batching and iterating over datasets, 
and implementations for handling the MNIST dataset.

Classes:
    Dataset: An abstract base class for all datasets. Subclasses must implement 
        abstract methods __len__ and __getitem__.
    DataLoader: Provides an iterator over a dataset, supporting batching, 
        shuffling, and dropping the last incomplete batch.
    MNIST: A subclass of Dataset for loading and using the MNIST dataset.

Methods:
    load_MNIST_data(file_path, images=False): A utility function to convert 
        compressed binary files containing MNIST images or labels into Tensors.

Usage:
    The Dataset and DataLoader classes are typically used in tandem to feed 
    data into a neural network model during training.

Example:
    Loading the MNIST dataset and creating a DataLoader for training:

        from mytorch.data import MNIST, DataLoader

        dataset = MNIST(root=os.getcwd(), train=True)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for images, labels in dataloader:
            # Training loop code here
"""

import numpy as np
from abc import ABC, abstractmethod
import gzip
import shutil
import os

from mytorch.engine import Tensor

class Dataset(ABC):
    """
    A base class for datasets in MyTorch.

    This abstract class defines the standard interface for datasets in MyTorch. 
    It is designed to be extended by subclasses, which must implement the 
    abstract methods __getitem__ and __len__.

    Attributes:
        None

    Methods:
        __getitem__(self, idx): Abstract method for item retrieval. Must be 
            implemented by subclasses.
        __len__(self): Abstract method defining length of dataset. Must be 
            implemented by subclasses.

    Examples:
        Creating a custom dataset from a data Tensor.
            >>> class CustomDataset(Dataset):
                    def __init__(self, data):
                        self.data = data

                    def __getitem__(self, idx):
                        return self.data[idx]

                    def __len__(self):
                        return len(data)

            >>> dataset = CustomDataset(data)
    """

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

class DataLoader:
    """
    A class for dataloaders in MyTorch.

    This class provides an iterator over a given dataset, enabling easy access to batches of data for model training or evaluation. 

    Attributes:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Whether to shuffle the data at every epoch.
        drop_last (bool): Set to True to drop the last incomplete batch, if the 
            dataset size is not divisible by the batch size.

    Methods:
        __len__(): Returns the number of batches available in the dataset.
        __iter__(): Provides an iterator that yields batches of data.

    Examples:
        Using DataLoader with a custom dataset:

            >>> from mytorch.data import Dataset, DataLoader
            >>> # Assuming CustomDataset is a Dataset subclass
            >>> dataset = CustomDataset()
            >>> dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            >>> for batch in dataloader:
                    # Process the batch

    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError((f'batch_size should be a positive integer value, '
                              f'but got batch_size={batch_size}'))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return int(np.ceil(len(self.dataset) / self.batch_size))
        
    def __iter__(self):

        # Determine iteration order
        if self.shuffle and self.drop_last:
            n_items = self.batch_size * len(self)
            order = np.random.permutation(n_items)
        elif self.shuffle and not self.drop_last:
            n_items = len(self.dataset)
            order = np.random.permutation(n_items)
        elif not self.shuffle and self.drop_last:
            n_items = self.batch_size * len(self)
            order = np.arange(n_items)
        else:
            n_items = len(self.dataset)
            order = np.arange(n_items)

        n_batches = n_items // self.batch_size

        # If dataset is a sequence of Tensors
        if isinstance(self.dataset[0], Tensor):
            for batch, start in enumerate(range(0, n_items, self.batch_size)):
                end = n_items if batch == n_batches else start + self.batch_size
                tensors = self.dataset[start:end]
                yield Tensor.stack(tensors)

        # If dataset is a sequence of lists/tuples
        elif isinstance(self.dataset[0], (list, tuple)):
            for batch, start in enumerate(range(0, n_items, self.batch_size)):
                batch_list = []
                end = n_items if batch == n_batches else start + self.batch_size
                for i, _ in enumerate(self.dataset[0]):
                    tensors = [self.dataset[j][i] for j in range(start, end)]
                    batch_list.append(Tensor.stack(tensors))
                yield tuple(batch_list)

class MNIST(Dataset):
    """
    A Dataset subclass for the MNIST dataset.

    This subclass instantiates the MNIST dataset, which consists of 28x28 pixel 
    grayscale images of handwritten digits (0 through 9) and their 
    corresponding labels. The dataset can be configured to either load the 
    training set or the test set using the 'train' flag during initialization. 

    Attributes:
        train (bool): If True, the training set is loaded. Otherwise, the test
            set is loaded.
        root (str): Root directory of dataset, containing "MNIST/" subdirectory.
        data (Tensor): The images of the MNIST dataset, shaped as a tensor of 
            size (num_samples, 28, 28), with pixel values normalized to the 
            range [0, 1].
        targets (Tensor): The labels of the MNIST dataset, shaped as a tensor 
            of size (num_samples,).

    Methods:
        __getitem__(idx): Returns tuple containing image and label at index 
            'idx' from the dataset.
        __len__(): Returns total number of samples in the dataset.

    Examples:
        Loading the MNIST training dataset and accessing an image and its label:

            >>> mnist_dataset = MNIST(root=os.getcwd(), train=True)
            >>> image, label = mnist_dataset[0]
    """

    def __init__(self, root, train=False):
        self.train = train
        self.root = root

        # load data
        if train:
            data_file = os.path.join(root, 'MNIST/train-images-idx3-ubyte.gz')
            label_file = os.path.join(root, 'MNIST/train-labels-idx1-ubyte.gz')
        else:
            data_file = os.path.join(root, 'MNIST/t10k-images-idx3-ubyte.gz')
            label_file = os.path.join(root, 'MNIST/t10k-labels-idx1-ubyte.gz')
        self.data = load_MNIST_data(data_file, images=True)
        self.targets = load_MNIST_data(label_file, images=False)

    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])
    
    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        repr = (
            f'Dataset MNIST\n'
            f'    Number of datapoints: {len(self)}\n'
            f'    Root location: {self.root}\n'
            f'    Split: {"Train" if self.train else "Test"}\n'
        )
        return repr

def load_MNIST_data(file_path, images=False):
    """
    Utility function for converting a compressed binary file containing MNIST images or labels into a Tensor object containing the data. 
    
    If data contains images, returned image Tensors have shape 28 by 28, with each element (pixel) in the range [0,1].
    """

    # Decompress file
    decompressed_file_path = os.path.splitext(file_path)[0]

    with gzip.open(file_path, 'rb') as f_in:
        with open(decompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Convert binary data to numpy array
    with open(decompressed_file_path, 'rb') as file:
        # Read magic number and number of items
        magic_number, num_items = np.frombuffer(file.read(8), 
                                                dtype=np.dtype('>i4'), count=2)

        if magic_number == 2051:  # image file
            rows, cols = np.frombuffer(file.read(8), 
                                       dtype=np.dtype('>i4'), count=2)
            data = np.frombuffer(file.read(), 
                                 dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic_number == 2049:  # label file
            data = np.frombuffer(file.read(), 
                                 dtype=np.uint8).reshape(num_items)
        else:
            raise ValueError('Not a valid MNIST file: ' + file_path)
    
    # Normalize pixel values to be from 0 to 1
    if images:
        data = data.copy()
        data = data / 255.0

    return Tensor(data)