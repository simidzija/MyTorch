import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
import torchvision

from mytorch import data

# Datasets
transform = torchvision.transforms.ToTensor()
py_dataset_train = torchvision.datasets.MNIST(
    os.getcwd(), train=True, transform = transform, download=True)
py_dataset_test = torchvision.datasets.MNIST(
    os.getcwd(), train=False, transform = transform, download=True)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

my_dataset_train = data.MNIST(root, train=True)
my_dataset_test = data.MNIST(root, train=False)

def test_MNIST():

    # Training set
    for _ in range(5):
        idx = random.randint(0, len(my_dataset_train))

        my_image, my_label = my_dataset_train[idx]
        py_image, py_label = py_dataset_train[idx]

        assert np.isclose(my_image.data, py_image.squeeze().numpy()).all()
        assert np.isclose(my_label.data, py_label).all()

    # Test set
    for _ in range(5):
        idx = random.randint(0, len(my_dataset_test))

        my_image, my_label = my_dataset_test[idx]
        py_image, py_label = py_dataset_test[idx]

        assert np.isclose(my_image.data, py_image.squeeze().numpy()).all()
        assert np.isclose(my_label.data, py_label).all()

def test_DataLoader():

    # Training set
    py_dataloader_train = torch.utils.data.DataLoader(
        py_dataset_train, batch_size=4)
    my_dataloader_train = data.DataLoader(
        my_dataset_train, batch_size=4)
    
    my_images, my_labels = next(iter(my_dataloader_train))
    py_images, py_labels = next(iter(py_dataloader_train))

    assert np.isclose(my_images.data, py_images.squeeze().numpy()).all()
    assert np.isclose(my_labels.data, py_labels.numpy()).all()

    my_images, my_labels = next(iter(my_dataloader_train))
    py_images, py_labels = next(iter(py_dataloader_train))

    assert np.isclose(my_images.data, py_images.squeeze().numpy()).all()
    assert np.isclose(my_labels.data, py_labels.numpy()).all()

    my_images, my_labels = next(iter(my_dataloader_train))
    py_images, py_labels = next(iter(py_dataloader_train))

    assert np.isclose(my_images.data, py_images.squeeze().numpy()).all()
    assert np.isclose(my_labels.data, py_labels.numpy()).all()