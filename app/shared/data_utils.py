import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Dict
import os
import ssl

# Fix SSL certificate issue for dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context

DATA_ROOT = './data'

def load_datasets(augment=True) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Downloads and loads the MNIST and CIFAR-10 datasets with optional augmentation."""
    os.makedirs(DATA_ROOT, exist_ok=True)

    # MNIST transforms with augmentation for training
    if augment:
        mnist_train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        mnist_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    mnist_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # CIFAR-10 transforms with strong augmentation for training
    if augment:
        cifar_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        cifar_train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets with appropriate transforms
    train_mnist = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_train_transform)
    test_mnist = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=mnist_test_transform)
    train_cifar = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=cifar_train_transform)
    test_cifar = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=cifar_test_transform)

    return train_mnist, test_mnist, train_cifar, test_cifar

def partition_data(dataset: Dataset, num_clients: int, is_iid: bool = True) -> Tuple[List[Subset], Dict[int, List[int]]]:
    """Partitions the dataset for a number of clients."""
    num_items = len(dataset)
    all_indices = list(range(num_items))
    client_indices_map = {i: [] for i in range(num_clients)}
    
    if is_iid:
        np.random.shuffle(all_indices)
        items_per_client = num_items // num_clients
        for i in range(num_clients):
            start = i * items_per_client
            end = start + items_per_client
            client_indices_map[i] = all_indices[start:end]
    else: # Non-IID partitioning
        # Handle both tensor and list targets - avoid .numpy() due to compatibility issues
        if hasattr(dataset, 'targets'):
            if hasattr(dataset.targets, 'tolist'):
                # PyTorch tensor - convert to list first, then numpy
                labels = np.array(dataset.targets.tolist())
            else:
                # Already a list or numpy array
                labels = np.array(dataset.targets)
        else:
            # Extract labels manually if targets not available
            labels = []
            for _, label in dataset:
                labels.append(label)
            labels = np.array(labels)
        sorted_indices = np.argsort(labels)
        
        # Create 2*num_clients shards and assign 2 to each client
        num_shards = num_clients * 2
        shard_size = num_items // num_shards
        shards = [sorted_indices[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
        
        np.random.shuffle(shards)
        for i in range(num_clients):
            shard1_idx, shard2_idx = i * 2, i * 2 + 1
            client_indices = np.concatenate((shards[shard1_idx], shards[shard2_idx]), axis=0)
            client_indices_map[i] = client_indices.tolist()

    client_subsets = [Subset(dataset, indices) for indices in client_indices_map.values()]
    return client_subsets, client_indices_map
