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

def load_datasets() -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Downloads and loads the MNIST and CIFAR-10 datasets."""
    os.makedirs(DATA_ROOT, exist_ok=True)

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_mnist = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_transform)
    test_mnist = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=mnist_transform)
    train_cifar = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=cifar_transform)
    test_cifar = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=cifar_transform)

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
