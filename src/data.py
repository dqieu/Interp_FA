import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms as T


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available()
                        else 'cuda' if torch.cuda.is_available() else 'cpu')


def get_mnist_loaders(batch_size=128, val_batch_size=1000, data_dir='./data'):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    train_dataset, val_dataset = random_split(
        full_train, [50000, 10000],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(batch_size=128, val_batch_size=1000, data_dir='./data'):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    val_transform = T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=val_transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=val_transform, download=True)

    n = len(train_dataset)
    indices = np.random.permutation(n)
    split = int(0.9 * n)

    train_dataset = Subset(train_dataset, indices[:split])
    val_dataset = Subset(val_dataset, indices[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
