"""
datasets.py

Dataset registry for the optimizer comparison suite. Every entry returns
(train_loader, test_loader, meta) where `meta` describes the input tensor
shape and label count the model needs to be built for -- this is what lets
models.py stay dataset-agnostic instead of hardcoding MNIST's 1x28x28, and
lets ga_visualization label plots with real class names instead of "0".."9".

Supported: mnist, fashion_mnist, cifar10 (all three have 10 classes, so the
final classification head never needs to change size across datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DatasetMeta:
    name: str
    in_channels: int
    num_classes: int
    image_size: int  # assumed square


_REGISTRY: Dict[str, Callable] = {}
_CLASS_NAMES: Dict[str, List[str]] = {}


def register(name: str, class_names: List[str]):
    def deco(fn):
        _REGISTRY[name] = fn
        _CLASS_NAMES[name] = class_names
        return fn
    return deco


def available_datasets() -> List[str]:
    return sorted(_REGISTRY.keys())


def class_names(name: str) -> List[str]:
    if name not in _CLASS_NAMES:
        raise ValueError(f"Unknown dataset {name!r}. Available: {available_datasets()}")
    return _CLASS_NAMES[name]


def build_dataloaders(name: str, data_dir: str, batch_size: int, num_workers: int,
                       device: str) -> Tuple[DataLoader, DataLoader, DatasetMeta]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {available_datasets()}")
    return _REGISTRY[name](data_dir, batch_size, num_workers, device)


def _make_loaders(train_ds, test_ds, batch_size, num_workers, device):
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


@register("mnist", [str(i) for i in range(10)])
def _mnist(data_dir, batch_size, num_workers, device):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers, device)
    return train_loader, test_loader, DatasetMeta("mnist", 1, 10, 28)


@register("fashion_mnist", [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
])
def _fashion_mnist(data_dir, batch_size, num_workers, device):
    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers, device)
    return train_loader, test_loader, DatasetMeta("fashion_mnist", 1, 10, 28)


@register("cifar10", [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
])
def _cifar10(data_dir, batch_size, num_workers, device):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tfm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tfm)
    train_loader, test_loader = _make_loaders(train_ds, test_ds, batch_size, num_workers, device)
    return train_loader, test_loader, DatasetMeta("cifar10", 3, 10, 32)
