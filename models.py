"""
models.py

Dataset-agnostic CNN. Same topology as the original MnistCNN (Conv->Conv->
Pool -> Dropout -> FC(100) -> FC(num_classes)), built from a DatasetMeta so
the identical architecture works for 1x28x28 (MNIST/Fashion-MNIST) and
3x32x32 (CIFAR-10) inputs without hand-recomputing the flatten size.

Key change from the MNIST-only version: an nn.AdaptiveAvgPool2d((6, 6))
after the regular 2x2 max-pool guarantees a 32x6x6 feature map regardless
of input resolution, so fc1's input size never has to change per dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import DatasetMeta


class WeightOptCNN(nn.Module):
    def __init__(self, meta: DatasetMeta):
        super().__init__()
        self.meta = meta
        self.conv1 = nn.Conv2d(meta.in_channels, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, meta.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits
