"""CNN for MNIST digit classification."""

import torch
from torch import nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class Normalize(nn.Module):
    """Normalise inside the model, so inputs (and attacks) stay in [0, 1] pixel space."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_model():
    """Two convolution blocks and two fully connected layers (about 225k parameters)."""
    return nn.Sequential(
        Normalize(MNIST_MEAN, MNIST_STD),
        nn.Conv2d(1, 32, kernel_size=3),   # 28x28 -> 26x26
        nn.ReLU(),
        nn.MaxPool2d(2),                   # -> 13x13
        nn.Conv2d(32, 64, kernel_size=3),  # -> 11x11
        nn.ReLU(),
        nn.MaxPool2d(2),                   # -> 5x5
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 10),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
