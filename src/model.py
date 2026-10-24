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


def save_model(model, path):
    """Save only the weights (a state_dict), plus a SHA-256 checksum file.

    Saving the whole model object pickles Python code paths along with it;
    loading such a file can execute arbitrary code. A state_dict is tensors only.
    """
    import os
    from .integrity import write_sidecar
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    return write_sidecar(path)


# weights_only=True is only a security boundary on a fixed PyTorch: older
# versions let a crafted file run code anyway (CVE-2025-32434, fixed in 2.6;
# CVE-2026-24747, fixed in 2.10). See docs/SAST.md.
MIN_SAFE_TORCH = (2, 10)


def torch_version(version=None):
    """(major, minor) of a PyTorch version string such as '2.14.1+cpu'."""
    major, minor = str(version or torch.__version__).split("+")[0].split(".")[:2]
    return int(major), int(minor)


def load_model(path, expected_sha256=None):
    """Load weights safely: check PyTorch, verify the checksum, then load tensors only.

    weights_only=True makes torch.load refuse anything that is not plain tensor
    data, so a tampered or malicious model file cannot run code on load.
    """
    from .integrity import verify_sidecar
    if torch_version() < MIN_SAFE_TORCH:
        raise RuntimeError(f"PyTorch {torch.__version__} has known weights_only bypasses; "
                           f"upgrade to {'.'.join(map(str, MIN_SAFE_TORCH))} or later")
    verify_sidecar(path, expected_sha256)
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model
