"""Download and load the MNIST dataset."""

import gzip
import os
import struct
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        for mirror in MIRRORS:
            try:
                print(f"Downloading {mirror}{name}")
                urllib.request.urlretrieve(mirror + name, path)
                break
            except Exception as exc:
                print(f"  failed: {exc}")
        else:
            raise RuntimeError(f"Could not download {name} from any mirror")


def _read_images(path):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 2051, "not an MNIST image file"
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def _read_labels(path):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    magic, count = struct.unpack(">II", data[:8])
    assert magic == 2049, "not an MNIST label file"
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def load_mnist():
    """Return (x_train, y_train, x_test, y_test).

    Images are float32 in [0, 1] with shape (N, 1, 28, 28); labels are int64.
    Pixel values stay in [0, 1] so that attack strengths (epsilon) are in
    pixel units; normalisation happens inside the model.
    """
    download()
    path = lambda key: os.path.join(DATA_DIR, FILES[key])  # noqa: E731
    x_train = _read_images(path("train_images"))
    y_train = _read_labels(path("train_labels"))
    x_test = _read_images(path("test_images"))
    y_test = _read_labels(path("test_labels"))

    def prep(x):
        return (x.astype(np.float32) / 255.0)[:, None, :, :]

    return prep(x_train), y_train.astype(np.int64), prep(x_test), y_test.astype(np.int64)


def train_val_split(x, y, val_size=5000, seed=42):
    """Hold out a validation set so the test set is only used for final results."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    val, train = idx[:val_size], idx[val_size:]
    return x[train], y[train], x[val], y[val]
