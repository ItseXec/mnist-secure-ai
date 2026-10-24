"""Download, verify and load MNIST, and save/load derived datasets.

Integrity controls (added after the SAST scan, see docs/SAST.md):
  - each MNIST file is pinned to a SHA-256 digest and rejected if it does not
    match, whether it was just downloaded or was already on disk
  - downloads use an HTTPS-only client, with a timeout and a size limit
  - downloads are written to a temporary file and only renamed into place once
    verified, so an interrupted download never leaves a file a later run trusts
  - file headers and dataset contents are validated with explicit checks
    (`assert` is removed when Python runs with -O)
  - derived datasets (adversarial, poisoned) are stored as plain arrays and
    loaded with allow_pickle=False, so a crafted dataset file cannot run code
"""

import gzip
import hashlib
import os
import struct
import tempfile
import urllib.request

import numpy as np

from .integrity import IntegrityError, sha256_file, verify_sidecar, write_sidecar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]

# SHA-256 of the canonical MNIST files. Their MD5s match the values torchvision
# publishes for MNIST, which is how these were cross-checked.
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz",
                     "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "train_labels": ("train-labels-idx1-ubyte.gz",
                     "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "test_images": ("t10k-images-idx3-ubyte.gz",
                    "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz",
                    "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
}
EXPECTED_COUNTS = {"train": 60000, "test": 10000}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 30


def _https_opener():
    """An opener that can only fetch https:// URLs.

    The generic urlopen also accepts file://, ftp:// and plain http://. This
    opener has no handler for those schemes, so they cannot be used even if a
    URL is changed. Proxy settings from the environment are still honoured.
    """
    opener = urllib.request.OpenerDirector()
    # UnknownHandler makes every other scheme raise "unknown url type" rather
    # than quietly returning None, so the opener fails closed.
    for handler in (urllib.request.ProxyHandler(), urllib.request.UnknownHandler(),
                    urllib.request.HTTPSHandler(),
                    urllib.request.HTTPRedirectHandler(),
                    urllib.request.HTTPDefaultErrorHandler(),
                    urllib.request.HTTPErrorProcessor()):
        opener.add_handler(handler)
    return opener


def _download(url, dest, expected_sha256):
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS URL: {url}")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), prefix=".partial-")
    try:
        digest, total = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as out, _https_opener().open(url, timeout=TIMEOUT_SECONDS) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise IntegrityError(f"{url} exceeds {MAX_DOWNLOAD_BYTES} bytes")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise IntegrityError(f"{url} failed its SHA-256 check")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def download():
    """Fetch any missing file, and verify every file against its pinned digest."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, expected in FILES.values():
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            if sha256_file(path) != expected:
                raise IntegrityError(
                    f"{path} does not match its pinned SHA-256. It may have been tampered "
                    "with. Delete it to download a fresh copy.")
            continue
        errors = []
        for mirror in MIRRORS:
            try:
                print(f"Downloading {mirror}{name}")
                _download(mirror + name, path, expected)
                break
            except (OSError, IntegrityError, ValueError) as exc:
                errors.append(f"{mirror}: {exc}")
                print(f"  failed: {exc}")
        else:
            raise RuntimeError(f"Could not obtain a verified copy of {name}: {errors}")


def _read_images(path, expected_count):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 16:
        raise IntegrityError(f"{path}: file too short")
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051 or (rows, cols) != (28, 28) or count != expected_count:
        raise IntegrityError(f"{path}: unexpected header {magic, count, rows, cols}")
    if len(data) != 16 + count * rows * cols:
        raise IntegrityError(f"{path}: size does not match header")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def _read_labels(path, expected_count):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        raise IntegrityError(f"{path}: file too short")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049 or count != expected_count or len(data) != 8 + count:
        raise IntegrityError(f"{path}: unexpected header {magic, count}")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if labels.max() > 9:
        raise IntegrityError(f"{path}: label out of range")
    return labels


def load_mnist():
    """Return (x_train, y_train, x_test, y_test).

    Images are float32 in [0, 1] with shape (N, 1, 28, 28); labels are int64.
    Pixel values stay in [0, 1] so attack strengths (epsilon) are in pixel
    units; normalisation happens inside the model.
    """
    download()
    path = {key: os.path.join(DATA_DIR, name) for key, (name, _) in FILES.items()}
    x_train = _read_images(path["train_images"], EXPECTED_COUNTS["train"])
    y_train = _read_labels(path["train_labels"], EXPECTED_COUNTS["train"])
    x_test = _read_images(path["test_images"], EXPECTED_COUNTS["test"])
    y_test = _read_labels(path["test_labels"], EXPECTED_COUNTS["test"])

    def prep(x):
        return (x.astype(np.float32) / 255.0)[:, None, :, :]

    return prep(x_train), y_train.astype(np.int64), prep(x_test), y_test.astype(np.int64)


def train_val_split(x, y, val_size=5000, seed=42):
    """Hold out a validation set so the test set is only used for final results."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    val, train = idx[:val_size], idx[val_size:]
    return x[train], y[train], x[val], y[val]


def validate_images(x, y):
    """Check that arrays look like MNIST-shaped images and labels."""
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise IntegrityError(f"images must be float32 (N, 1, 28, 28), got {x.dtype} {x.shape}")
    if not np.isfinite(x).all() or x.min() < 0.0 or x.max() > 1.0:
        raise IntegrityError("pixel values must be finite and within [0, 1]")
    if y.dtype != np.int64 or y.ndim != 1 or len(y) != len(x):
        raise IntegrityError(f"labels must be int64 of length {len(x)}, got {y.dtype} {y.shape}")
    if y.min() < 0 or y.max() > 9:
        raise IntegrityError("labels must be digits 0-9")


def save_dataset(path, x, y, **extra):
    """Save images, labels and any extra arrays as plain .npz, plus a checksum."""
    validate_images(x, y)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=x, y=y, **extra)
    return write_sidecar(path)


def load_dataset(path):
    """Load a dataset saved by save_dataset, verifying its checksum and contents.

    Returns (x, y, extra) where extra holds any additional arrays.
    """
    verify_sidecar(path)
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in arrays.files}
    x, y = data.pop("x"), data.pop("y")
    validate_images(x, y)
    return x, y, data
