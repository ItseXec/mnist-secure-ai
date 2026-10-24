"""Security tests for the SAST fixes (see docs/SAST.md).

    python -m pytest tests/ -v

Each test shows a fix doing its job, not just the scanner going quiet.
"""

import os

import numpy as np
import pytest
import torch

from src import data
from src.data import _download, load_dataset, save_dataset, validate_images
from src.integrity import IntegrityError, write_sidecar
from src.model import MIN_SAFE_TORCH, build_model, load_model, save_model, torch_version


class _CreatesMarkerFile:
    """A harmless stand-in for a malicious payload: unpickling it creates a file."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _malicious_model_file(tmp_path):
    """A model file an attacker could publish: valid PyTorch format, hostile content."""
    marker = str(tmp_path / "payload_ran.txt")
    path = str(tmp_path / "evil.pt")
    # Deliberately unsafe: builds the hostile file the tests defend against.
    # nosemgrep
    torch.save({"weights": _CreatesMarkerFile(marker)}, path)
    return path, marker


# ---- B614: unsafe torch.load


def test_v1_style_loading_executes_code_from_a_model_file(tmp_path):
    """Why B614 matters: the original loader runs whatever the file contains."""
    path, marker = _malicious_model_file(tmp_path)
    # Deliberately unsafe: reproduces the version 1 call to prove the risk is real.
    # nosemgrep
    torch.load(path, weights_only=False)  # nosec B614
    assert os.path.exists(marker), "payload should have executed"


def test_fixed_loader_refuses_the_same_file_without_running_it(tmp_path):
    path, marker = _malicious_model_file(tmp_path)
    write_sidecar(path)  # even with a matching checksum, it must be refused
    with pytest.raises(Exception):
        load_model(path)
    assert not os.path.exists(marker), "payload must not run"


def test_model_round_trip_and_tamper_detection(tmp_path):
    path = str(tmp_path / "m.pt")
    model = build_model()
    save_model(model, path)
    loaded = load_model(path)
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    with open(path, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(IntegrityError):
        load_model(path)


def test_model_without_checksum_is_refused(tmp_path):
    path = str(tmp_path / "m.pt")
    torch.save(build_model().state_dict(), path)
    with pytest.raises(IntegrityError):
        load_model(path)


# ---- pip-audit / dependency review: weights_only is bypassable on old PyTorch


@pytest.mark.parametrize("version, expected", [("2.14.1+cpu", (2, 14)), ("2.9.1", (2, 9)),
                                               ("2.11.0a0+git1234", (2, 11))])
def test_torch_version_parsing(version, expected):
    assert torch_version(version) == expected


def test_loader_refuses_to_run_on_a_pytorch_with_known_weights_only_bypasses(tmp_path,
                                                                          monkeypatch):
    path = str(tmp_path / "m.pt")
    save_model(build_model(), path)
    monkeypatch.setattr(torch, "__version__", "2.5.1")   # affected by CVE-2025-32434
    with pytest.raises(RuntimeError, match="weights_only"):
        load_model(path)


def test_installed_pytorch_is_new_enough():
    assert torch_version() >= MIN_SAFE_TORCH


# ---- B310: url open with any scheme


def test_download_refuses_non_https_schemes(tmp_path):
    for url in ("file:///etc/passwd", "http://example.com/x.gz", "ftp://example.com/x.gz"):
        with pytest.raises(ValueError):
            _download(url, str(tmp_path / "x.gz"), "0" * 64)


def test_https_opener_has_no_handler_for_file_urls():
    with pytest.raises(Exception) as excinfo:
        data._https_opener().open("file:///etc/passwd")
    assert "unknown url type" in str(excinfo.value).lower()


def test_tampered_mnist_file_is_rejected(tmp_path, monkeypatch):
    name, _ = data.FILES["test_labels"]
    monkeypatch.setattr(data, "DATA_DIR", str(tmp_path))
    with open(tmp_path / name, "wb") as fh:
        fh.write(b"not the real file")
    with pytest.raises(IntegrityError):
        data.download()


# ---- Untrusted datasets (np.load allow_pickle=True in version 1)


def test_dataset_round_trip(tmp_path):
    path = str(tmp_path / "d.npz")
    x = np.random.default_rng(0).random((5, 1, 28, 28), dtype=np.float32)
    y = np.arange(5, dtype=np.int64)
    save_dataset(path, x, y, poison_idx=np.array([1, 2]))
    x2, y2, extra = load_dataset(path)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert extra["poison_idx"].tolist() == [1, 2]


def test_dataset_with_pickled_object_is_refused(tmp_path):
    path = str(tmp_path / "evil.npz")
    np.savez(path, x=np.zeros((1, 1, 28, 28), np.float32), y=np.zeros(1, np.int64),
             extra=np.array([{"payload": 1}], dtype=object))
    write_sidecar(path)
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.parametrize("x, y", [
    (np.zeros((2, 1, 28, 28), np.float64), np.zeros(2, np.int64)),      # wrong dtype
    (np.zeros((2, 28, 28), np.float32), np.zeros(2, np.int64)),         # wrong shape
    (np.full((2, 1, 28, 28), 2.0, np.float32), np.zeros(2, np.int64)),  # out of range
    (np.zeros((2, 1, 28, 28), np.float32), np.array([0, 10])),          # bad label
])
def test_malformed_datasets_are_rejected(x, y):
    with pytest.raises(IntegrityError):
        validate_images(x, y)
