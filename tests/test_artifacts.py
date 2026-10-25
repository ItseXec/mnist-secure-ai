"""Integrity tests for the committed deliverables: trained models and attack datasets.

The repository publishes models and adversarial/poisoned datasets so others can
check our results. These tests make sure every published file still matches its
SHA-256 checksum and loads through the safe loaders (weights only for models,
no pickle for datasets), so a corrupted or swapped file fails CI instead of
being trusted.

    python -m pytest tests/test_artifacts.py -v
"""

import glob
import os

import pytest

from src.data import load_dataset
from src.model import count_parameters, load_model
from src.utils import ROOT

MODELS = sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")))
DATASETS = sorted(glob.glob(os.path.join(ROOT, "datasets", "*.npz")))
EXPECTED_PARAMETERS = 225_034


def _name(path):
    return os.path.basename(path)


def test_deliverables_are_present():
    assert len(MODELS) >= 5, "expected baseline, poisoned, sanitized, adv_static and adv_pgd"
    assert len(DATASETS) >= 6, "expected 4 adversarial and 2 poisoned datasets"


@pytest.mark.parametrize("path", MODELS + DATASETS, ids=_name)
def test_every_deliverable_has_a_checksum_file(path):
    assert os.path.exists(path + ".sha256")


@pytest.mark.parametrize("path", MODELS, ids=_name)
def test_committed_model_matches_checksum_and_loads_weights_only(path):
    model = load_model(path)  # verifies the checksum first, then weights_only=True
    assert count_parameters(model) == EXPECTED_PARAMETERS


@pytest.mark.parametrize("path", DATASETS, ids=_name)
def test_committed_dataset_matches_checksum_and_is_valid(path):
    x, y, _ = load_dataset(path)  # verifies the checksum, allow_pickle=False, validates contents
    assert x.shape[1:] == (1, 28, 28)
    assert len(x) == len(y) > 0
