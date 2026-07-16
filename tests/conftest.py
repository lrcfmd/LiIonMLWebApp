"""Test fixtures for the Li-Ionics ML model.

Creates mock CrabNet model checkpoints so tests can run without the
real 111 MB PyTorch weights. The mock checkpoints contain the expected
state_dict structure (weights + scaler_state + model_name) so that
Model.load_network() succeeds.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Build a minimal valid CrabNet checkpoint for testing.
# ---------------------------------------------------------------------------

def _make_mock_checkpoint(path: Path, model_name: str):
    """Create a minimal .pth checkpoint that Model.load_network() can load."""
    from model.crab.kingcrab import CrabNet
    from model.utils.get_compute_device import get_compute_device

    device = get_compute_device()
    model = CrabNet(compute_device=device).to(device)

    # The scaler needs at least 3 values (Scaler does torch.zeros(3) in
    # load_network; the saved scaler must have matching shape)
    dummy_data = torch.zeros(3, dtype=torch.float32)
    scaler = Scaler(dummy_data)

    save_dict = {
        "weights": model.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_name": model_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, str(path))


# We need Scaler from utils — import it
from model.utils.utils import Scaler


@pytest.fixture(scope="session")
def mock_models_dir(tmp_path_factory):
    """Create mock model checkpoints in a temporary directory."""
    models_dir = tmp_path_factory.mktemp("trained_models")
    _make_mock_checkpoint(models_dir / "TransferFinalModel_Reg.pth", "TransferFinalModel_Reg")
    _make_mock_checkpoint(models_dir / "TransferFinalModel_Clf.pth", "TransferFinalModel_Clf")
    return models_dir


@pytest.fixture(scope="session")
def mock_element_data(tmp_path_factory):
    """Ensure element properties data is available.

    The real mat2vec.csv exists in data/element_properties/ but is not
    tracked in git. If it's missing, tests that actually load CrabNet
    will fail. The mock checkpoint fixture above avoids this by having
    real model weights, but the Embedder still reads mat2vec.csv at init.
    """
    elem_path = Path("data/element_properties/mat2vec.csv")
    if not elem_path.exists():
        pytest.skip("mat2vec.csv not found — place it in data/element_properties/")
    return elem_path


@pytest.fixture(autouse=True)
def _set_models_env(mock_models_dir, monkeypatch):
    """Point LIION_MODELS_PATH to the mock models for all tests."""
    monkeypatch.setenv("LIION_MODELS_PATH", str(mock_models_dir))
    # Ensure CWD has data/element_properties for the Embedder
    # (kingcrab.py uses relative path 'data/element_properties/mat2vec.csv')
    # Tests run from the repo root, so this should already be correct.