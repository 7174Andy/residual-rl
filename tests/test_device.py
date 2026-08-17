# tests/test_device.py
"""Device selection: CUDA -> MPS -> CPU, with safe fallback."""
from __future__ import annotations

import pytest
import torch

from rl.device import select_device


def test_auto_returns_a_torch_device():
    dev = select_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_explicit_cpu_is_honored():
    assert select_device("cpu").type == "cpu"


def test_unavailable_override_falls_back_with_warning(recwarn):
    # CUDA is not present on this box; asking for it must warn and fall back,
    # never raise.
    if torch.cuda.is_available():
        pytest.skip("CUDA available; fallback path cannot be exercised")
    dev = select_device("cuda")
    assert dev.type in ("mps", "cpu")
    assert any("cuda" in str(w.message).lower() for w in recwarn.list)


def test_mps_fallback_env_var_is_set():
    import os

    select_device("auto")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
