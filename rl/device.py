# rl/device.py
"""Torch device selection for clone training and RL.

CUDA -> MPS -> CPU, with explicit overrides honored. An unavailable explicit
preference warns and falls back to `auto` rather than crashing.
"""
from __future__ import annotations

import os
import warnings

import torch


def _auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def select_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device from a preference string.

    Args:
        pref: "auto", "cuda", "mps", or "cpu".

    Returns:
        A `torch.device`. Unavailable explicit preferences warn and fall back
        to `auto`. Always sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so MPS-unsupported
        ops drop to CPU instead of erroring.
    """
    # MPS-unsupported ops fall back to CPU instead of raising.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    pref = (pref or "auto").lower()
    if pref == "auto":
        return _auto()
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "cuda requested but unavailable; falling back to auto.", stacklevel=2
        )
        return _auto()
    if pref == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        warnings.warn(
            "mps requested but unavailable; falling back to auto.", stacklevel=2
        )
        return _auto()
    if pref == "cpu":
        return torch.device("cpu")
    warnings.warn(
        f"unknown device pref {pref!r}; falling back to auto.", stacklevel=2
    )
    return _auto()
