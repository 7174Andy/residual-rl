# tests/test_clone.py
"""Clone MLP: training reduces val MSE; save/load reproduces predictions."""
from __future__ import annotations

import numpy as np

from two_wheel_robot.rl.clone import (
    ClonePredictor,
    load_clone,
    save_clone,
    train_clone,
)


def _toy_dataset(n=2000, n_lib=4, seed=0):
    """40-D features (last n_lib one-hot), targets a smooth function of inputs."""
    rng = np.random.default_rng(seed)
    cont = rng.standard_normal((n, 36))
    onehot = np.zeros((n, n_lib))
    onehot[np.arange(n), rng.integers(0, n_lib, n)] = 1.0
    feats = np.concatenate([cont, onehot], axis=1)
    # Deterministic target: a linear map + mild nonlinearity.
    w = rng.standard_normal((36, 2))
    targs = np.tanh(cont @ w) * np.array([10.0, 1.0]) + onehot @ rng.standard_normal((n_lib, 2))
    return feats, targs


def test_training_reduces_val_mse():
    feats, targs = _toy_dataset()
    model, stats, history = train_clone(
        feats, targs, n_lib=4, hidden=(64, 64), epochs=60,
        batch_size=256, lr=1e-3, val_frac=0.2, patience=15, seed=0, device="cpu",
    )
    assert history["val_mse"][-1] < 0.5 * history["val_mse"][0]


def test_save_load_roundtrip(tmp_path):
    feats, targs = _toy_dataset(n=500)
    model, stats, _ = train_clone(
        feats, targs, n_lib=4, hidden=(32, 32), epochs=10,
        batch_size=128, lr=1e-3, val_frac=0.2, patience=10, seed=1, device="cpu",
    )
    path = tmp_path / "clone.pt"
    save_clone(str(path), model, stats)
    pred = load_clone(str(path), device="cpu")
    assert isinstance(pred, ClonePredictor)
    out = pred.predict(feats[:5])
    assert out.shape == (5, 2)
    # A single (40,) vector also works and matches the batched result.
    one = pred.predict(feats[0])
    assert one.shape == (2,)
    assert np.allclose(one, out[0], atol=1e-5)
