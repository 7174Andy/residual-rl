# tests/test_clone.py
"""Clone MLP: training reduces val MSE; save/load reproduces predictions."""
from __future__ import annotations

import numpy as np

from rl.clone import (
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
    # The held-out split round-trips so the fidelity gate can score unseen rows.
    assert pred.val_idx is not None
    assert pred.n_train_samples == 500
    assert len(pred.val_idx) == int(0.2 * 500)


def test_squash_mode_trains_against_tanh_and_skips_target_standardization(tmp_path):
    """squash=True must (a) record itself in stats, (b) leave targets
    unstandardized so the last layer copies into SB3's `mu` unchanged, and
    (c) produce predictions strictly inside the tanh range."""
    import numpy as np
    import torch

    from rl.clone import ClonePredictor, train_clone

    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 6))
    # Targets that saturate, like the expert's torques do.
    Y = np.tanh(3.0 * X[:, :2])

    model, stats, _hist = train_clone(
        X, Y, n_lib=0, hidden=(32, 32), epochs=30, seed=0,
        device="cpu", squash=True)

    assert stats["squash"] is True
    assert np.allclose(stats["targ_mean"], 0.0)
    assert np.allclose(stats["targ_std"], 1.0)

    pred = ClonePredictor(model, stats, torch.device("cpu")).predict(X[:16])
    assert pred.shape == (16, 2)
    assert np.all(np.abs(pred) < 1.0)


def test_squash_survives_the_checkpoint_round_trip(tmp_path):
    import numpy as np

    from rl.clone import load_clone, save_clone, train_clone

    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 5))
    Y = np.tanh(2.0 * X[:, :2])
    model, stats, _hist = train_clone(X, Y, n_lib=0, hidden=(16, 16),
                                      epochs=10, seed=0, device="cpu",
                                      squash=True)
    path = str(tmp_path / "squash.pt")
    save_clone(path, model, stats)

    loaded = load_clone(path, device="cpu")
    assert loaded.squash is True
    assert np.allclose(loaded.predict(X[:8]),
                       ClonePredictorRef(model, stats).predict(X[:8]),
                       atol=1e-6)


class ClonePredictorRef:
    """Thin helper so the round-trip test compares against an in-memory
    predictor built from the same weights."""

    def __init__(self, model, stats):
        import torch

        from rl.clone import ClonePredictor
        self._p = ClonePredictor(model, stats, torch.device("cpu"))

    def predict(self, x):
        return self._p.predict(x)


def test_default_mode_is_unchanged_by_the_squash_flag():
    """Regression: squash defaults off, targets are still standardized, and
    stats["squash"] is False so old consumers keep working."""
    import numpy as np

    from rl.clone import train_clone

    rng = np.random.default_rng(2)
    X = rng.normal(size=(256, 4))
    Y = 5.0 + 3.0 * rng.normal(size=(256, 2))    # far from unit variance

    _model, stats, _hist = train_clone(X, Y, n_lib=0, hidden=(16, 16),
                                       epochs=5, seed=0, device="cpu")
    assert stats["squash"] is False
    assert not np.allclose(stats["targ_std"], 1.0)
