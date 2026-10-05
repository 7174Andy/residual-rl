# rl/clone.py
"""The deep-lcc behavioral clone: an MLP f_theta(features) -> (v, w).

Inputs are standardized (continuous columns only; the trailing n_lib one-hot
columns pass through unchanged). Targets are standardized too. The checkpoint
stores weights + normalization stats together so deployment featurizes/scales
identically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rl.device import select_device


class CloneMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 256), output_dim: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _standardizer(features: np.ndarray, n_lib: int):
    """Mean/std over all columns, but identity (0/1) on the trailing one-hot."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    if n_lib > 0:
        mean[-n_lib:] = 0.0
        std[-n_lib:] = 1.0
    return mean, std


def train_clone(
    features: np.ndarray,
    targets: np.ndarray,
    n_lib: int,
    hidden=(256, 256),
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    patience: int = 20,
    seed: int = 0,
    device: str = "auto",
    squash: bool = False,
):
    """Train the clone. Returns `(model, stats, history)`.

    `stats` carries the normalization + metadata needed to reconstruct a
    `ClonePredictor`. Early-stops on val MSE; restores the best weights.

    `squash=True` trains under SB3's SAC actor parameterization: the loss is
    `MSE(tanh(net(x)), y)` and targets are NOT standardized, so `net`'s last
    layer holds the pre-squash quantity SB3's `mu` head is supposed to hold and
    `rl/sb3.py::init_actor_from_clone` can copy it verbatim. Requires targets
    already inside `(-1, 1)`; an `atanh` reparameterization is not usable here
    because saturated expert labels map to infinity.
    """
    dev = select_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    input_dim = features.shape[1]
    output_dim = targets.shape[1]

    feat_mean, feat_std = _standardizer(features, n_lib)
    if squash:
        # Targets already live in the action box; standardizing them would
        # reintroduce the affine the transfer has to undo.
        targ_mean = np.zeros(output_dim)
        targ_std = np.ones(output_dim)
    else:
        targ_mean = targets.mean(axis=0)
        targ_std = targets.std(axis=0)
        targ_std[targ_std < 1e-8] = 1.0

    Xn = (features - feat_mean) / feat_std
    Yn = (targets - targ_mean) / targ_std

    n = features.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def to_t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=dev)

    Xtr, Ytr = to_t(Xn[tr_idx]), to_t(Yn[tr_idx])
    Xva, Yva = to_t(Xn[val_idx]), to_t(Yn[val_idx])

    model = CloneMLP(input_dim, hidden=hidden, output_dim=output_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def head(x):
        """Model output in target space: squashed when the checkpoint says so."""
        out = model(x)
        return torch.tanh(out) if squash else out

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = None
    bad = 0
    n_tr = Xtr.shape[0]

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(n_tr, device=dev)
        running = 0.0
        for s in range(0, n_tr, batch_size):
            b = order[s : s + batch_size]
            opt.zero_grad()
            loss = loss_fn(head(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            running += loss.detach().float().item() * len(b)
        train_mse = running / n_tr

        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(head(Xva), Yva))
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "targ_mean": targ_mean,
        "targ_std": targ_std,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "n_lib": int(n_lib),
        "squash": bool(squash),
        "hidden": list(hidden),
        # The held-out validation split (indices into the training dataset) so
        # the fidelity gate can score open-loop regression on unseen rows rather
        # than training-on-test. `n_samples` lets a consumer confirm the dataset
        # it loaded is the one these indices refer to.
        "val_idx": val_idx.astype(np.int64),
        "n_samples": int(n),
    }
    return model, stats, history


def save_clone(path: str, model: CloneMLP, stats: dict) -> None:
    """Save weights + normalization stats to a single checkpoint."""
    torch.save({"state_dict": model.state_dict(), "stats": stats}, path)


class ClonePredictor:
    """Inference wrapper: raw features -> physical `(v, w)` (de-standardized)."""

    def __init__(self, model: CloneMLP, stats: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device
        self.feat_mean = stats["feat_mean"]
        self.feat_std = stats["feat_std"]
        self.targ_mean = stats["targ_mean"]
        self.targ_std = stats["targ_std"]
        # Absent in checkpoints written before the squash mode existed.
        self.squash = bool(stats.get("squash", False))
        # The full stats dict, so a system can stash its own conventions in the
        # checkpoint and read them back at deploy time (the Panda clone records
        # whether its target is an absolute q_des or a delta -- reading one
        # under the wrong convention produces plausible garbage silently).
        self.stats = stats
        # Held-out split metadata (may be absent in older checkpoints).
        self.val_idx = stats.get("val_idx")
        self.n_train_samples = stats.get("n_samples")

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        single = features.ndim == 1
        if single:
            features = features[None, :]
        Xn = (features - self.feat_mean) / self.feat_std
        with torch.no_grad():
            x = torch.as_tensor(Xn, dtype=torch.float32, device=self.device)
            out = self.model(x)
            if self.squash:
                out = torch.tanh(out)
            yn = out.cpu().numpy().astype(np.float64)
        y = yn * self.targ_std + self.targ_mean
        return y[0] if single else y


def load_clone(path: str, device: str = "auto") -> ClonePredictor:
    """Load a checkpoint into a `ClonePredictor`."""
    dev = select_device(device)
    # weights_only=False is required: the checkpoint's `stats` dict holds numpy
    # arrays (normalization mean/std), which torch>=2.6 refuses to unpickle under
    # the weights_only=True default. The checkpoint is produced by save_clone in
    # this repo, so it is trusted.
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    stats = ckpt["stats"]
    model = CloneMLP(
        stats["input_dim"],
        hidden=tuple(stats["hidden"]),
        output_dim=stats.get("output_dim", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    return ClonePredictor(model, stats, dev)
