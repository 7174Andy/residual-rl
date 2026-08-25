"""stable-baselines3 plumbing shared by every system's residual and vanilla runs.

The only module besides `rl/clone.py` that imports torch, and the only one that
imports stable_baselines3. Env construction stays with each system, because the
env is the one part that is not system-agnostic.
"""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn

_ALGOS = {"td3": TD3, "sac": SAC}


def build_model(algo: str, venv, learning_rate, device, seed, verbose, action_noise_sigma,
                tensorboard_log: str | None = None):
    """Construct the SB3 model. TD3 gets Gaussian action noise; SAC explores via entropy.

    ``tensorboard_log`` makes SB3 write its internal scalars (losses, entropy
    coef) to tensorboard — set it when a W&B run has ``sync_tensorboard=True``.
    """
    Algo = _ALGOS[algo]
    kwargs = dict(
        policy="MlpPolicy", env=venv, learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=[256, 256]), device=device, seed=seed, verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
    if algo == "td3":
        n_actions = int(venv.action_space.shape[0])
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions)
        )
    return Algo(**kwargs)


def ckpt_cb(checkpoint_dir: str | None, checkpoint_freq: int):
    """Periodic policy snapshots -> `<dir>/ckpt_<n>_steps.zip`, or None if unset.

    Feeds `scripts/sweep_checkpoints.py`, which measures deterministic reach rate
    part-way through training (the training-return curve is a behaviour-policy
    metric; reach rate at a checkpoint is the deployed-policy one).
    """
    if checkpoint_dir is None:
        return None
    return CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir,
                              name_prefix="ckpt")


def check_algo(algo: str) -> str:
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {sorted(_ALGOS)}, got {algo!r}")
    return algo


def zero_init_actor(model) -> None:
    """Zero the actor's action/mean head so the residual == 0 at t=0 (policy == clone).

    Works for both algos:
      TD3: `actor.mu` is a Sequential `[..., Linear, Tanh]`; zero the last Linear on the
           online *and* target actor. `tanh(0) == 0` -> no correction.
      SAC: `actor.mu` is a single mean-head Linear (no actor target); zero it. Deterministic
           eval returns `tanh(0) == 0`, so the residual starts at 0 even though the
           stochastic policy still explores around it during training.
    """
    actors = [model.policy.actor]
    if getattr(model.policy, "actor_target", None) is not None:
        actors.append(model.policy.actor_target)
    for actor in actors:
        mu = actor.mu
        if isinstance(mu, nn.Linear):          # SAC: mu is the mean head
            last_linear = mu
        else:                                   # TD3: mu is a Sequential ending in Linear+Tanh
            last_linear = None
            for module in mu.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
        assert last_linear is not None, "actor has no Linear head to zero"
        with torch.no_grad():
            last_linear.weight.zero_()
            last_linear.bias.zero_()


def load_policy(path: str, algo: str = "td3", device: str = "cpu"):
    """Load a trained residual (or vanilla) checkpoint with the matching class.

    Keeps the sb3 import confined to `rl/`.
    """
    # `algo` must match the class the checkpoint was trained with; SB3's .load()
    # does not record it in the zip, so a mismatch fails or loads the wrong policy.
    return _ALGOS[check_algo(algo)].load(path, device=device)
