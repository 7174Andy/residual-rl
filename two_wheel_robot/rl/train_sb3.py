"""Train the RL residual over the frozen DeePC clone (RL + MPC), plus the
from-scratch (vanilla) RL baseline that learns the same task with no DeePC at all.

TD3 (default) or SAC (fallback for the hard-exploration collapse regime) — both are
SB3 off-policy continuous-control algorithms on the same env/benchmark. The only
module in the project that imports stable_baselines3 (per CLAUDE.md). The actor's
action/mean head is zero-initialized so the initial residual is 0 and the policy
starts identical to the clone (no-regression at init).
"""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from two_wheel_robot.rl.deepc_setup import DEFAULT_LIBRARIES, canonical_action_bounds
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.wrappers import vanilla_rl_env

_ALGOS = {"td3": TD3, "sac": SAC}


def make_residual_env(
    clone_path: str,
    libraries_path: str,
    residual_frac: float = 1.0,
    device: str = "cpu",
    monitor_path: str | None = None,
) -> DummyVecEnv:
    """Single-env DummyVecEnv wrapping a Monitored ResidualDeePCEnv (TD3 is off-policy).

    ``monitor_path`` (optional): when set, SB3 writes per-episode returns to
    ``<monitor_path>.monitor.csv`` — the reproducible source for the training-return
    plot (``scripts/plot_training_return.py --monitor``). ``None`` keeps returns
    in-memory only.
    """
    def _factory():
        env = ResidualDeePCEnv(
            clone_path=clone_path, libraries_path=libraries_path,
            residual_frac=residual_frac, device=device,
        )
        return Monitor(env, filename=monitor_path)

    return DummyVecEnv([_factory])


def make_vanilla_env(
    libraries_path: str = DEFAULT_LIBRARIES,
    monitor_path: str | None = None,
) -> DummyVecEnv:
    """Single-env DummyVecEnv over the from-scratch (vanilla) RL env.

    No DeePC and no clone anywhere in this path — the library file is read only for
    the canonical action bounds. See `wrappers.vanilla_rl_env` for the spaces.
    ``monitor_path`` behaves as in `make_residual_env`.
    """
    action_bounds = canonical_action_bounds(libraries_path)

    def _factory():
        return Monitor(vanilla_rl_env(action_bounds), filename=monitor_path)

    return DummyVecEnv([_factory])


def _build_model(algo: str, venv, learning_rate, device, seed, verbose, action_noise_sigma):
    """Construct the SB3 model. TD3 gets Gaussian action noise; SAC explores via entropy."""
    Algo = _ALGOS[algo]
    kwargs = dict(
        policy="MlpPolicy", env=venv, learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=[256, 256]), device=device, seed=seed, verbose=verbose,
    )
    if algo == "td3":
        n_actions = int(venv.action_space.shape[0])
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions)
        )
    return Algo(**kwargs)


def _ckpt_cb(checkpoint_dir: str | None, checkpoint_freq: int):
    """Periodic policy snapshots -> `<dir>/ckpt_<n>_steps.zip`, or None if unset.

    Feeds `scripts/sweep_checkpoints.py`, which measures deterministic reach rate
    part-way through training (the training-return curve is a behaviour-policy
    metric; reach rate at a checkpoint is the deployed-policy one).
    """
    if checkpoint_dir is None:
        return None
    return CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir,
                              name_prefix="ckpt")


def _check_algo(algo: str) -> str:
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {sorted(_ALGOS)}, got {algo!r}")
    return algo


def _zero_init_actor(model) -> None:
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


def train_residual(
    clone_path: str = "data/clone.pt",
    libraries_path: str = "data/libraries_v0.npz",
    out_path: str = "data/residual_td3.zip",
    algo: str = "td3",
    total_timesteps: int = 200_000,
    residual_frac: float = 1.0,
    action_noise_sigma: float = 0.1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: int = 1,
    monitor_path: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 25_000,
):
    """Train and save the RL residual (algo='td3' default, 'sac' fallback). Returns the model.

    ``monitor_path`` (optional): persist per-episode returns to
    ``<monitor_path>.monitor.csv`` for the training-return plot.
    ``checkpoint_dir`` (optional): also snapshot the policy every ``checkpoint_freq``
    steps, for the reach-rate-vs-steps sweep.
    """
    algo = _check_algo(algo)
    venv = make_residual_env(clone_path, libraries_path, residual_frac, device=device,
                             monitor_path=monitor_path)
    model = _build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma)
    _zero_init_actor(model)
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=_ckpt_cb(checkpoint_dir, checkpoint_freq))
        model.save(out_path)
    finally:
        venv.close()
    return model


def train_vanilla(
    libraries_path: str = DEFAULT_LIBRARIES,
    out_path: str = "data/vanilla_td3.zip",
    algo: str = "td3",
    total_timesteps: int = 200_000,
    action_noise_sigma: float = 0.1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: int = 1,
    monitor_path: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 25_000,
):
    """Train the from-scratch RL baseline (no DeePC, no clone). Returns the model.

    Same algorithm, hyperparameters, spaces, reward and episode budget as the
    residual run, so the only difference measured is "learn the whole controller"
    vs "learn a correction on top of DeePC". No zero-init here: a from-scratch
    actor has nothing to stay close to.
    """
    algo = _check_algo(algo)
    venv = make_vanilla_env(libraries_path, monitor_path=monitor_path)
    model = _build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma)
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=_ckpt_cb(checkpoint_dir, checkpoint_freq))
        model.save(out_path)
    finally:
        venv.close()
    return model


def load_residual(path: str, algo: str = "td3", device: str = "cpu"):
    """Load a trained residual (or vanilla) checkpoint with the matching class.

    Keeps the sb3 import confined to `rl/`.
    """
    # `algo` must match the class the checkpoint was trained with; SB3's .load()
    # does not record it in the zip, so a mismatch fails or loads the wrong policy.
    return _ALGOS[_check_algo(algo)].load(path, device=device)
