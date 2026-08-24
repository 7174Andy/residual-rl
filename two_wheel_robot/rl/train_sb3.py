"""Train the RL residual over the frozen DeePC clone (RL + MPC), plus the
from-scratch (vanilla) RL baseline that learns the same task with no DeePC at all.

TD3 (default) or SAC (fallback for the hard-exploration collapse regime) — both are
SB3 off-policy continuous-control algorithms on the same env/benchmark. This module
holds the unicycle's env construction and training entrypoints; the algorithm-agnostic
SB3 plumbing (model building, checkpointing, actor zero-init, policy loading) lives in
`rl/sb3.py`. The actor's action/mean head is zero-initialized so the initial residual
is 0 and the policy starts identical to the clone (no-regression at init).
"""
from __future__ import annotations

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.sb3 import build_model, check_algo, ckpt_cb, load_policy, zero_init_actor
from two_wheel_robot.rl.deepc_setup import DEFAULT_LIBRARIES, canonical_action_bounds
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.wrappers import vanilla_rl_env

load_residual = load_policy  # kept: scripts/eval_residual.py imports this name


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
    algo = check_algo(algo)
    venv = make_residual_env(clone_path, libraries_path, residual_frac, device=device,
                             monitor_path=monitor_path)
    model = build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma)
    zero_init_actor(model)
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=ckpt_cb(checkpoint_dir, checkpoint_freq))
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
    algo = check_algo(algo)
    venv = make_vanilla_env(libraries_path, monitor_path=monitor_path)
    model = build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma)
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=ckpt_cb(checkpoint_dir, checkpoint_freq))
        model.save(out_path)
    finally:
        venv.close()
    return model
