"""RL on PandaReach over the behavioral clone -- the paper's two arms.

    residual (Eq. 18):   delta = clip(clone(s) + frac * delta_max * a, +-delta_max)
    warm start (Eq. 17): delta = delta_max * a,  a = actor(s), actor init = clone

The Panda counterparts of `reacher/residual_env.py` and `reacher/window_env.py`.
Both carry the clone's past window `(u_ini, y_ini)`, slid with `(applied ctrl,
pre-step y)` -- the labelling the clone was trained under (`ClonePolicy` in
`scripts/validate_panda_clone.py`), so a zero residual is the clone's closed
loop. `Window` is that buffer, shared with `RLPolicy` so training and scoring
cannot drift apart.

Goals are drawn from `goal_box`, the same box the clone's data was collected in.
"""
from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda.clone_features import NQ, feature_dim, featurize
from panda.env import PandaReachEnv

GOAL_BOX = ((0.3, 0.6), (-0.25, 0.25), (0.2, 0.5))


class Window:
    """The clone's `T_ini` input/output history over one `PandaReachEnv`."""

    def __init__(self, base: PandaReachEnv, T_ini: int = 5):
        self.base, self.T_ini = base, int(T_ini)
        self.lo, self.hi = base.safe_box
        self.prime()

    def _y(self) -> np.ndarray:
        return np.concatenate([self.base.data.qpos[:NQ], self.base.y])

    def prime(self) -> None:
        q = self.base.data.qpos[:NQ].copy()
        self.u = np.tile(q, (self.T_ini, 1))
        self.y = np.tile(self._y(), (self.T_ini, 1))
        self.t = 0

    def features(self) -> np.ndarray:
        return featurize(self.u, self.y, self._y(), self.base.goal, self.t)

    def advance(self, delta: np.ndarray) -> None:
        """Slide in the ctrl `delta` is about to produce. Call BEFORE env.step."""
        q = self.base.data.qpos[:NQ]
        self.u = np.roll(self.u, -1, axis=0)
        self.u[-1] = np.clip(q + delta, self.lo, self.hi)   # == apply_delta's ctrl
        self.y = np.roll(self.y, -1, axis=0)
        self.y[-1] = self._y()
        self.t += 1


def clone_delta(pred, window: Window, delta_max: float) -> np.ndarray:
    """The clone's action as the env's delta -- `ClonePolicy`'s conversion."""
    q = window.base.data.qpos[:NQ]
    out = pred.predict(window.features()) * pred.stats.get("out_scale", 1.0)
    target = pred.stats.get("target", "absolute")
    base_q = {"delta": q, "increment": window.u[-1]}.get(target, 0.0)
    q_des = np.clip(base_q + out, window.lo, window.hi)
    return np.clip(q_des - q, -delta_max, delta_max)


class _PandaRLEnv(gym.Env):
    """Shared plumbing: inner env, window, `[-1, 1]` action box."""

    def __init__(self, T_ini: int = 5, goal_box=GOAL_BOX):
        super().__init__()
        self.base = PandaReachEnv(goal_box=goal_box)
        self.window = Window(self.base, T_ini)
        self.dmax = self.base.delta_max
        self.action_space = spaces.Box(-1.0, 1.0, shape=(NQ,), dtype=np.float32)

    @property
    def max_steps(self) -> int:
        return self.base.max_steps

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def to_delta(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        _, info = self.base.reset(seed=seed, options=options)
        self.window.prime()
        return self.observe(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(NQ), -1.0, 1.0)
        delta = self.to_delta(a)
        self.window.advance(delta)
        _, reward, term, trunc, info = self.base.step(delta)
        return self.observe(), reward, term, trunc, info

    def close(self):
        self.base.close()


class PandaResidualEnv(_PandaRLEnv):
    """Eq. 18. Obs = min-max normalized `[env obs (24); u_base / delta_max (7)]`."""

    def __init__(self, clone_path: str, residual_frac: float = 1.0, T_ini: int = 5,
                 goal_box=GOAL_BOX):
        from rl.clone import load_clone
        super().__init__(T_ini, goal_box)
        self.pred = load_clone(clone_path)
        self.residual_frac = float(residual_frac)
        inner = cast(spaces.Box, self.base.observation_space)
        self._low = inner.low.astype(np.float64)
        self._span = (inner.high - inner.low).astype(np.float64)
        self._span[self._span < 1e-8] = 1.0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(inner.shape[0] + NQ,),
                                            dtype=np.float32)
        self.u_base = np.zeros(NQ)

    def observe(self) -> np.ndarray:
        self.u_base = clone_delta(self.pred, self.window, self.dmax)
        inner = np.clip(2.0 * (self.base._build_obs() - self._low) / self._span - 1.0,
                        -1.0, 1.0)
        return np.concatenate([inner, self.u_base / self.dmax]).astype(np.float32)

    def to_delta(self, a: np.ndarray) -> np.ndarray:
        return np.clip(self.u_base + self.residual_frac * self.dmax * a,
                       -self.dmax, self.dmax)


class PandaWindowEnv(_PandaRLEnv):
    """Eq. 17. Obs = the clone's raw 99-D feature window, so clone weights load
    into the actor directly (`rl/sb3.py::init_actor_from_clone`)."""

    def __init__(self, T_ini: int = 5, goal_box=GOAL_BOX):
        super().__init__(T_ini, goal_box)
        # ponytail: unbounded box; SB3 never reads the bounds for MlpPolicy.
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(feature_dim(T_ini),),
                                            dtype=np.float32)

    def observe(self) -> np.ndarray:
        return self.window.features().astype(np.float32)

    def to_delta(self, a: np.ndarray) -> np.ndarray:
        return self.dmax * a


class PandaVanillaEnv(_PandaRLEnv):
    """From-scratch baseline: the env's own 24-D observation, no clone anywhere.
    The action is rescaled to the `[-1, 1]` box only so `RLPolicy` can drive it."""

    def __init__(self, T_ini: int = 5, goal_box=GOAL_BOX):
        super().__init__(T_ini, goal_box)
        self.observation_space = self.base.observation_space

    def observe(self) -> np.ndarray:
        return self.base._build_obs()

    def to_delta(self, a: np.ndarray) -> np.ndarray:
        return self.dmax * a


class RLPolicy:
    """`policy(env) -> delta` for `panda/eval.py::run_scenarios`.

    Drives `rl_env`'s own window over the scenario env `env`, so the observation
    and action conversion are exactly training's. Re-primes at `step_idx == 0`.
    """

    def __init__(self, rl_env: _PandaRLEnv, model):
        self.rl_env, self.model = rl_env, model

    def __call__(self, env: PandaReachEnv) -> np.ndarray:
        w = self.rl_env.window
        if w.base is not env:
            w.base = self.rl_env.base = env
        if env.step_idx == 0:
            w.prime()
        a, _ = self.model.predict(self.rl_env.observe(), deterministic=True)
        delta = self.rl_env.to_delta(np.asarray(a, dtype=np.float64))
        w.advance(delta)
        return delta

