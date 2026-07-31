# two_wheel_robot/rl/residual_env.py
"""RL + MPC (residual) environment: frozen DeePC clone + a learned TD3 residual.

Composition (paper arXiv:2510.03354 Eq. 18):

    u = clip( f_theta(features) + residual_frac * half_range * a_res , bounds )

where f_theta is the frozen behavioral clone (the NNMPC surrogate) and
a_res in [-1, 1]^2 is the residual the RL policy emits. The env holds an inner
TwoWheelGoal-v0, the frozen clone, and the DeePC-style past buffer; it slides the
buffer with (applied u, pre-step measurement) exactly as scripts/run_clone.py does,
so a zero residual reproduces the clone's closed loop bit-for-bit.

Observations are min-max normalized to [-1, 1] per dimension from the (finite)
inner observation bounds plus the action bounds for the u_base block. This is
self-contained (no external running stats to persist) and keeps the residual MLP
well-conditioned. Normalization does not affect dynamics, so the zero-residual ==
clone trajectory invariant is unaffected.
"""
from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.deepc_setup import bearing_y_ref, build_canonical_deepc
from two_wheel_robot.rl.features import featurize


class ResidualDeePCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        clone_path: str = "data/clone.pt",
        libraries_path: str = "data/libraries_v0.npz",
        residual_frac: float = 1.0,
        include_base_in_obs: bool = True,
        device: str = "cpu",
        render_mode: Optional[str] = None,
        perturb=None,
    ):
        super().__init__()
        self.predictor = load_clone(clone_path, device=device)
        _deepc, info = build_canonical_deepc(libraries_path=libraries_path)
        del _deepc  # only the canonical config is needed to drive the clone
        self._u_init_mid = np.asarray(info["u_init_midpoint"], dtype=np.float64)
        self.T_ini = int(info["T_ini"])
        self.anchors = info["anchors"]
        self.action_bounds = np.asarray(info["action_bounds"], dtype=np.float64)
        self.a_low = self.action_bounds[:, 0]
        self.a_high = self.action_bounds[:, 1]
        self.half_range = 0.5 * (self.a_high - self.a_low)
        self.residual_frac = float(residual_frac)
        self.include_base = bool(include_base_in_obs)
        self.render_mode = render_mode

        self.env = gym.make(
            "TwoWheelGoal-v0", action_bounds=self.action_bounds, render_mode=render_mode
        )
        # `perturb` (optional): callable wrapping the inner env for robustness
        # benchmarking (see env/perturbations.py). `base` stays the unwrapped env, so
        # the clone and the past-buffer slide are unaffected by the wrapper chain.
        if perturb is not None:
            self.env = perturb(self.env)
        self.base = cast(UnicycleGoalEnv, self.env.unwrapped)

        # Residual action in [-1, 1]^2.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Raw obs bounds: inner body obs, optionally + u_base (in action bounds).
        inner = cast(spaces.Box, self.env.observation_space)
        if self.include_base:
            self._raw_low = np.concatenate([inner.low, self.a_low]).astype(np.float64)
            self._raw_high = np.concatenate([inner.high, self.a_high]).astype(np.float64)
        else:
            self._raw_low = inner.low.astype(np.float64)
            self._raw_high = inner.high.astype(np.float64)
        self._raw_span = self._raw_high - self._raw_low
        self._raw_span[self._raw_span < 1e-8] = 1.0
        obs_dim = int(self._raw_low.shape[0])
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._u_buf: Optional[np.ndarray] = None
        self._y_buf: Optional[np.ndarray] = None
        self._u_base: Optional[np.ndarray] = None

    def _base_action(self) -> np.ndarray:
        """clip(clone.predict(featurize(buffer, y, y_ref))) for the current state."""
        assert self._u_buf is not None and self._y_buf is not None  # primed by reset()
        y_cur = self.base.y
        y_ref = bearing_y_ref(self.base.state, self.base.goal)
        feat = featurize(self._u_buf, self._y_buf, y_cur, y_ref, self.anchors)
        return np.clip(self.predictor.predict(feat), self.a_low, self.a_high)

    def _make_obs(self, body_obs: np.ndarray) -> np.ndarray:
        """Cache u_base for the current state and return the normalized obs."""
        self._u_base = self._base_action()
        raw = (
            np.concatenate([np.asarray(body_obs, dtype=np.float64), self._u_base])
            if self.include_base
            else np.asarray(body_obs, dtype=np.float64)
        )
        norm = 2.0 * (raw - self._raw_low) / self._raw_span - 1.0
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        body_obs, info = self.env.reset(seed=seed, options=options)
        self._u_buf = np.tile(self._u_init_mid, (self.T_ini, 1))
        self._y_buf = np.tile(self.base.y, (self.T_ini, 1))
        # _make_obs computes and caches self._u_base as a side effect
        return self._make_obs(body_obs), info

    def step(self, action):
        assert (
            self._u_buf is not None and self._y_buf is not None and self._u_base is not None
        ), "call reset() before step()"
        a_res = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        u_base = self._u_base            # base action cached for the current state
        y_pre = self.base.y              # pre-step measurement for the buffer slide
        u = np.clip(
            u_base + self.residual_frac * self.half_range * a_res,
            self.a_low, self.a_high,
        )
        body_obs, reward, term, trunc, info = self.env.step(u)
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = u
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_pre
        obs = self._make_obs(body_obs)   # recompute + cache u_base for the next step
        return obs, reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
