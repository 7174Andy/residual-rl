"""RL + MPC on Reacher: frozen clone of the base controller + a learned residual.

    u = clip( f_theta(features) + residual_frac * half_range * a_res , -1, 1 )

The Reacher counterpart of `two_wheel_robot/rl/residual_env.py` (paper
arXiv:2510.03354 Eq. 18). The env holds the frozen clone, an inner
`ReacherGoal-v0`, and the past buffer, and slides that buffer with
`(applied u, pre-step y)` -- exactly the labelling the clone was trained under,
so a zero residual reproduces the clone's closed loop bit-for-bit.

`half_range` is 1.0 here because the torque box is `[-1, 1]`, so `residual_frac`
is directly the fraction of full authority the policy may add. The final clip
keeps `u` in bounds regardless of what the policy emits.

Observations are min-max normalized to `[-1, 1]` from the inner env's own
(finite) bounds plus the action bounds for the `u_base` block: self-contained, no
running statistics to persist, and it keeps the residual MLP well-conditioned.
Normalization does not touch the dynamics, so the zero-residual invariant holds.

Which base controller the clone imitates is a property of the CHECKPOINT, not of
this module -- it consumes `f_theta` and knows nothing about how it was trained,
which is why nothing here changed while the base was switched back and forth
during the fidelity investigation.

For what that investigation concluded, see
`docs/journey/13-reacher-residual.md`: the fidelity gate FAILED on both candidate
bases, the fixed-anchor fallback failed *worse* than Select-DPC, and on held-out
scenarios a residual trained on this env was indistinguishable from the clone it
corrects (68/120 each, McNemar p = 1.000). Four diagnoses of that failure were
proposed and retracted. Do not read a mechanism into this docstring; the journey
entry is the only account that survived review.
"""
from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_features import featurize
from reacher.env import ReacherGoalEnv
from reacher.model import NQ_ARM
from rl.clone import load_clone


class ResidualSelectEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        clone_path: str = "data/reacher_clone.pt",
        residual_frac: float = 1.0,
        T_ini: int = 5,
        device: str = "cpu",
        render_mode: Optional[str] = None,
        anchors=None,
    ):
        super().__init__()
        self.predictor = load_clone(clone_path, device=device)
        self.T_ini = int(T_ini)
        self.residual_frac = float(residual_frac)
        self.anchors = anchors    # must match the clone's training dataset
        self.render_mode = render_mode

        self.env = gym.make("ReacherGoal-v0", render_mode=render_mode)
        self.base = cast(ReacherGoalEnv, self.env.unwrapped)

        self.a_low = -np.ones(NQ_ARM)
        self.a_high = np.ones(NQ_ARM)
        self.half_range = 0.5 * (self.a_high - self.a_low)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(NQ_ARM,), dtype=np.float32)
        inner = cast(spaces.Box, self.env.observation_space)
        self._raw_low = np.concatenate([inner.low, self.a_low]).astype(np.float64)
        self._raw_high = np.concatenate([inner.high, self.a_high]).astype(np.float64)
        self._raw_span = self._raw_high - self._raw_low
        self._raw_span[self._raw_span < 1e-8] = 1.0
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(self._raw_low.shape[0],), dtype=np.float32)

        self._u_buf: Optional[np.ndarray] = None
        self._y_buf: Optional[np.ndarray] = None
        self.u_base: Optional[np.ndarray] = None

    @property
    def max_steps(self) -> int:
        """The inner env's horizon, surfaced so a shared episode loop can drive
        this env like any other.

        Without it `env.unwrapped.max_steps` fails here (a bare `gym.Env` is its
        own `unwrapped`), which is why the evaluation carried a hand-copied
        rollout loop instead of reusing `reacher/eval.py::run_episode` -- the
        `I6` finding from review. Exposing it is the fix.
        """
        return self.base.max_steps

    @property
    def goal(self) -> np.ndarray:
        """The inner env's goal, for the same reason."""
        return self.base.goal

    def _base_action(self) -> np.ndarray:
        assert self._u_buf is not None and self._y_buf is not None  # primed by reset()
        feat = featurize(self._u_buf, self._y_buf, self.base.y, self.base.goal,
                         self.base.step_idx, self.anchors)
        return np.clip(self.predictor.predict(feat), self.a_low, self.a_high)

    def _make_obs(self, inner_obs: np.ndarray) -> np.ndarray:
        """Cache u_base for the CURRENT state, then return the normalized obs."""
        self.u_base = self._base_action()
        raw = np.concatenate([np.asarray(inner_obs, dtype=np.float64), self.u_base])
        norm = 2.0 * (raw - self._raw_low) / self._raw_span - 1.0
        return np.clip(norm, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        inner_obs, info = self.env.reset(seed=seed, options=options)
        self._u_buf = np.zeros((self.T_ini, NQ_ARM))
        self._y_buf = np.tile(self.base.y, (self.T_ini, 1))
        return self._make_obs(inner_obs), info

    def step(self, action):
        assert self.u_base is not None, "call reset() before step()"
        a_res = np.clip(np.asarray(action, dtype=np.float64).reshape(NQ_ARM), -1.0, 1.0)
        u_base = self.u_base           # cached for the CURRENT state
        y_pre = self.base.y            # pre-step measurement, for the buffer slide
        u = np.clip(u_base + self.residual_frac * self.half_range * a_res,
                    self.a_low, self.a_high)

        inner_obs, reward, term, trunc, info = self.env.step(u)
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = info["action"]
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_pre
        return self._make_obs(inner_obs), reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
