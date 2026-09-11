"""The clone's feature window as an RL observation — the paper's Eq. 17 arm.

`ResidualSelectEnv` computes this same 43-D vector every step to get `u_base`,
then throws it away and hands the policy a body-frame observation instead. Here
the vector IS the observation, and the action is the torque directly. That is
what makes clone weights transferable into an SB3 actor: the paper requires the
actor's input to match the surrogate's ("the architecture of the actor network
should match the NNMPC settings", arXiv:2510.03354 SIII.C.1), and the two
architectures there share one input `s_k`.

Observations are RAW, not min-max normalized as in `residual_env.py`. Two
reasons. Folding a normalization into the transferred first layer would couple
`rl/sb3.py` to this env's bounds, and the window needs no normalizing anyway:
measured over all 25,000 rows of `data/dagger_r3.npz` every block sits in
O(0.2-3.3), against the 8-D policy observation whose `qvel` block is bounded at
+-50.

The window is a sufficient statistic, so this is not a POMDP: `q_t` is in
`y_cur` and `q_{t-1}` is in `y_ini[-1]`, so velocity is recoverable at
`dt = 0.02 s`. The representation is redundant, not lossy.
"""
from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_features import feature_dim, featurize
from reacher.env import REL_GOAL_OBS_LIMIT, ReacherGoalEnv
from reacher.model import NQ_ARM

# joint1's model limit is +-3.0, but MuJoCo's joint limits are soft and the
# existing dataset reaches 3.214. Bound the box from the layout plus margin, not
# from one dataset's extremes.
_Q1_LIMIT = 3.3
# The fingertip's reach is 0.21 m (reacher/model.py's module docstring).
_TIP_LIMIT = 0.22


class CloneWindowEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, T_ini: int = 5, render_mode: Optional[str] = None):
        super().__init__()
        self.T_ini = int(T_ini)
        self.render_mode = render_mode

        self.env = gym.make("ReacherGoal-v0", render_mode=render_mode)
        self.base = cast(ReacherGoalEnv, self.env.unwrapped)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(NQ_ARM,),
                                       dtype=np.float32)
        low, high = self._obs_bounds()
        assert low.shape == (feature_dim(self.T_ini, 0),), (
            f"bounds are {low.shape[0]}-D but featurize emits "
            f"{feature_dim(self.T_ini, 0)}-D")
        self._obs_low, self._obs_high = low, high
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

        self._u_buf: Optional[np.ndarray] = None
        self._y_buf: Optional[np.ndarray] = None

    def _obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-block bounds, read off `clone_features`' layout."""
        # One expanded y row: (cos q0, sin q0, q1, tip_x, tip_y).
        y_row = np.array([1.0, 1.0, _Q1_LIMIT, _TIP_LIMIT, _TIP_LIMIT])
        high = np.concatenate([
            np.ones(self.T_ini * NQ_ARM),          # u_ini, post-clip
            np.tile(y_row, self.T_ini),            # y_ini
            y_row,                                 # y_cur
            np.full(2, REL_GOAL_OBS_LIMIT),        # tip - goal
            np.array([1.0]),                       # buffer validity
        ]).astype(np.float32)
        low = -high.copy()
        low[-1] = 0.0                              # validity is [0, 1]
        return low, high

    @property
    def max_steps(self) -> int:
        """The inner env's horizon, so a shared episode loop can drive this env
        like any other (journey 13's `I6` finding)."""
        return self.base.max_steps

    @property
    def goal(self) -> np.ndarray:
        """The inner env's goal, for the same reason."""
        return self.base.goal

    def _make_obs(self) -> np.ndarray:
        assert self._u_buf is not None and self._y_buf is not None
        raw = featurize(self._u_buf, self._y_buf, self.base.y, self.base.goal,
                        self.base.step_idx, None)
        # Clip rather than widen: the space contract must hold under transient
        # limit overshoot, matching `ReacherGoalEnv.build_obs`.
        return np.clip(raw, self._obs_low, self._obs_high).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        _inner, info = self.env.reset(seed=seed, options=options)
        self._u_buf = np.zeros((self.T_ini, NQ_ARM))
        self._y_buf = np.tile(self.base.y, (self.T_ini, 1))
        return self._make_obs(), info

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64).reshape(NQ_ARM),
                    -1.0, 1.0)
        y_pre = self.base.y                # pre-step measurement, for the slide
        _inner, reward, term, trunc, info = self.env.step(u)
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = info["action"]   # the APPLIED torque, post-clip
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_pre
        return self._make_obs(), reward, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
