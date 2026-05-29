"""Gymnasium env: kinematic unicycle navigating to a goal in a 2D workspace."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics import step_unicycle, wrap_to_pi


class UnicycleGoalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        workspace_bounds=((-10.0, 10.0), (-10.0, 10.0)),
        goal_tolerance: float = 0.5,
        min_start_goal_dist: float = 2.0,
        max_steps: int = 200,
        dt: float = 0.025,
        action_bounds=((0.0, 20.0), (-np.pi / 2, np.pi / 2)),
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        reach_bonus: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.workspace_bounds = np.asarray(workspace_bounds, dtype=np.float64)
        self.goal_tolerance = float(goal_tolerance)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.action_bounds = np.asarray(action_bounds, dtype=np.float64)
        # DeePC-style cost on the 3-D output y = (x, y, delta). Heading entry
        # is zero by default (don't-care at the goal) but Q stays 3x3 so heading
        # remains visible to the behavioral predictor.
        self.Q = (
            np.diag([1.0, 1.0, 2.0])
            if Q is None
            else np.asarray(Q, dtype=np.float64)
        )
        self.R = (
            1.3e-3 * np.eye(2, dtype=np.float64)
            if R is None
            else np.asarray(R, dtype=np.float64)
        )
        self.reach_bonus = float(reach_bonus)
        self.render_mode = render_mode

        a_low = self.action_bounds[:, 0].astype(np.float32)
        a_high = self.action_bounds[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        dx = self.workspace_bounds[0, 1] - self.workspace_bounds[0, 0]
        dy = self.workspace_bounds[1, 1] - self.workspace_bounds[1, 0]
        max_dist = float(np.hypot(dx, dy))
        # v_prev/w_prev are zero on reset; widen their bounds to include 0 so
        # the post-reset obs is always inside observation_space even when
        # action_bounds excludes zero (e.g., paper bounds v ∈ [10, 20]).
        obs_low = np.array(
            [0.0, -1.0, -1.0, min(0.0, float(a_low[0])), min(0.0, float(a_low[1]))],
            dtype=np.float32,
        )
        obs_high = np.array(
            [max_dist, 1.0, 1.0, max(0.0, float(a_high[0])), max(0.0, float(a_high[1]))],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # State holders — populated by reset()
        self.state: np.ndarray = np.zeros(3, dtype=np.float64)
        self.goal: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_action: np.ndarray = np.zeros(2, dtype=np.float64)
        self.step_idx: int = 0

        self._renderer = None  # lazy

    # ----- DeePC-compatible accessors ----------------------------------------
    # u = action (the env's action_space), dim 2.
    # y = output measurement, dim 3. For this fully-observed unicycle, y == state.
    # y_ref = constant reference the DeePC controller drives `y - y_ref` to zero.
    # Buffer/Hankel management lives in the controller, not the env.

    @property
    def y(self) -> np.ndarray:
        """Current output measurement: (x, y, delta). Returns a copy."""
        return self.state.copy()

    @property
    def y_ref(self) -> np.ndarray:
        """Reference output: (g_x, g_y, 0). Heading reference is zero (don't-care)."""
        return np.array([self.goal[0], self.goal[1], 0.0], dtype=np.float64)

    def _sample_position(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(*self.workspace_bounds[0])
        y = rng.uniform(*self.workspace_bounds[1])
        return np.array([x, y], dtype=np.float64)

    def _build_obs(self) -> np.ndarray:
        x, y, delta = self.state
        gx, gy = self.goal
        dx, dy = gx - x, gy - y
        distance = float(np.hypot(dx, dy))
        bearing = np.arctan2(dy, dx)
        bearing_rel = float(wrap_to_pi(bearing - delta))
        return np.array(
            [
                distance,
                np.sin(bearing_rel),
                np.cos(bearing_rel),
                self.last_action[0],
                self.last_action[1],
            ],
            dtype=np.float32,
        )

    def _build_info(self, reached: bool) -> dict[str, Any]:
        pos_err = self.state[:2] - self.goal
        return {
            "state": self.state.copy(),
            "goal": self.goal.copy(),
            "y": self.y,
            "y_ref": self.y_ref,
            "pos_error": pos_err.copy(),
            "distance": float(np.linalg.norm(pos_err)),
            "action": self.last_action.copy(),
            "step_idx": self.step_idx,
            "reached": bool(reached),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        if "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float64).reshape(3).copy()
            # Wrap caller-supplied heading so subsequent step()s don't see a jump.
            self.state[2] = float(wrap_to_pi(self.state[2]))
        else:
            x, y = self._sample_position(rng)
            delta = rng.uniform(-np.pi, np.pi)
            self.state = np.array([x, y, delta], dtype=np.float64)

        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64).reshape(2).copy()
        else:
            for _ in range(100):
                candidate = self._sample_position(rng)
                if np.linalg.norm(self.state[:2] - candidate) >= self.min_start_goal_dist:
                    self.goal = candidate
                    break
            else:
                self.goal = candidate  # fall back to last sample

        self.last_action = np.zeros(2, dtype=np.float64)
        self.step_idx = 0

        obs = self._build_obs()
        info = self._build_info(reached=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        clipped = np.clip(action, self.action_bounds[:, 0], self.action_bounds[:, 1])

        self.state = step_unicycle(self.state, clipped, self.dt)
        # Wall-clip position to workspace; heading wraps to [-pi, pi].
        self.state[0] = np.clip(
            self.state[0], self.workspace_bounds[0, 0], self.workspace_bounds[0, 1]
        )
        self.state[1] = np.clip(
            self.state[1], self.workspace_bounds[1, 0], self.workspace_bounds[1, 1]
        )
        self.state[2] = float(wrap_to_pi(self.state[2]))
        self.last_action = clipped
        self.step_idx += 1

        # Termination criterion is position-only (matches goal_tolerance contract).
        pos_err = self.state[:2] - self.goal
        distance = float(np.linalg.norm(pos_err))
        reached = distance < self.goal_tolerance

        # Stage cost in DeePC form: ||y - y_ref||_Q^2 + ||u||_R^2.
        err_y = self.y - self.y_ref
        cost = float(err_y @ self.Q @ err_y + clipped @ self.R @ clipped)
        reward = -cost + (self.reach_bonus if reached else 0.0)

        terminated = bool(reached)
        truncated = (not terminated) and (self.step_idx >= self.max_steps)

        obs = self._build_obs()
        info = self._build_info(reached=reached)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .rendering import PygameRenderer

            self._renderer = PygameRenderer(self, self.render_mode)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
