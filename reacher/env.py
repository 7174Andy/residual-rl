"""`ReacherGoal-v0` — goal-reaching on Gymnasium's 2-link planar arm.

The RL-facing wrapper over `reacher/model.py`. Classical controllers keep driving
MuJoCo directly through that module; this env is additive and changes no existing
result.

Three deliberate departures from stock `Reacher-v5`, each measured or argued in
`docs/superpowers/specs/2026-08-16-reacher-residual-rl-design.md`:

* **Goals are rejection-sampled against the reachable annulus.** At
  `SAFE_MARGIN = 0.02` the fingertip attains `[0.0291, 0.21]` m while goals are
  drawn from a 0.20 m disc, so 2.1% of stock draws are physically impossible.
  Journey 12 records the 14.7% version of this bug reading as controller failure.
* **The episode never terminates on reach.** Terminating makes "arrive and hold"
  score the same as "arrive and leave" -- which is precisely the drift
  (`best -> final` by 2.1-2.3x on every controller) this env exists to let a
  policy fix. Episodes always run the full `max_steps`.
* **Dense per-step distance, not squared distance.** At a 10 mm tolerance on a
  210 mm workspace the squared term is ~1e-4 and its gradient vanishes as the tip
  approaches -- the exact regime being targeted.

`y = [q; fingertip]` and `y_ref = [0, 0, g]` are identical to
`reacher/deepc_setup.py`, so a DeePC/Select-DPC buffer and this env's measurement
are the same object.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reacher.deepc_setup import y_ref_for
from reacher.model import (
    NQ_ARM, fingertip, frame_skip, is_reachable, load_model, safe_box,
    sample_config, sample_goal, set_state, step_torque,
)

# Observation bound for joint velocity. Not a physical limit -- torque 200 x gear
# can exceed it transiently -- so `build_obs` CLIPS into the box. An observation
# outside its declared space is a silent contract break that SB3's normalization
# and gymnasium's checker both assume cannot happen.
QVEL_OBS_LIMIT = 50.0
REL_GOAL_OBS_LIMIT = 0.5     # fingertip reach 0.21 + target disc 0.2, rounded up


class ReacherGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        goal_tolerance: float = 0.01,
        max_steps: int = 50,
        reach_bonus: float = 1.0,
        ctrl_cost: float = 1.0e-3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.model, self.data = load_model()
        self.frame_skip = frame_skip(self.model)
        self._lo, self._hi = safe_box(self.model)

        self.goal_tolerance = float(goal_tolerance)
        self.max_steps = int(max_steps)
        # Paid on EVERY step the tip holds inside tolerance, because the episode
        # does not terminate -- so this term IS the station-keeping reward. Kept
        # far below the unicycle's 100: at typical distances of 0.01-0.2 a bonus
        # of 100 would swamp the distance signal and starve the approach phase.
        self.reach_bonus = float(reach_bonus)
        self.ctrl_cost = float(ctrl_cost)
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(NQ_ARM,), dtype=np.float32)
        # obs = [cos q(2), sin q(2), qvel(2), tip - goal(2)] = 8.
        # Angles enter as cos/sin because joint0 is UNLIMITED and wraps: a raw
        # angle is discontinuous at +-pi and the policy would see a cliff there.
        self._obs_low = np.concatenate([
            np.full(2 * NQ_ARM, -1.0),
            np.full(NQ_ARM, -QVEL_OBS_LIMIT),
            np.full(2, -REL_GOAL_OBS_LIMIT),
        ]).astype(np.float32)
        self._obs_high = -self._obs_low          # every bound is symmetric
        self.observation_space = spaces.Box(
            low=self._obs_low, high=self._obs_high, dtype=np.float32)

        self.goal: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_action: np.ndarray = np.zeros(NQ_ARM, dtype=np.float64)
        self.step_idx: int = 0
        self.path: float = 0.0
        self._prev_tip: np.ndarray = np.zeros(2, dtype=np.float64)

    # ----- DeePC-compatible accessors ---------------------------------------

    @property
    def state(self) -> np.ndarray:
        """`(4,)` = `[q(2); qvel(2)]`, arm joints only. Returns a copy."""
        return np.concatenate([
            self.data.qpos[:NQ_ARM], self.data.qvel[:NQ_ARM]
        ]).astype(np.float64)

    @property
    def tip(self) -> np.ndarray:
        """Fingertip position `(2,)`."""
        return fingertip(self.data)

    @property
    def y(self) -> np.ndarray:
        """`(4,)` = `[q; fingertip]` -- identical to `deepc_setup.outputs`."""
        return np.concatenate([self.data.qpos[:NQ_ARM], self.tip]).astype(np.float64)

    @property
    def y_ref(self) -> np.ndarray:
        """`(4,)` = `[0, 0, g]`. The joint block is unweighted, so its value is free."""
        return y_ref_for(self.goal)

    # ----- internals ---------------------------------------------------------

    def build_obs(self) -> np.ndarray:
        """The policy-facing observation for the CURRENT state.

        Public because consumers outside the class need it: an evaluation script
        driving a trained policy through a shared episode loop has to compute the
        observation itself, and reaching into a private method to do that is the
        defect, not the caller.
        """
        q = np.asarray(self.data.qpos[:NQ_ARM], dtype=np.float64)
        raw = np.concatenate([
            np.cos(q), np.sin(q), self.data.qvel[:NQ_ARM], self.tip - self.goal
        ]).astype(np.float32)
        # Clip rather than widen the box: qvel is transiently unbounded under
        # torque, and an obs outside its space breaks the checker's contract.
        return np.clip(raw, self._obs_low, self._obs_high)

    def _build_info(self, reached: bool, dist: float) -> dict[str, Any]:
        return {
            "y": self.y,
            "y_ref": self.y_ref,
            "goal": self.goal.copy(),
            "state": self.state,
            "dist": float(dist),
            "reached": bool(reached),
            "path": float(self.path),
            "step_idx": int(self.step_idx),
            "action": self.last_action.copy(),
        }

    def _sample_reachable_goal(self, rng) -> np.ndarray:
        """Draw as Gym draws, then reject what the arm cannot physically attain."""
        for _ in range(200):
            g = sample_goal(rng)
            if is_reachable(self.model, g):
                return g
        raise RuntimeError(
            "no reachable goal in 200 draws -- check SAFE_MARGIN against "
            "reachable_annulus(); this should be a ~2% rejection rate"
        )

    # ----- API ---------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64).reshape(2).copy()
        else:
            self.goal = self._sample_reachable_goal(rng)

        if "qpos" in options:
            q0 = np.clip(
                np.asarray(options["qpos"], dtype=np.float64).reshape(NQ_ARM),
                self._lo, self._hi,
            )
        else:
            q0, _ = sample_config(self.model, self.data, rng)

        # set_state writes qpos[0:2] AND qpos[2:4] (the target's own slide
        # joints) and zeroes qvel. Writing qpos wholesale here instead would
        # silently move the goal -- the trap reacher/model.py documents.
        set_state(self.model, self.data, q0, goal=self.goal)

        self.last_action = np.zeros(NQ_ARM, dtype=np.float64)
        self.step_idx = 0
        self.path = 0.0
        self._prev_tip = self.tip
        dist = float(np.linalg.norm(self.tip - self.goal))
        return self.build_obs(), self._build_info(reached=dist < self.goal_tolerance,
                                                 dist=dist)

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64).reshape(NQ_ARM), -1.0, 1.0)
        applied = step_torque(self.model, self.data, u, self.frame_skip)

        self.last_action = applied
        self.step_idx += 1
        tip = self.tip
        self.path += float(np.linalg.norm(tip - self._prev_tip))
        self._prev_tip = tip

        dist = float(np.linalg.norm(tip - self.goal))
        reached = dist < self.goal_tolerance
        reward = -dist - self.ctrl_cost * float(applied @ applied)
        if reached:
            reward += self.reach_bonus

        # Never terminate on reach -- see the module docstring.
        terminated = False
        truncated = self.step_idx >= self.max_steps
        return (self.build_obs(), reward, terminated, truncated,
                self._build_info(reached=reached, dist=dist))

    def render(self):
        return None

    def close(self):
        return None
