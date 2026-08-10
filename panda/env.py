"""Gymnasium env: 7-DoF Franka Panda driving its end-effector to a 3-D goal.

Reward is the same DeePC stage-cost form the unicycle env uses:

    r_t = -(y_t - y_ref)^T Q (y_t - y_ref) - u_t^T R u_t + reach_bonus * [reached]

with `y = tip position (x, y, z)` and `y_ref = goal`. Unlike the unicycle there
is no heading component, so `y_ref` is the goal directly -- no bearing reference
to supply. Termination is position-only.

The agent's action `u` is a *delta* joint target, but the plant's true input is
`ctrl` -- an absolute joint-angle target, `clip(qpos_current + u, safe_box)`
(`panda.model.apply_delta`). The two differ whenever the box clip fires, which
happens on a large fraction of steps under random or far-goal excitation
(measured 24-48% depending on policy). A system-identification or DeePC stage
built on this env should collect and identify `ctrl -> y`, not `delta -> y`:
the delta interface is state-dependent (the same `u` produces a different
`ctrl` depending on where `qpos` already is relative to the box), so Hankel
matrices or transfer functions fit on `(delta, y)` pairs would be wrong
precisely where excitation spends most of its time. `info["ctrl"]` (see
`_build_info`) is what a collection script should record as `u`.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from panda.model import (
    CTRL_HZ,
    apply_delta,
    frame_skip,
    load_model,
    safe_box,
    sample_config,
    tip_id,
    tip_position,
)

# Reach rate is flat at 0.90 across delta_max in [0.15, 0.40] -- only episode
# length changes (p90 steps 93 / 70 / 48 / 39 for
# delta_max 0.15 / 0.20 / 0.30 / 0.40), so this bound is not chosen for reach
# performance. Measured during design with a DLS-IK oracle since removed as
# out-of-scope (see docs/superpowers/specs/2026-08-10-panda-reach-env-design.md's
# appendix); not reproducible from this tree. It is chosen because:
#   - q_dot_ss measures 1.63 rad/s here, against a (kp/kd)*DELTA_MAX = 2.0
#     prediction -- the largest bound still meaningfully below torque
#     saturation (at 0.4 the servo delivers only 1.98 rad/s against a
#     predicted 4.0, i.e. the command has stopped being tracked) -- and it
#     stays under the real Panda's 2.175 rad/s joint-velocity limit.
#   - p90 = 70 steps inside the 150-step budget is 2.1x headroom.
# 0.05 needs ~187 steps to cover the same ground and so cannot solve the task
# inside this budget. 0.3 is a defensible alternative (more headroom, cheaper
# episodes, still under the velocity limit) if a later DeePC stage's per-step
# QP cost makes evaluation sweeps painful -- that is the knob to reach for.
DELTA_MAX = 0.2

# Generous finite bound for the qvel block of the observation. Measured |qvel|
# stays under 2 rad/s; finite bounds keep the space well-formed for SB3 later.
QVEL_OBS_LIMIT = 20.0
# Workspace diameter is ~2.4 m, so tip-goal difference cannot exceed this.
REL_GOAL_OBS_LIMIT = 2.5


class PandaReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        delta_max: float = DELTA_MAX,
        goal_tolerance: float = 0.05,
        min_start_goal_dist: float = 0.15,
        max_steps: int = 150,
        dt_ctrl: float = 1.0 / CTRL_HZ,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        reach_bonus: float = 100.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.model, self.data = load_model()
        self.nq = int(self.model.nq)
        self.frame_skip = frame_skip(self.model, dt_ctrl)
        # The true control period, not merely the requested one: frame_skip()
        # rounds dt_ctrl to the nearest multiple of the physics timestep, so a
        # non-default dt_ctrl (e.g. 0.025) would otherwise desync self.dt_ctrl
        # from the rate the sim actually runs at, and from render_fps (below),
        # which derives from it -- three clocks that must agree for
        # "playback is real time" (see scripts/record_panda_video.py) to hold.
        self.dt_ctrl = self.frame_skip * self.model.opt.timestep
        self._tip_id = tip_id(self.model)
        self._lo, self._hi = safe_box(self.model)

        self.delta_max = float(delta_max)
        self.goal_tolerance = float(goal_tolerance)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.max_steps = int(max_steps)
        self.reach_bonus = float(reach_bonus)
        self.render_mode = render_mode
        self._min_dist_relaxed: bool = False

        # Instance copy: render_fps must track this instance's actual
        # dt_ctrl, but `metadata` is a class attribute shared by every
        # instance -- mutating it in place would leak one instance's rate
        # into every other.
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.dt_ctrl))

        # R is ratio-matched to the unicycle's control/state cost balance
        # (4.4e-3 vs its 4.5e-3), not copied from it: |u| here is ~30x smaller,
        # so reusing 1.3e-3 would make control effectively free.
        self.Q = np.eye(3) if Q is None else np.asarray(Q, dtype=np.float64)
        self.R = (
            1.0e-2 * np.eye(self.nq) if R is None else np.asarray(R, dtype=np.float64)
        )

        self.action_space = spaces.Box(
            low=-self.delta_max, high=self.delta_max, shape=(self.nq,), dtype=np.float32
        )
        # obs = [q(7), qvel(7), tip - goal(3), u_prev(7)] = 24
        obs_low = np.concatenate([
            self.model.jnt_range[:, 0],
            np.full(self.nq, -QVEL_OBS_LIMIT),
            np.full(3, -REL_GOAL_OBS_LIMIT),
            np.full(self.nq, -self.delta_max),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            self.model.jnt_range[:, 1],
            np.full(self.nq, QVEL_OBS_LIMIT),
            np.full(3, REL_GOAL_OBS_LIMIT),
            np.full(self.nq, self.delta_max),
        ]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.goal: np.ndarray = np.zeros(3, dtype=np.float64)
        self.last_action: np.ndarray = np.zeros(self.nq, dtype=np.float64)
        self.step_idx: int = 0
        self._renderer = None  # lazy

    # ----- DeePC-compatible accessors ----------------------------------------

    @property
    def state(self) -> np.ndarray:
        """`(14,)` = `concatenate([qpos, qvel])`. Returns a copy."""
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64)

    @property
    def y(self) -> np.ndarray:
        """DeePC output: end-effector position `(3,)`. Returns a copy."""
        return tip_position(self.data, self._tip_id)

    @property
    def y_ref(self) -> np.ndarray:
        """DeePC reference: the goal position `(3,)`. Returns a copy."""
        return self.goal.copy()

    @property
    def tip_site_id(self) -> int:
        """MuJoCo site id of the end-effector.

        Public for direct `site_xpos`/`mj_jacSite` access; currently only
        `tests/test_panda_env.py`'s independent-FK freshness check reads it
        from outside this class. Resolved once in __init__.
        """
        return self._tip_id

    @property
    def safe_box(self) -> tuple[np.ndarray, np.ndarray]:
        """The joint box `apply_delta` actually clips against, `(lo, hi)`.

        Public so `panda/validity.py` and offline data-collection code read
        the authoritative box instead of recomputing
        `panda.model.safe_box(model)` themselves -- same precedent as
        `tip_site_id` above. This property and the
        `panda.model.safe_box` free function it wraps share a name but not a
        namespace: the function is a module-level import used once, here, to
        compute `self._lo`/`self._hi`; everything else should read this
        property instead of importing the function again. Returns copies.
        """
        return self._lo.copy(), self._hi.copy()

    # ----- internals ----------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self.y - self.goal,
            self.last_action,
        ]).astype(np.float32)

    def _build_info(self, reached: bool, ctrl: Optional[np.ndarray] = None) -> dict[str, Any]:
        err = self.y - self.y_ref
        return {
            "state": self.state,
            "goal": self.goal.copy(),
            "y": self.y,
            "y_ref": self.y_ref,
            "pos_error": err,
            "distance": float(np.linalg.norm(err)),
            "action": self.last_action.copy(),
            # What the plant actually received (clip(qpos + action, safe_box)),
            # distinct from "action" above whenever the box clip fires -- see
            # the module docstring. step() passes the value apply_delta just
            # computed; reset() has none to pass (no apply_delta call yet this
            # episode), so it falls back to data.ctrl, which reset() already
            # parked on q0.
            "ctrl": (ctrl if ctrl is not None else self.data.ctrl).copy(),
            "step_idx": self.step_idx,
            "reached": bool(reached),
            "min_dist_relaxed": self._min_dist_relaxed,
            # Contacts are a diagnostic, never a termination condition:
            # terminating on contact would confound "failed to reach" with
            # "grazed itself on the way".
            "ncon": int(self.data.ncon),
        }

    # ----- API ----------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        relaxed = False

        # Goal first: it is sampled by forward kinematics from a random valid
        # configuration, which is what guarantees every goal is reachable. A
        # Cartesian box sample would put an unknown fraction of goals outside the
        # workspace, and every reach-rate number built on that would be junk.
        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64).reshape(3).copy()
        else:
            _, self.goal = sample_config(
                self.model, self.data, rng, self._lo, self._hi, self._tip_id
            )

        if "qpos" in options:
            q0 = np.clip(
                np.asarray(options["qpos"], dtype=np.float64).reshape(self.nq),
                self._lo,
                self._hi,
            )
        else:
            # Sample, then check -- matching UnicycleGoalEnv.reset's fallback
            # pattern. Sampling first means the candidate tested on the last
            # iteration is also the one kept if every draw fails: the fallback
            # is a genuinely-rejected candidate, not an untested extra draw.
            for _ in range(100):
                q0, tip = sample_config(
                    self.model, self.data, rng, self._lo, self._hi, self._tip_id
                )
                if np.linalg.norm(tip - self.goal) >= self.min_start_goal_dist:
                    break
            else:
                # Accept the last draw rather than raising, matching
                # UnicycleGoalEnv.reset's fallback: a too-tight
                # min_start_goal_dist should degrade, not kill a sweep mid-run.
                relaxed = True

        self._min_dist_relaxed = relaxed

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = 0.0
        # Park the servo target on the start pose so nothing yanks if anything
        # integrates before the first apply_delta.
        self.data.ctrl[:] = q0
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.nq, dtype=np.float64)
        self.step_idx = 0

        info = self._build_info(reached=False)
        return self._build_obs(), info

    def step(self, action):
        u = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.nq),
            -self.delta_max,
            self.delta_max,
        )
        ctrl = apply_delta(self.data, u, self._lo, self._hi)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # REQUIRED. mj_step ends after integration, so site_xpos (and ncon) still
        # describe the pre-integration state. Without this, `y` lags one control
        # step and every reward is computed against a stale tip position.
        mujoco.mj_forward(self.model, self.data)

        self.last_action = u
        self.step_idx += 1

        err = self.y - self.y_ref
        distance = float(np.linalg.norm(err))
        reached = distance < self.goal_tolerance
        cost = float(err @ self.Q @ err + u @ self.R @ u)
        reward = -cost + (self.reach_bonus if reached else 0.0)

        terminated = bool(reached)
        truncated = (not terminated) and (self.step_idx >= self.max_steps)

        return (
            self._build_obs(),
            reward,
            terminated,
            truncated,
            self._build_info(reached=reached, ctrl=ctrl),
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from panda.rendering import MujocoRenderer

            self._renderer = MujocoRenderer(self)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
