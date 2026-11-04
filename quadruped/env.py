"""Gymnasium env: Unitree Go1 (MuJoCo) walking to a goal point in a 2D workspace.

Mirrors `TwoWheelGoal-v0`. The plant is the Go1 with the open-loop trot gait
from `quadruped/gait.py` inside it (see there for the measured limits); the
agent commands a body-frame velocity u = (vx, vy, yaw_rate) at 50 Hz (dt=0.02).

Action bounds vx [-0.25, 0.6], vy [-0.2, 0.2], yaw_rate [-0.8, 0.8]: the gait's
stable range for commands changed every 0.5 s (0/10 falls; vx [-0.5, 0.6] falls
7/10). Faster changes with backward vx are NOT safe: run_random over these bounds
falls 3/10 at 0.02 s holds and 4/10 at 0.1 s, but 0/10 at 0.02 s with vx (0, 0.6)
(run_random, seeds s = 0..9, 20 s, default noise, ranges = env bounds, or vx (0, 0.6));
per-step uniform random falls 6/20 (see Solvability). Excitation/exploration
should hold commands several steps, smooth them, or narrow the backward range.

Scale: workspace [-1.5, 1.5]^2 (position is NOT wall-clipped; the robot may
leave the box), goal_tolerance 0.15 m, max_steps 400.

Falls: if the trunk tips over (`Go1Sim.fell()`) and the goal was not reached
this step, the episode terminates with `info["fell"]=True` and reward
= stage_reward * (max_steps - step_idx + 1), i.e. charged as if the robot stood
still until timeout (stage_reward excludes the reach bonus). Reached wins.
In deepc mode a fall step can reach about -7000 (~70x the reach bonus), and the
"same as standing still" equivalence is undiscounted (under gamma<1 a fall is
over-charged), so RL reward scaling/normalisation is a deliberate decision for
the next spec.

info["ctrl"] has shape (4, 3): LEGS order (FR, FL, RR, RL) x (hip, thigh, calf);
`.ravel()` gives the 12 actuator targets in menagerie actuator order.

`reset(options={"state": ...})` is the PRE-settle pose: the robot is placed
there, then settles for SETTLE_S at zero command. Measured drift: 6.3 cm and -0.040 rad yaw,
deterministic (reset with options["state"], compare env.y to the state; 3 poses).

Solvability (2026-10-29, menagerie feadf76), `scripts/check_quadruped_env.py --seeds 20`:
    random: reached 0/20, mean steps-to-reach n/a, falls 6, truncations 14
    pcontrol: reached 20/20, mean steps-to-reach 255.6 (min 148, max 370), falls 0, truncations 0

Import rule: this is the only file in `quadruped/` that imports gymnasium.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from quadruped.checks import SETTLE_S
from quadruped.gait import TrotGait
from quadruped.model import Go1Sim


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class QuadrupedGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        workspace_bounds=((-1.5, 1.5), (-1.5, 1.5)),
        goal_tolerance: float = 0.15,
        min_start_goal_dist: float = 0.5,
        max_steps: int = 400,
        action_bounds=((-0.25, 0.6), (-0.2, 0.2), (-0.8, 0.8)),
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        reach_bonus: float = 100.0,
        render_mode: Optional[str] = None,
        reward: str = "deepc",
    ):
        super().__init__()
        self.workspace_bounds = np.asarray(workspace_bounds, dtype=np.float64)
        self.goal_tolerance = float(goal_tolerance)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.max_steps = int(max_steps)
        self.dt = 0.02
        self.action_bounds = np.asarray(action_bounds, dtype=np.float64)
        self.Q = np.diag([1.0, 1.0, 2.0]) if Q is None else np.asarray(Q, dtype=np.float64)
        self.R = (
            np.diag([3.3e-2, 3.3e-2, 1.1e-4]) if R is None else np.asarray(R, dtype=np.float64)
        )
        self.reach_bonus = float(reach_bonus)
        if reward not in ("deepc", "linear"):
            raise ValueError(f"reward must be 'deepc' or 'linear', got {reward!r}")
        self.reward_mode = reward
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        self.render_mode = render_mode
        self._renderer = None
        self._cam = None

        a_low = self.action_bounds[:, 0].astype(np.float32)
        a_high = self.action_bounds[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        # Obs distance bound is a margin, not a clip: workspace diagonal + max|vx|*max_steps*dt.
        # Covers start/goal inside the box for at most max_steps (measured worst case 8.46
        # vs bound 9.04); void if options place start/goal outside the box or the caller
        # keeps stepping past truncation.
        diag = float(np.hypot(*(self.workspace_bounds[:, 1] - self.workspace_bounds[:, 0])))
        travel = max(abs(self.action_bounds[0, 0]), self.action_bounds[0, 1]) * self.max_steps * self.dt
        obs_low = np.concatenate([[0.0, -1.0, -1.0], np.minimum(0.0, a_low)]).astype(np.float32)
        obs_high = np.concatenate(
            [[diag + travel, 1.0, 1.0], np.maximum(0.0, a_high)]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.sim = Go1Sim()
        self.gait = TrotGait(dt=self.dt)
        self.goal = np.zeros(2, dtype=np.float64)
        self.start_request = np.zeros(3, dtype=np.float64)  # pre-settle (x, y, delta) of the last reset
        self.last_action = np.zeros(3, dtype=np.float64)
        self.step_idx = 0
        self._ctrl = np.zeros((4, 3))

    @property
    def y(self) -> np.ndarray:
        """Output (x, y, yaw) of the trunk, world frame. Returns a copy."""
        return np.array([*self.sim.data.qpos[0:2], _wrap(self.sim.rpy()[2])], dtype=np.float64)

    @property
    def state(self) -> np.ndarray:
        return self.y

    @property
    def y_ref(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], 0.0], dtype=np.float64)

    def _sample_position(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [rng.uniform(*self.workspace_bounds[0]), rng.uniform(*self.workspace_bounds[1])]
        )

    def _build_obs(self) -> np.ndarray:
        x, y, yaw = self.y
        dx, dy = self.goal[0] - x, self.goal[1] - y
        b = _wrap(np.arctan2(dy, dx) - yaw)
        return np.array(
            [np.hypot(dx, dy), np.sin(b), np.cos(b), *self.last_action], dtype=np.float32
        )

    def _build_info(self, reached: bool, fell: bool = False) -> dict[str, Any]:
        pos_err = self.y[:2] - self.goal
        return {
            "state": self.y,
            "goal": self.goal.copy(),
            "y": self.y,
            "y_ref": self.y_ref,
            "pos_error": pos_err.copy(),
            "distance": float(np.linalg.norm(pos_err)),
            "action": self.last_action.copy(),
            "step_idx": self.step_idx,
            "reached": bool(reached),
            "fell": bool(fell),
            "ctrl": self._ctrl.copy(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        if "state" in options:
            x, y, delta = np.asarray(options["state"], dtype=np.float64).reshape(3)
        else:
            x, y = self._sample_position(rng)
            delta = rng.uniform(-np.pi, np.pi)
        start = np.array([x, y])
        self.start_request = np.array([x, y, delta], dtype=np.float64)

        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float64).reshape(2).copy()
        else:
            for _ in range(100):
                candidate = self._sample_position(rng)
                if np.linalg.norm(start - candidate) >= self.min_start_goal_dist:
                    break
            self.goal = candidate

        sim = self.sim
        sim.reset()
        sim.data.qpos[0:2] = start
        sim.data.qpos[3:7] = [np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)]
        mujoco.mj_forward(sim.model, sim.data)
        self.gait.reset()
        for _ in range(round(SETTLE_S / self.dt)):
            sim.step(self.gait.step(np.zeros(3), v_meas=sim.body_vel()[:2]))

        self.step_idx = 0
        self.last_action = np.zeros(3)
        self._ctrl = sim.data.ctrl[sim.act_idx].copy()  # real applied (settle) targets
        return self._build_obs(), self._build_info(reached=False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(3)
        u = np.clip(action, self.action_bounds[:, 0], self.action_bounds[:, 1])

        self._ctrl = np.asarray(self.gait.step(u, v_meas=self.sim.body_vel()[:2]))
        self.sim.step(self._ctrl)
        self.last_action = u
        self.step_idx += 1

        distance = float(np.linalg.norm(self.y[:2] - self.goal))
        reached = distance < self.goal_tolerance
        if self.reward_mode == "linear":
            cost = distance + float(u @ self.R @ u)
        else:
            err = self.y - self.y_ref
            cost = float(err @ self.Q @ err + u @ self.R @ u)
        stage_reward = -cost
        fell = bool(self.sim.fell())
        if reached:
            reward = stage_reward + self.reach_bonus
        elif fell:
            reward = stage_reward * (self.max_steps - self.step_idx + 1)
        else:
            reward = stage_reward

        terminated = bool(reached or fell)
        truncated = (not terminated) and (self.step_idx >= self.max_steps)
        return self._build_obs(), reward, terminated, truncated, self._build_info(reached, fell)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 2.0, 135.0, -20.0
        data = self.sim.data
        self._cam.lookat[:] = data.xpos[self.sim.trunk_id]
        self._renderer.update_scene(data, camera=self._cam)
        scn = self._renderer.scene
        if scn.ngeom < scn.maxgeom:  # visual-only goal marker, no physics effect
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.goal_tolerance, 0, 0],
                pos=[self.goal[0], self.goal[1], 0.01],
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.1, 0.1, 0.5],
            )
            scn.ngeom += 1
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
