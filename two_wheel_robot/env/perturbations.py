# two_wheel_robot/env/perturbations.py
"""Plant-side perturbation wrappers for robustness benchmarking.

Every wrapper here perturbs the *plant*, never the observation pipeline, so all
controllers are affected identically no matter how they read state: DeePC and the
clone pull `env.unwrapped.y`, the RL arms consume the Gym observation, and both
routes see the same disturbed dynamics.

Deliberately NOT provided: measurement/observation noise. The arms read state
through different channels (`unwrapped.y` vs the normalized body obs), so noising
one channel would perturb some arms and not others -- an unfair comparison
disguised as a robustness result. Injecting it fairly needs the noise applied at
the `y` property itself, which is an env change, not a wrapper.

Noise is drawn from a generator seeded off the episode seed, so for a given
`reset(seed=s)` every arm sees the *same* disturbance realization at the same step
index -- the pairing the per-seed comparison depends on.
"""
from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
import numpy as np

from two_wheel_robot.env.dynamics import wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv


class _SeededPerturbation(gym.Wrapper):
    """Base: re-seeds `self.rng` from the episode seed so realizations are paired."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Same seed -> same noise stream, for every arm.
        self.rng = np.random.default_rng(0 if seed is None else int(seed))
        return self.env.reset(seed=seed, options=options)


class StateDisturbance(_SeededPerturbation):
    """Unmeasured process noise: kick the pose before each dynamics step.

    Applied *before* `env.step`, so the resulting observation and reward both
    reflect the disturbance consistently, and the controller could not have seen
    the kick when it chose this step's action (which is what "unmeasured" means).
    """

    def __init__(self, env: gym.Env, pos_sigma: float = 0.0, heading_sigma: float = 0.0):
        super().__init__(env)
        self.pos_sigma = float(pos_sigma)
        self.heading_sigma = float(heading_sigma)

    def step(self, action):
        base = cast(UnicycleGoalEnv, self.env.unwrapped)
        if self.pos_sigma > 0.0:
            base.state[:2] += self.rng.normal(0.0, self.pos_sigma, size=2)
            wb = base.workspace_bounds
            base.state[0] = np.clip(base.state[0], wb[0, 0], wb[0, 1])
            base.state[1] = np.clip(base.state[1], wb[1, 0], wb[1, 1])
        if self.heading_sigma > 0.0:
            base.state[2] = wrap_to_pi(
                base.state[2] + self.rng.normal(0.0, self.heading_sigma)
            )
        return self.env.step(action)


class ActuatorGain(gym.Wrapper):
    """Calibration mismatch: the plant executes `gain * u`, not `u`.

    Models the real-hardware case CLAUDE.md warns about -- a PCA9685 running a few
    percent fast -- where the controller's model of its own actuator is wrong.
    `gain` is per-channel `(v_gain, w_gain)` or a scalar. Deterministic (no rng),
    so it needs no seeding.
    """

    def __init__(self, env: gym.Env, gain=1.0):
        super().__init__(env)
        self.gain = np.broadcast_to(np.asarray(gain, dtype=np.float64), (2,)).copy()

    def step(self, action):
        u = np.asarray(action, dtype=np.float64).reshape(2) * self.gain
        return self.env.step(u)


class ActionLatency(_SeededPerturbation):
    """Control delay: the plant executes the action issued `k` steps ago.

    The first `k` steps execute the action-space midpoint (the same neutral command
    DeePC's `u_init_midpoint` primes its buffer with), so no arm gets a free
    zero-velocity head start that the others don't.
    """

    def __init__(self, env: gym.Env, k: int = 0):
        super().__init__(env)
        self.k = int(k)
        bounds = cast(UnicycleGoalEnv, self.env.unwrapped).action_bounds
        self._neutral = 0.5 * (bounds[:, 0] + bounds[:, 1])
        self._queue: list = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._queue = [self._neutral.copy() for _ in range(self.k)]
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.k <= 0:
            return self.env.step(action)
        self._queue.append(np.asarray(action, dtype=np.float64).reshape(2))
        applied = self._queue.pop(0)
        return self.env.step(applied)
