"""Hand-built trot gait for the Unitree Go1: leg geometry, IK and FK.

Frame
-----
Every foot position here is expressed in the trunk frame (x forward, y left,
z up), *relative to that leg's hip-abduction joint*. The target is the foot
**site** (sphere centre, radius 0.023 m), so ground contact is 0.023 m lower.

Conventions (checked against the MuJoCo model by
`tests/test_quadruped_model.py::test_ik_matches_mujoco_fk`): the abduction joint
rotates about +x by q_ab; thigh and knee (calf) rotate about +y; a foot hanging
straight below the hip has negative z.

Imports `numpy` only -- no `mujoco`, no `gymnasium` -- so the kinematics are
usable and testable without the simulator.

Measured limit (2026-10-29, menagerie feadf76)
----------------------------------------------
Gait: `TrotGait` defaults -- period 0.4 s, lift 0.07 m, height 0.27 m,
k_foot 0 (open loop; no stabilisation was needed). Re-measure with
`quadruped/checks.py` (`run_held`, `run_random`, `FLOOR`, `GRID`).

* Held-command range (each command held 10 s from standing; no fall, tracking
  within 50%; `scripts/check_go1_gait.py`): every grid point passes --
  vx in [-0.5, 1.0] m/s, vy in [-0.4, 0.4] m/s, yaw_rate in [-1.5, 1.5] rad/s,
  plus (0.5, 0.2, 0), (0.5, 0, 0.8), (0.3, -0.2, -0.5). Changing commands are
  only measured safe on the floor below (0/10 falls).
* Changing commands (`run_random`, 20 s, 10 seeds, new command every 0.5 s):
  the floor (vx [0, 0.6], vy +-0.2, yaw_rate +-0.8) falls 0/10
  (`scripts/check_go1_gait.py --random-seeds 10`). Widening any one axis:
  vx [0, 0.75], vx [-0.25, 0.6], yaw_rate +-1.5 fall 0/10; vy +-0.4 falls
  1/10; vx [0, 1.0] or [-0.5, 0.6] falls 7/10; the full held range falls
  10/10. Recipe: `run_random(np.random.default_rng(s), ranges={...})` for
  s = 0..9, `ranges` = FLOOR with the one axis replaced. Draw changing
  commands from the floor, not from the held range.
* Faster changes on the floor: a new command every step (`hold_s=0.02`) or
  every 0.1 s (`hold_s=0.1`) falls 0/10 --
  `run_random(np.random.default_rng(s), hold_s=...)`, s = 0..9.
* vx offset: vx is about -0.09 m/s lower than commanded on EVERY command
  (zero -> -0.086, pure vy 0.2 -> -0.090, pure yaw 0.8 -> -0.089; backward
  overshoots, -0.5 -> -0.590, -0.25 -> -0.332; 0.5 -> 0.409), so no command
  holds the robot still (~0.9 m drift per 10 s at zero command). The
  command -> velocity map is affine, not linear; the command-layer controller
  (DeePC) must account for it. Measured with `run_held(cmd)["v_mean"][0]`
  (mean of `Go1Sim.body_vel()` over the last 5 s of a 10 s hold).
* Yaw rate oscillates within each 0.4 s gait cycle (std 0.29 rad/s at zero
  command, 0.39 at (0.5, 0, 0); `body_vel()[2]` sampled over 5 s), so
  single-sample readings are phase-locked; only cycle-averaged means are
  meaningful.
"""
from __future__ import annotations

import numpy as np

# Menagerie actuator order; every (4, ...) array below uses it.
LEGS = ("FR", "FL", "RR", "RL")

# Hip-abduction joint position in the trunk frame, per leg.
HIP_XY = np.array([[0.1881, -0.04675], [0.1881, 0.04675],
                   [-0.1881, -0.04675], [-0.1881, 0.04675]])
SIDE = np.array([-1, 1, -1, 1])  # +1 = left
# Trot: pair A (FL+RR) at phase 0, pair B (FR+RL) half a cycle later.
PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5])

ABD, THIGH, CALF = 0.08, 0.213, 0.213  # abduction offset, thigh and calf lengths

# Joint ctrl ranges (abduction, thigh, calf).
Q_LO = np.array([-0.863, -0.686, -2.818])
Q_HI = np.array([0.863, 4.501, -0.888])


def leg_ik(p: np.ndarray, side: int) -> np.ndarray:
    """Foot site `p` (3,) relative to the hip -> (q_ab, q_thigh, q_knee), unclipped.

    Unreachable targets are projected onto the workspace boundary, so the
    result is always finite.
    """
    x, y, z = p
    a, b, l1 = THIGH, CALF, side * ABD
    Lp = np.sqrt(max(y * y + z * z - l1 * l1, 1e-12))
    q_ab = np.arctan2(z, y) - np.arctan2(-Lp, l1)
    r = min(np.hypot(x, Lp), a + b - 1e-6)
    q_knee = -(np.pi - np.arccos(np.clip((a * a + b * b - r * r) / (2 * a * b), -1, 1)))
    q_thigh = np.arctan2(-x, Lp) - np.arctan2(b * np.sin(q_knee), a + b * np.cos(q_knee))
    return np.array([q_ab, q_thigh, q_knee])


def leg_fk(q: np.ndarray, side: int) -> np.ndarray:
    """Inverse of `leg_ik`: joint angles (q_ab, q_thigh, q_knee) -> foot site."""
    q_ab, q_t, q_k = q
    xp = -THIGH * np.sin(q_t) - CALF * np.sin(q_t + q_k)
    zp = -THIGH * np.cos(q_t) - CALF * np.cos(q_t + q_k)
    l1 = side * ABD
    c, s = np.cos(q_ab), np.sin(q_ab)
    return np.array([xp, l1 * c - zp * s, l1 * s + zp * c])


class TrotGait:
    """Phase-clock trot: stance feet slide back at the commanded velocity,
    swing feet follow a cosine-eased arc to a landing point ahead of nominal.

    `feet` and `nominal` are relative to each hip-abduction joint (the frame
    `leg_ik` takes). `height` is hip-to-foot-site depth; `k_foot` is the
    foot-placement gain on velocity error (0 = off).
    """

    def __init__(self, period=0.4, lift=0.07, height=0.27, dt=0.02, k_foot=0.0):
        self.period, self.lift, self.height, self.dt, self.k_foot = period, lift, height, dt, k_foot
        self.nominal = np.column_stack([np.zeros(4), SIDE * ABD, np.full(4, -height)])
        self.r = HIP_XY + np.column_stack([np.zeros(4), SIDE * ABD])  # nominal foot, trunk frame
        self.reset()

    def reset(self) -> None:
        self.phase = 0.0
        self.feet = self.nominal.copy()
        self.stance = PHASE_OFFSET % 1 < 0.5
        self.liftoff = self.nominal[:, :2].copy()

    def step(self, cmd, v_meas=None) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=float)
        self.phase = (self.phase + self.dt / self.period) % 1
        s = np.round((self.phase + PHASE_OFFSET) % 1, 9) % 1  # round: fp drift must not split a diagonal pair at the 0.5 boundary
        was, self.stance = self.stance, s < 0.5
        u = np.column_stack([cmd[0] - cmd[2] * self.r[:, 1], cmd[1] + cmd[2] * self.r[:, 0]])
        land = self.nominal[:, :2] + u * self.period / 4
        if self.k_foot and v_meas is not None:
            land = land + self.k_foot * (np.asarray(v_meas) - cmd[:2])
        self.liftoff[was & ~self.stance] = self.feet[was & ~self.stance, :2]
        p = (s - 0.5) / 0.5
        ease = (1 - np.cos(np.pi * p)) / 2
        sw = ~self.stance
        self.feet[self.stance, :2] -= u[self.stance] * self.dt
        self.feet[self.stance, 2] = -self.height
        self.feet[sw, :2] = self.liftoff[sw] + (land[sw] - self.liftoff[sw]) * ease[sw, None]
        self.feet[sw, 2] = -self.height + self.lift * np.sin(np.pi * p[sw])
        q = np.array([leg_ik(self.feet[i], SIDE[i]) for i in range(4)])
        return np.clip(q, Q_LO, Q_HI)
