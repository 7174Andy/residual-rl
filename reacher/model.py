"""MuJoCo layer for Gymnasium's `Reacher-v5`: load, joint box, FK, torque step.

A 2-link planar arm, used as the tractable control for the Panda work. It removes
four things that made `PandaReach-v0` hard, all at once:

* **No redundancy.** 2 joints drive a 2-D fingertip, so `q -> tip` is generically
  locally invertible. There is no self-motion manifold, so nothing is hidden from
  the output the way the Panda's 4-D null space hides state from the tip.
* **No gravity term.** The arm is planar with gravity perpendicular to the plane,
  so `g(q) = 0`. Measured: holding `u = 0` for 50 steps moves the joints by
  `0.0000` rad. `u = 0` genuinely means "hold", unlike the Panda under torque,
  where the tip falls 185 mm in 0.2 s.
* **Torque directly.** `gaintype=FIXED, biastype=NONE, gear=200`, so `ctrl` is
  torque with no PD servo in between -- the interface the Panda only reaches by
  rewriting its actuators.
* **A 2-D configuration space** (and see `q0` below -- effectively 1-D for the
  dynamics), against the Panda's ~5.7 effective dimensions. Covering it at the
  measured local-validity radius costs tens of trajectories, not ~10^5.

Model facts, read off `Reacher-v5` (gymnasium's `assets/reacher.xml`):

    nq = nv = 4, nu = 2      dt = 0.02 (frame_skip 2 x timestep 0.01), 50 Hz
    qpos[0:2]                arm joints: joint0 UNLIMITED (wraps), joint1 +-3.0
    qpos[2:4]                the TARGET's x/y slide joints, +-0.27 -- not the arm
    ctrl in [-1, 1]^2        torque = 200 * ctrl
    dof_damping [1, 1]       armature [1, 1]
    link lengths 0.1, 0.1    fingertip radius over the box: [0.018, 0.210] m

`qpos[2:4]` being the target is the one real trap here: the goal is part of the
simulator state, so "set the goal" means writing into `qpos`, and any code that
resets `qpos` wholesale will silently move the target too.
"""
from __future__ import annotations

import mujoco
import numpy as np

# Arm joints only. qpos[2:4] are the target slides and are NOT part of the state
# the controller acts on.
NQ_ARM = 2
CTRL_HZ = 50.0
# joint0 is unlimited and wraps, so it has no range in the model; +-pi is the
# natural period. joint1's +-3.0 is trimmed only slightly.
#
# The Panda uses SAFE_MARGIN = 0.10 to keep excitation off a hard limit, which is
# a nonlinearity the local libraries would otherwise have to model. That reasoning
# INVERTS here: joint1's limit is what lets the arm fold, and folding is what
# reaches targets near the origin. Measured -- the reachable annulus is
#     |q1| <= 2.40 (margin 0.10) -> [0.0767, 0.21] m
#     |q1| <= 2.88 (margin 0.02) -> [0.0291, 0.21] m
# and goals are drawn uniformly from a disc of radius 0.20, so a 0.10 margin makes
# (0.0767/0.2)^2 = 14.7% of the task IMPOSSIBLE, against 2.1% at 0.02.
SAFE_MARGIN = 0.02
Q0_RANGE = (-np.pi, np.pi)
# Gym samples targets uniformly in a disc of this radius; the arm reaches 0.21.
TARGET_RADIUS = 0.2


def model_path() -> str:
    """Path to gymnasium's bundled `reacher.xml`."""
    import os

    import gymnasium.envs.mujoco as gm

    return os.path.join(os.path.dirname(gm.__file__), "assets", "reacher.xml")


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile `reacher.xml` and return `(MjModel, MjData)`."""
    model = mujoco.MjModel.from_xml_path(model_path())
    return model, mujoco.MjData(model)


def frame_skip(model: mujoco.MjModel, dt_ctrl: float = 1.0 / CTRL_HZ) -> int:
    n = int(round(dt_ctrl / model.opt.timestep))
    if n < 1:
        raise ValueError(f"dt_ctrl={dt_ctrl} shorter than timestep {model.opt.timestep}")
    return n


def safe_box(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Arm-joint box `(lo, hi)`, shape `(2,)` each. joint0 wraps; see module doc."""
    lo1, hi1 = model.jnt_range[1]
    m = SAFE_MARGIN * (hi1 - lo1)
    return (np.array([Q0_RANGE[0], lo1 + m]), np.array([Q0_RANGE[1], hi1 - m]))


def fingertip(data: mujoco.MjData) -> np.ndarray:
    """Fingertip position in the plane, `(2,)`. The z component is constant."""
    return data.body("fingertip").xpos[:2].copy()


def set_state(model, data, q_arm, goal=None) -> None:
    """Place the arm at `q_arm` at rest, optionally moving the target to `goal`.

    Writes `qpos[0:2]` and, when a goal is given, `qpos[2:4]` -- the target's own
    slide joints. Always zeroes `qvel`, so this is a reset, not a nudge.
    """
    data.qpos[:NQ_ARM] = q_arm
    if goal is not None:
        data.qpos[NQ_ARM:] = np.asarray(goal, dtype=np.float64)[:2]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def step_torque(model, data, u, fs: int) -> np.ndarray:
    """Apply `ctrl = clip(u, -1, 1)` for one control step. Returns the applied ctrl."""
    ctrl = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data, nstep=fs)
    # REQUIRED: mj_step returns before the forward pass, so body xpos would
    # otherwise describe the pre-integration pose. Same trap as the Panda.
    mujoco.mj_forward(model, data)
    return ctrl


def sample_config(model, data, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform arm configuration and its fingertip. No rejection needed.

    Unlike the Panda there is nothing to reject: the links cannot self-collide in
    this geometry and the arm's 0.21 m reach never touches the 0.3 m walls.
    """
    lo, hi = safe_box(model)
    q = rng.uniform(lo, hi)
    set_state(model, data, q)
    return q, fingertip(data)


def sample_goal(rng: np.random.Generator) -> np.ndarray:
    """A target drawn as `Reacher-v5` draws it: uniform in a disc of radius 0.2."""
    while True:
        g = rng.uniform(-TARGET_RADIUS, TARGET_RADIUS, 2)
        if np.linalg.norm(g) <= TARGET_RADIUS:
            return g


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]. joint0 is periodic, so raw differences lie."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def config_distance(q: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Distance from `q` to each anchor, wrapping joint0. Shape `(n_anchors,)`.

    joint0 is unlimited, so `-3.1` and `+3.1` rad are 0.08 rad apart, not 6.2.
    Every anchor/coverage computation on this system must use this rather than a
    plain norm.
    """
    d = np.asarray(anchors, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    d = np.column_stack([wrap(d[:, 0]), d[:, 1]])
    return np.linalg.norm(d, axis=1)


# Link lengths, read off the body chain: body1 sits L1 along body0, the fingertip
# L2 along body1. Hard-coding them would rot if the XML changed.
def link_lengths(model) -> tuple[float, float]:
    return float(model.body("body1").pos[0]), float(model.body("fingertip").pos[0])


def reachable_annulus(model) -> tuple[float, float]:
    """Radii the fingertip can attain INSIDE the safe box, `(r_min, r_max)`.

    `r^2 = L1^2 + L2^2 + 2 L1 L2 cos(q1)`, so the inner radius is set by how far
    joint1 may fold -- i.e. by `SAFE_MARGIN`, not by the hardware. That is the
    trap this function exists to make visible: trimming joint1 by 10% (the Panda's
    margin) lifts `r_min` from 0.018 to 0.077 m and makes 14.7% of `Reacher-v5`'s
    goal disc IMPOSSIBLE, which reads as controller failure if not checked.
    """
    L1, L2 = link_lengths(model)
    _, hi = safe_box(model)
    return (float(np.sqrt(L1**2 + L2**2 + 2 * L1 * L2 * np.cos(hi[1]))), L1 + L2)


def is_reachable(model, goal) -> bool:
    """Can the fingertip attain `goal` without leaving the safe box?"""
    r_min, r_max = reachable_annulus(model)
    r = float(np.linalg.norm(np.asarray(goal)[:2]))
    return r_min <= r <= r_max
