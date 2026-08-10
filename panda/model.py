"""MuJoCo layer for the Panda reaching env: load, safe joint box, FK, sampling.

Imports `mujoco` and `numpy` only -- no `gymnasium`. This mirrors the role of
`two_wheel_robot/env/dynamics.py`: the physics is usable from controllers and
tests without pulling in Gym. The one exception is `model_path()`, which
lazily imports `robot_descriptions` to resolve the model file; that package's
first call shallow-clones `mujoco_menagerie` into `~/.cache/robot_descriptions/`,
so the first run after a fresh clone needs network access -- every call after
reads from the cache.

All constants here were measured against the real model; see
`docs/superpowers/specs/2026-08-10-panda-reach-env-design.md` for the numbers and
the reasoning. `scripts/mujoco_hello.py` reproduces them.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

# The only site in `panda_nohand.xml` -- the flange where a hand would attach.
TIP_SITE = "attachment_site"

# Fraction of each joint's span trimmed at BOTH ends. A joint resting against its
# limit is a hard nonlinearity, and the DeePC libraries built on this env in a
# later spec need locally-linear data; keeping excitation off the limits protects
# that. Measured self-collision rate inside the resulting box: 0.4%.
SAFE_MARGIN = 0.10

# `panda_nohand.xml` has no ground plane, so nothing stops the arm reaching below
# z=0. Start and goal configurations are rejected below this height so episodes
# begin and end above ground, even though the arm may pass under it mid-episode.
# ponytail: no floor geom -- add `<geom type="plane">` plus contact exclusions if
# obstacle-aware reaching is ever wanted. Contacts would break the local
# linearity the DeePC stage depends on, which is why it is absent now.
MIN_TIP_Z = 0.05

CTRL_HZ = 50.0

# Tip distance from the base over 4000 uniform safe-box samples. Used by tests to
# catch "wrong model loaded" / "wrong site" without re-deriving kinematics.
TIP_RADIUS_RANGE = (0.030, 1.186)


def model_path() -> str:
    """Absolute path to menagerie's `panda_nohand.xml`.

    Deliberately not `panda_mj_description.MJCF_PATH`, which points at the
    with-hand `panda.xml`. The no-hand variant gives a clean nq = nv = nu = 7.
    """
    from robot_descriptions import panda_mj_description

    path = os.path.join(panda_mj_description.PACKAGE_PATH, "panda_nohand.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"panda_nohand.xml not found at {path}. `robot_descriptions` clones "
            "mujoco_menagerie into ~/.cache/robot_descriptions on first use; "
            "delete that directory and retry, or vendor the model plus its "
            "assets/ into panda/assets/ and point model_path() at it."
        )
    return path


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the model and return `(MjModel, MjData)`."""
    model = mujoco.MjModel.from_xml_path(model_path())
    # `panda_nohand.xml` ships no lights -- menagerie puts them in `scene.xml`,
    # which includes the *with-hand* model and so is unusable here. Bump the
    # built-in headlight so `rgb_array` frames are lit. Visual only: `vis` does
    # not participate in dynamics, so this cannot affect any rollout.
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.8
    return model, mujoco.MjData(model)


def frame_skip(model: mujoco.MjModel, dt_ctrl: float = 1.0 / CTRL_HZ) -> int:
    """Physics steps per control step. 0.02 s / 0.002 s timestep -> 10."""
    n = int(round(dt_ctrl / model.opt.timestep))
    if n < 1:
        raise ValueError(
            f"dt_ctrl={dt_ctrl} is shorter than the model timestep "
            f"{model.opt.timestep}; frame_skip would round to {n}"
        )
    return n


def safe_box(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Joint ranges trimmed by `SAFE_MARGIN` at both ends. Returns `(lo, hi)`."""
    lo = model.jnt_range[:, 0].astype(np.float64)
    hi = model.jnt_range[:, 1].astype(np.float64)
    margin = SAFE_MARGIN * (hi - lo)
    return lo + margin, hi - margin


def tip_id(model: mujoco.MjModel) -> int:
    """Site id of the end-effector. Cache this; do not call it per step."""
    return int(model.site(TIP_SITE).id)


def tip_position(data: mujoco.MjData, tip_site_id: int) -> np.ndarray:
    """End-effector world position `(3,)`, as a copy.

    Valid only after `mj_forward` (or `mj_kinematics`). Note that `mj_step` alone
    leaves `site_xpos` holding *pre-integration* kinematics -- see `env.step`.
    """
    return data.site_xpos[tip_site_id].copy()


def apply_delta(
    data: mujoco.MjData, u: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Write `ctrl = clip(q_current + u, safe_box)` and return it.

    The model's actuators are PD position servos (`force = kp(ctrl - q) - kd*qdot`)
    whose `ctrlrange` equals the joint range, so `ctrl` is an absolute joint
    target. Integrating the delta from *measured* `q` rather than from the
    previous target bounds the servo lag and makes every command achievable, at
    the cost of capping joint speed at roughly `(kp/kd) * delta_max`.
    """
    ctrl = np.clip(data.qpos + np.asarray(u, dtype=np.float64), lo, hi)
    data.ctrl[:] = ctrl
    return ctrl


def sample_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    lo: np.ndarray,
    hi: np.ndarray,
    tip_site_id: int,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample a valid configuration. Returns `(qpos, tip)`.

    Valid means: inside the safe box, self-collision free (`ncon == 0`), and tip
    at or above `MIN_TIP_Z`. Measured rejection rate is ~3.5% (0.4% contact,
    3.1% low tip), so exhausting `max_attempts` indicates a misconfiguration
    rather than bad luck -- hence the raise with a breakdown by cause.

    NOTE: this mutates `data` (qpos, qvel, and everything `mj_forward` derives).
    Callers that care about `data`'s prior contents must save it themselves.
    """
    n_contact = n_low = 0
    for _ in range(max_attempts):
        q = rng.uniform(lo, hi)
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            n_contact += 1
            continue
        tip = data.site_xpos[tip_site_id].copy()
        if tip[2] < MIN_TIP_Z:
            n_low += 1
            continue
        return q, tip
    raise RuntimeError(
        f"sample_config failed in {max_attempts} attempts: "
        f"{n_contact} rejected for self-collision, {n_low} for tip below "
        f"z={MIN_TIP_Z}. Check SAFE_MARGIN and that panda_nohand.xml loaded."
    )
