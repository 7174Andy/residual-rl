"""MuJoCo layer for the Panda reaching env: load, safe joint box, FK, sampling.

Robot model source
------------------
Franka Emika Panda, MJCF from google-deepmind/mujoco_menagerie, file
`franka_emika_panda/panda_nohand.xml` (the no-gripper variant: nq = nv = nu = 7).

    upstream : https://github.com/google-deepmind/mujoco_menagerie
    license  : Apache-2.0 (franka_emika_panda/LICENSE in that repo)
    derived  : from Franka Emika's franka_description URDF,
               https://github.com/frankaemika/franka_ros
    revision : menagerie feadf76d42f8a2162426f7d226a3b539556b3bf5 (2026-03-18)
    fetched  : by the `robot_descriptions` package into
               ~/.cache/robot_descriptions/ on first use -- NOT vendored here,
               so the first run needs network access

The revision matters: every measured constant below is a property of that model
revision, not a law of nature. `scripts/mujoco_hello.py` reprints all of them, so a
menagerie update can be checked against these values rather than assumed compatible.

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
# ponytail: no *colliding* floor geom -- `load_model(scene=True)` adds a
# visual-only one (contype=0 conaffinity=0). Add a colliding plane plus contact
# exclusions if obstacle-aware reaching is ever wanted. Contacts would break the
# local linearity the DeePC stage depends on, which is why none exists now.
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


# Skybox, ground texture and a directional light, so rendered frames show the arm
# against a scene instead of the void `panda_nohand.xml` renders into on its own.
#
# EVERY element here is visual. The floor carries `contype=0 conaffinity=0`, so it
# is invisible to the collision system: `mj_forward` reports the same `ncon`, and
# `nq`/`nu`/`jnt_range`/`actuator_gainprm` are untouched. That is what makes it
# safe to render an episode whose numbers were measured on the bare model -- a
# *colliding* floor would invalidate `data/panda_libraries_v0.npz` and the
# measured constants above. `tests/test_panda_model.py` asserts the equivalence.
#
# The floor is a finite `box` slab, NOT the `plane` a MuJoCo scene would normally
# use, and that is a hard requirement rather than a style choice: MuJoCo sizes an
# infinite plane's render geometry from the current view, so two `render()` calls
# on identical state disagree on ~131k pixels (max 129/255). A finite plane still
# disagrees on ~6. A box is bit-identical, which is what
# `tests/test_panda_env.py::test_render_is_repeatable_and_marker_follows_goal`
# requires and what makes recorded video reproducible.
#
# Its top face is at z=-0.01, just under the arm's base, to avoid z-fighting with
# link0's own geometry at z=0. The tip legitimately dips below z=0 mid-episode
# (see `panda.validity`), so it can visually pass through the floor; nothing
# physical happens when it does.
_SCENE_XML = """
  <statistic center="0.3 0 0.4" extent="1.1"/>
  <visual>
    <rgba haze="0.60 0.68 0.75 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.44 0.60"
             rgb2="0.62 0.72 0.82" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge"
             rgb1="0.30 0.34 0.39" rgb2="0.22 0.26 0.31"
             markrgb="0.75 0.78 0.80" width="300" height="300"/>
    <!-- reflectance MUST stay 0. A reflective floor renders the goal marker
         (injected as a scene geom in panda/rendering.py, not a model body)
         nondeterministically: repeated render() calls on identical state
         disagreed on up to 33 pixels at reflectance 0.15, all inside the
         marker's reflection. Model geoms reflect fine; the injected one does
         not. Turning the reflection pass off is the cheap half of that
         trade -- the sheen is worth less than reproducible video. -->
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6"
              reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.5 0.6 2.0" dir="-0.2 -0.25 -1" directional="true"
           diffuse="0.45 0.45 0.45"/>
    <geom name="backdrop_floor" type="box" pos="0 0 -0.06" size="3 3 0.05"
          material="grid" contype="0" conaffinity="0"/>
  </worldbody>
"""

SCENE_FLOOR_GEOM = "backdrop_floor"


def load_model(
    scene: bool = False, extra_xml: str = ""
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the model and return `(MjModel, MjData)`.

    `scene=True` splices `_SCENE_XML` in for a renderable backdrop. Physics is
    unchanged either way (see `_SCENE_XML`); pass it only when rendering.

    `extra_xml` splices caller-supplied top-level MJCF in the same place, so a
    caller can add bodies/cameras without duplicating the asset plumbing below.
    Unlike `_SCENE_XML` this is NOT guaranteed physics-neutral -- that is on the
    caller (mocap bodies with `contype=0 conaffinity=0` are; anything with a
    joint or a colliding geom is not, and would invalidate the constants above).
    """
    path = model_path()
    if scene or extra_xml:
        # Splicing XML text and supplying the meshes in memory, rather than the
        # obvious `<include file="panda_nohand.xml"/>` from a wrapper scene file:
        # MuJoCo resolves `meshdir` against the *including* file's directory, so
        # an include from anywhere outside menagerie's own directory cannot find
        # `assets/*.stl` -- and writing a wrapper into that directory means
        # writing into ~/.cache, which `robot_descriptions` owns and may wipe.
        assets_dir = os.path.join(os.path.dirname(path), "assets")
        with open(path) as fh:
            splice = (_SCENE_XML if scene else "") + extra_xml
            xml = fh.read().replace("</mujoco>", splice + "</mujoco>")
        assets = {}
        for name in os.listdir(assets_dir):
            with open(os.path.join(assets_dir, name), "rb") as fh:
                assets[f"assets/{name}"] = fh.read()
        model = mujoco.MjModel.from_xml_string(xml, assets)
    else:
        model = mujoco.MjModel.from_xml_path(path)
    # `panda_nohand.xml` ships no lights -- menagerie puts them in `scene.xml`,
    # which includes the *with-hand* model and so is unusable here. Bump the
    # built-in headlight so `rgb_array` frames are lit even without `scene`.
    # Visual only: `vis` does not participate in dynamics, so this cannot
    # affect any rollout.
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
