"""MuJoCo layer for the Unitree Go1 trot gait: load, step at 50 Hz, state readouts.

Robot model source
------------------
Unitree Go1, MJCF from google-deepmind/mujoco_menagerie, `unitree_go1/scene.xml`
(includes `go1.xml` and adds a plane geom named `floor`; nq = 19, nu = 12).

    upstream : https://github.com/google-deepmind/mujoco_menagerie
    license  : BSD-3-Clause, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
               ("Unitree Robotics") (unitree_go1/LICENSE in that repo)
    revision : menagerie feadf76 (2026-03-18)
    fetched  : by the `robot_descriptions` package into ~/.cache/robot_descriptions/
               on first use -- NOT vendored, so the first run needs network access

Legs, actuators, foot geoms and sites are all looked up by name (FR, FL, RR, RL;
actuators `{leg}_hip/_thigh/_calf`), never by index order.

Imports `mujoco` and `numpy` only -- no `gymnasium`. `model_path()` lazily imports
`robot_descriptions`, mirroring `panda/model.py`.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from quadruped.gait import LEGS

CTRL_HZ = 50.0
FOOT_GEOMS = LEGS
FALL_HEIGHT = 0.15  # trunk z below this (m) counts as fallen
FALL_TILT = 0.8     # |roll| or |pitch| above this (rad) counts as fallen
# A floor contact within this distance (m) of a foot site counts as a foot contact.
# The lower-shin capsule (`calf2`) ends 13 mm above the foot-sphere centre, so it
# brushes the floor (within MuJoCo's 1 mm contact margin) on ordinary steps with
# the trunk upright; measured shin contacts sit 12-16 mm from the foot centre.
FOOT_ZONE = 0.03


def model_path() -> str:
    from robot_descriptions import go1_mj_description

    path = os.path.join(go1_mj_description.PACKAGE_PATH, "scene.xml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Go1 scene.xml not found at {path}; robot_descriptions fetches "
            "mujoco_menagerie on first use and needs network access."
        )
    return path


class Go1Sim:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.frame_skip = round(1.0 / CTRL_HZ / m.opt.timestep)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        self.act_idx = np.array([[aid(f"{l}_{p}") for p in ("hip", "thigh", "calf")] for l in LEGS])
        self.foot_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS]
        self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_site_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in LEGS]
        self.hip_joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{l}_hip_joint") for l in LEGS]
        self._home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home)
        mujoco.mj_forward(self.model, self.data)

    def step(self, q: np.ndarray) -> None:
        self.data.ctrl[self.act_idx] = q
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def _R(self) -> np.ndarray:
        return self.data.xmat[self.trunk_id].reshape(3, 3)

    def trunk_height(self) -> float:
        return float(self.data.qpos[2])

    def rpy(self) -> np.ndarray:
        w, x, y, z = self.data.qpos[3:7]
        return np.array([
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ])

    def body_vel(self) -> np.ndarray:
        """(vx, vy, yaw_rate): linear velocity in the trunk frame, yaw rate about world z."""
        R = self._R()
        v = R.T @ self.data.qvel[0:3]  # freejoint linear velocity is world-frame
        wz = (R @ self.data.qvel[3:6])[2]  # freejoint angular velocity is body-frame
        return np.array([v[0], v[1], wz])

    def fell(self) -> bool:
        if self.trunk_height() < FALL_HEIGHT:
            return True
        if np.any(np.abs(self.rpy()[:2]) > FALL_TILT):
            return True
        feet = self.data.site_xpos[self.foot_site_ids]
        for c in self.data.contact[: self.data.ncon]:
            pair = (c.geom1, c.geom2)
            if self.floor_geom_id in pair:
                other = pair[1] if pair[0] == self.floor_geom_id else pair[0]
                if other not in self.foot_geom_ids and np.linalg.norm(feet - c.pos, axis=1).min() > FOOT_ZONE:
                    return True
        return False

    def foot_rel_hip(self) -> np.ndarray:
        """(4, 3) foot-site position minus its hip-abduction joint, in the trunk frame."""
        hip = self.data.xanchor[self.hip_joint_ids]
        foot = self.data.site_xpos[self.foot_site_ids]
        return (foot - hip) @ self._R()  # row-vector form of R.T @ d
