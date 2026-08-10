"""Offscreen MuJoCo renderer for PandaReachEnv. `rgb_array` only.

Interactive viewing is deliberately not implemented. `mujoco.viewer.launch_passive`
raises on macOS unless the process was started with the `mjpython` launcher, and
MuJoCo already ships a standalone viewer, so there is nothing worth writing:

    uv run mjpython -m mujoco.viewer --mjcf="$(uv run python -c \
        'import panda.model as m; print(m.model_path())')"

The goal marker's radius is read from `env.goal_tolerance`, so the sphere *is*
the tolerance ball -- when the tip visually enters it, the episode has reached.
"""
from __future__ import annotations

import mujoco
import numpy as np

GOAL_RGBA = np.array([0.2, 0.9, 0.2, 0.9], dtype=np.float32)
_IDENTITY_MAT = np.eye(3, dtype=np.float64).flatten()


class MujocoRenderer:
    def __init__(self, env, height: int = 480, width: int = 640):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self.camera)
        # Distance is set to clear the workspace's reachable extremes (tip z up
        # to ~1.18, radius up to 1.186 -- see panda.model.TIP_RADIUS_RANGE)
        # across sampled configurations, not just the home keyframe, so the
        # video cannot silently crop the arm at the poses it most needs to show.
        self.camera.distance = 2.4
        self.camera.azimuth = 135.0
        self.camera.elevation = -20.0
        self.camera.lookat[:] = [0.3, 0.0, 0.4]

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=self.camera)
        scene = self.renderer.scene
        # The goal is not a body in the model, so inject it as a scene geom after
        # update_scene (which resets ngeom) and before render.
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self.env.goal_tolerance, dtype=np.float64),
                np.asarray(self.env.goal, dtype=np.float64),
                _IDENTITY_MAT,
                GOAL_RGBA,
            )
            scene.ngeom += 1
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
