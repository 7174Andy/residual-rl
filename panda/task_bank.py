"""Goal-directed task-manifold collection for the Panda bank.

panda/data_collection.py's anchor libraries excite locally around four fixed
joint-1 anchors -- great near an anchor (skill 0.93, cos 0.98 per
panda/selectdpc.py's docstring) and useless past ~2 rad from one. This module
collects a structurally different kind of data: start at one valid
configuration, servo toward another (the goal's generating configuration)
with exploration noise, and record the plant's TRUE input `ctrl` (the box
clip fires on 24-48% of steps under excitation; `u -> y` would mislabel
those, per panda/env.py's module docstring).

Payload feeds two real downstream consumers, which is complicated by a real
u-CHANNEL SEMANTIC COLLISION between them (fix round 1 -- round 0's naive
"superset" shipped both consumers reading raw `u_i` and silently corrupted
Select-DPC's Willems regression; see `for_select_dpc` below):

1. The anchor/Hankel path -- `panda/deepc_setup.py::build_canonical_panda_deepc`
   (line 98: `uy = [(z[f"u_{i}"], z[f"{key}_{i}"]) ...]`, fed straight into
   `core.hankel.build_hankel`; no `ctrl_i` read anywhere in that file) and
   `panda/data_collection.py::coverage_report`. Its `u_i` is the DELTA
   `panda/env.py::PandaReachEnv.step()` takes -- confirmed by
   `panda/eval.py:99` replaying the identified policy's output straight
   through `env.step(u)`, whose `apply_delta` computes
   `ctrl = clip(q + u, safe_box)`. This module's `u_i`/`ctrl_i`/`y_i`/`yext_i`
   match `data_collection.collect_libraries`'s schema exactly (verified
   against `data/panda_libraries_v2.npz`'s own key list).
2. Select-DPC -- `panda/selectdpc.py::panda_bank`, `panda/qdes.py::build_libraries`,
   called the way `scripts/run_select_dpc.py` / `scripts/measure_selection_distance.py`
   call them. Its `u_i` is an ABSOLUTE `q_des` target: `panda/qdes.py::collect_anchor`
   (its collection) records `ctrl` (clipped absolute) as `u`, and
   `scripts/run_select_dpc.py:148` feeds the controller's output straight into
   `panda/qdes.py::step_qdes`, which does `ctrl = clip(q_des, safe_box)` --
   NEVER through `env.step()`'s delta interface. `panda/selectdpc.py`'s own
   module docstring calls this "the `u = q_des` adapter".

Both consumers hardcode the literal key `u_i`, with opposite units (a ~0.2 rad
delta vs. an absolute joint angle spanning the whole safe box) -- so ONE flat
dict cannot serve both under that name; there is no key-choice that avoids
this, short of editing `panda/selectdpc.py` (out of this module's scope).
The split: this module's own payload (and the npz `collect_panda_taskbank.py`
writes) keeps `u_i` = DELTA, matching consumer 1 -- the pipeline this task
exists to fix. `for_select_dpc()` below returns a SEPARATE dict, re-keyed for
consumer 2, built from `ctrl_i` (the recorded absolute target -- exactly
`panda/qdes.py`'s `u` semantics). Do NOT hand `panda_bank`/`build_libraries`
the raw payload or the on-disk npz directly -- only through `for_select_dpc`.

`tip_i` is exactly `y_i` (`env.y` already *is* the tip position) and `q_i` is
`q[:-1]` from the same rollout -- those two have no semantic collision, so
they cost only a few array aliases, not a second collection pass. `anchors`
is each trajectory's own start configuration: with no fixed anchor grid here,
that is the natural per-trajectory neighbourhood for nearest-anchor keying.

Uses `panda/env.py::PandaReachEnv`'s real accessors, not the aspirational
`env.reset_to`/`env.step_raw`/`env.q` names: reset-to-config is
`env.reset(options={"qpos": ...})`, current joints is `env.data.qpos`, and
the extended output is the `env.y_ext` property (not in `step()`'s info
dict -- only `info["ctrl"]`/`info["y"]`/etc. are).
"""
from __future__ import annotations

import numpy as np

from panda.model import sample_config


def servo_trajectory(env, T: int, rng: np.random.Generator,
                      alpha: float = 0.35, noise_sigma: float = 0.04) -> dict:
    """One trajectory: servo q -> q_goal with noise. Returns u/ctrl/y/yext/q."""
    lo, hi = env.safe_box
    q_start, _ = sample_config(env.model, env.data, rng, lo, hi, env.tip_site_id)
    q_goal, _ = sample_config(env.model, env.data, rng, lo, hi, env.tip_site_id)
    env.reset(options={"qpos": q_start})

    nq = env.nq
    u = np.zeros((T, nq)); ctrl = np.zeros((T, nq))
    y = np.zeros((T, 3)); yext = np.zeros((T, 10))
    q = np.zeros((T + 1, nq))
    q[0] = np.asarray(env.data.qpos, dtype=np.float64).copy()
    for t in range(T):
        step_u = np.clip(
            alpha * (q_goal - np.asarray(env.data.qpos, dtype=np.float64))
            + rng.normal(0.0, noise_sigma, nq),
            -env.delta_max, env.delta_max,
        )
        # y/yext observed BEFORE u_t is applied, matching data_collection.py's
        # alignment so y_{t+1} is the response to u_t.
        y[t] = env.y
        yext[t] = env.y_ext
        _, _, _, _, info = env.step(step_u)
        u[t] = step_u
        ctrl[t] = info["ctrl"]
        q[t + 1] = np.asarray(env.data.qpos, dtype=np.float64).copy()
    return {"u": u, "ctrl": ctrl, "y": y, "yext": yext,
            "q": q, "q_goal": q_goal}


def collect_task_bank(env, n_traj: int, T: int, seed: int,
                       alpha: float = 0.35, noise_sigma: float = 0.04) -> dict:
    rng = np.random.default_rng(seed)
    payload: dict = {"n_traj": np.int64(n_traj), "T": np.int64(T),
                      "meta_alpha": alpha, "meta_noise_sigma": noise_sigma,
                      "meta_seed": np.int64(seed),
                      "meta_kind": "task_servo"}
    anchors = np.zeros((n_traj, env.nq))
    for i in range(n_traj):
        out = servo_trajectory(env, T, rng, alpha, noise_sigma)
        for k in ("u", "ctrl", "y", "yext"):
            payload[f"{k}_{i}"] = out[k]
        # Select-DPC-compatible aliases (see module docstring): tip_i IS y_i
        # (env.y already is the tip position), q_i drops the trailing extra
        # sample servo_trajectory keeps for the distance-reduction test.
        payload[f"q_{i}"] = out["q"][:-1]
        payload[f"tip_{i}"] = out["y"]
        anchors[i] = out["q"][0]
    payload["anchors"] = anchors
    return payload


def for_select_dpc(payload: dict) -> dict:
    """Re-key a `collect_task_bank` payload for `panda_bank`/`build_libraries`.

    Those hardcode `u_i` as an ABSOLUTE `q_des` target (see module docstring);
    this payload's own `u_i` is the DELTA `env.step()` takes. Returns a NEW
    dict with `u_i` re-pointed at `ctrl_i` (the recorded absolute target that
    was actually applied -- exactly `panda/qdes.py`'s `u` semantics). `q_i`/
    `tip_i`/`anchors` carry over unchanged; they have no such collision.
    """
    n = int(payload["anchors"].shape[0])
    out = {"anchors": payload["anchors"]}
    for i in range(n):
        out[f"u_{i}"] = payload[f"ctrl_{i}"]
        out[f"q_{i}"] = payload[f"q_{i}"]
        out[f"tip_{i}"] = payload[f"tip_{i}"]
    return out
