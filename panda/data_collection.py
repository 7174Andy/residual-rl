"""Offline PE data collection on PandaReach-v0, for DeePC Hankel libraries.

A "library" is a length-T sequence of aligned (u_t, y_t) pairs collected around
one anchor configuration, exactly as the unicycle's
`two_wheel_robot/controllers/data_collection.py` does: `y_t` is the output
observed *before* `u_t` is applied, so `y_{t+1}` is the response to `u_t`.

Two settings here are measured, not inherited, and both matter:

1. The anchors are interior `q1` values. The unicycle's heading anchors
   (pi/4, 3pi/4, 5pi/4, 7pi/4) map to q1 = +-0.785, +-2.356 -- and joint 1's safe
   box is +-2.318, so two of them clamp to the boundary.

2. The excitation carries a restoring term toward the anchor. Plain OU is an
   integrated random walk: it reaches the safe-box edge and stays there, and the
   env's `ctrl = clip(q + u, safe_box)` then makes the recorded `u` differ from
   what the plant received on 41% of steps. Every such pair is a lie to the
   Hankel. With the restoring term the rate is 0.00% at T=400.
"""
from __future__ import annotations

import numpy as np

from panda.model import safe_box

LIBRARIES_PATH = "data/panda_libraries_v0.npz"

# Interior to joint 1's safe box of +-2.318, unlike the unicycle's heading anchors.
PANDA_ANCHOR_Q1 = (-1.8, -0.6, 0.6, 1.8)

OU_THETA = 0.85        # correlation time ~ 1/(1-theta) ~ 7 control steps
OU_SIGMA_FRAC = 0.6    # sigma = OU_SIGMA_FRAC * delta_max
K_RET = 0.05           # weakest restoring gain that still gives 0% clipping
# 400 was the earlier value, chosen because rank is already full there ("so more is
# waste"). That reasoning gated on the wrong property: full row rank means the library
# is persistently exciting, not that it VISITS the configurations the goals need.
# Measured counterexample -- rank 170 of 170 and still 0/10 reached beyond 60 deg from
# an anchor. The comparison that motivated 3000 is against the unicycle, which the
# pipeline is supposed to treat comparably:
#
#              T     n_cols   Hankel rows   cols/rows
#   unicycle   1500    1484        85          17.5x
#   Panda@400   400     384       170           2.26x   (289 / 1.33x for output="ext")
#   Panda@3000 3000    2984       170          17.6x    (289 / 10.3x)
#
# So 3000 buys margin parity with the unicycle, and relieves the extended output's
# tight 1.33x. It costs ~60 s of sim per anchor. NOTE it buys REDUNDANCY, not reach:
# with K_RET pulling back toward the anchor, a longer walk stays in the same
# neighbourhood. Azimuth coverage is a separate knob (anchor placement), and changing
# both at once would make the result unattributable.
DEFAULT_T = 3000

# m_u(T_ini + N) + n_state = 7*17 + 14
RANK_FLOOR = 133


def anchor_qpos(model, q1: float) -> np.ndarray:
    """The `home` keyframe with joint 1 replaced by `q1`."""
    q = np.asarray(model.key_qpos[0], dtype=np.float64).copy()
    q[0] = float(q1)
    return q


def _reset_to_anchor(env, anchor: np.ndarray):
    """Reset to `anchor` with the goal placed far outside the workspace.

    The far goal is the unicycle's trick: it stops the env terminating
    mid-collection, since only (u, y) matter here.
    """
    import warnings

    with warnings.catch_warnings():
        # The far goal pushes tip-goal past the obs bound, so Gym's passive
        # checker warns on every reset and step. Expected; obs is unused here.
        warnings.filterwarnings("ignore", message=".*not within the observation space.*",
                                category=UserWarning)
        return env.reset(options={"qpos": anchor,
                                  "goal": np.array([100.0, 100.0, 100.0])})


def collect_trajectory(env, anchor: np.ndarray, T: int, rng: np.random.Generator,
                       theta: float = OU_THETA, sigma_frac: float = OU_SIGMA_FRAC,
                       k_ret: float = K_RET) -> dict:
    """One library's worth of data around `anchor`. Returns u, ctrl, y, clip_frac."""
    import warnings

    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    lo, hi = safe_box(env.model)
    sigma = sigma_frac * env.delta_max

    _reset_to_anchor(env, anchor)
    u = np.zeros(env.nq, dtype=np.float64)
    u_traj = np.zeros((T, env.nq)); ctrl_traj = np.zeros((T, env.nq))
    y_traj = np.zeros((T, 3)); n_clip = 0
    # Both output definitions from the SAME trajectory, so a tip-only vs
    # tip+configuration comparison varies only the output map -- never the data.
    yext_traj = np.zeros((T, 3 + env.nq))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not within the observation space.*",
                                category=UserWarning)
        for t in range(T):
            u = np.clip(theta * u + sigma * rng.standard_normal(env.nq)
                        + k_ret * (anchor - env.data.qpos),
                        -env.delta_max, env.delta_max)
            q_before = np.asarray(env.data.qpos).copy()
            # STRUCTURAL guarantee that the recorded u is what the plant receives.
            # `apply_delta` computes ctrl = clip(q + u, safe_box); if that clip ever
            # fires, the recorded u is a lie to the Hankel. Bounding u to
            # [lo - q, hi - q] here makes q + u already inside the safe box, so the
            # env's clip is a no-op and u is faithful BY CONSTRUCTION rather than by
            # luck. Measured over 8 seeds: without this the clip fires on 0.00-2.69%
            # of steps depending on seed (the restoring term alone is not enough at
            # the outer anchors, which sit only 0.518 rad from joint 1's edge);
            # with it, 0.00% at every seed, with tip-radius coverage unchanged.
            u = np.clip(u, lo - q_before, hi - q_before)
            y_traj[t] = env.y            # observed BEFORE u_t is applied
            yext_traj[t] = env.y_ext     # same instant, richer output
            _, _, _, _, info = env.step(u)
            u_traj[t] = env.last_action
            ctrl_traj[t] = info["ctrl"]
            if not np.allclose(info["ctrl"], np.clip(q_before + u_traj[t], lo, hi),
                               atol=1e-9) or not np.allclose(
                                   info["ctrl"], q_before + u_traj[t], atol=1e-9):
                n_clip += 1

    return {"u": u_traj, "ctrl": ctrl_traj, "y": y_traj, "yext": yext_traj,
            "clip_frac": n_clip / T}


def collect_libraries(env, T: int = DEFAULT_T,
                      rng: np.random.Generator | None = None) -> dict:
    """One library per anchor. Returns the npz payload."""
    if rng is None:
        rng = np.random.default_rng()
    payload: dict = {}
    azimuths = []
    clips = []
    for i, q1 in enumerate(PANDA_ANCHOR_Q1):
        anchor = anchor_qpos(env.model, q1)
        out = collect_trajectory(env, anchor, T, rng)
        payload[f"u_{i}"] = out["u"]
        payload[f"ctrl_{i}"] = out["ctrl"]
        payload[f"y_{i}"] = out["y"]
        payload[f"yext_{i}"] = out["yext"]
        clips.append(out["clip_frac"])
        # Azimuth is MEASURED from the anchor's FK tip, not assumed to equal q1.
        _reset_to_anchor(env, anchor)
        tip = env.y
        azimuths.append(float(np.arctan2(tip[1], tip[0])))
    payload["anchor_q1"] = np.asarray(PANDA_ANCHOR_Q1, dtype=np.float64)
    payload["anchor_azimuths"] = np.asarray(azimuths, dtype=np.float64)
    payload["clip_frac"] = np.asarray(float(np.mean(clips)))
    payload["delta_max"] = np.asarray(float(env.delta_max))
    payload["excitation"] = np.asarray([OU_THETA, OU_SIGMA_FRAC, K_RET], dtype=np.float64)
    payload["T"] = np.asarray(int(T))
    return payload


def coverage_report(payload: dict, T_ini: int = 5, N: int = 12) -> dict:
    """Rank and spectrum per library, plus clip rate and tip coverage.

    The cheapest possible early warning that a library is junk -- run this before
    spending closed-loop time on it.
    """
    from core.hankel import build_hankel

    libs = []
    radii = []
    n_lib = int(payload["anchor_q1"].shape[0])
    for i in range(n_lib):
        blocks = build_hankel(payload[f"u_{i}"], payload[f"y_{i}"], T_ini=T_ini, N=N)
        M = np.vstack(blocks)
        s = np.linalg.svd(M, compute_uv=False)
        libs.append({
            "index": i,
            "n_cols": int(blocks[0].shape[1]),
            "rank": int(np.linalg.matrix_rank(M)),
            "s_ratio_133": float(s[min(RANK_FLOOR - 1, len(s) - 1)] / s[0]),
        })
        radii.append(np.linalg.norm(payload[f"y_{i}"], axis=1))
    r = np.concatenate(radii)
    return {
        "libraries": libs,
        "clip_frac": float(payload["clip_frac"]),
        "tip_radius_min": float(r.min()),
        "tip_radius_max": float(r.max()),
        "anchor_azimuths": payload["anchor_azimuths"].tolist(),
    }
