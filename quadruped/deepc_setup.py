"""Canonical DeePC / Select-DPC builder for `QuadrupedGoal-v0`, plus `run_episode`.

Mirrors `two_wheel_robot/rl/deepc_setup.py` (copied, not imported): heading-keyed
libraries, bearing reference, Q = diag(1, 1, 2), lambda_g=2, lambda_y=3e6, SCS.
Defaults T_ini=20, N=25 (the unicycle's 5/12 is the "straight port"; see below).

Yaw branch: yaw is wrapped to [-pi, pi], so a trace crossing +-pi jumps by 2pi
and would corrupt the Hankel past window. `_YawBranch` unwraps yaw (current, past
buffer, and bearing reference) onto the branch nearest the selected library's
anchor (or, for Select-DPC, the previous yaw wrapped to [-pi, pi], so the
window stays contiguous but cannot drift 2*pi off the bank after a crossing). The libraries in
`data/quadruped_libraries_v0.npz` measured n_wraps = 0,0,0,0 (re-measure:
`.venv/bin/python scripts/collect_quadruped_data.py`), but closed-loop runs can
still cross +-pi, so the branch path is always taken.

Measured lag (command -> measured velocity, 63% of the step; corrected ~= raw minus the
~0.2 s the 20-step trailing average adds, +-0.05 s), 2026-10-30:
`.venv/bin/python scripts/measure_quadruped_lag.py`

    axis     c0     c1  raw63_s  corr63_s   final   gain
    vx      0.0    0.3     0.34      0.14   0.200   0.67
    vx      0.3    0.6     0.34      0.14   0.510   0.85
    vx      0.0   -0.2     0.32      0.12  -0.274   1.37
    vy      0.0    0.2     0.34      0.14   0.215   1.07
    vy      0.0   -0.2     0.36      0.16  -0.219   1.10
    yaw     0.0    0.8     0.28      0.08   0.737   0.92
    yaw     0.0   -0.8     0.30      0.10  -0.732   0.91
    yaw     0.4   -0.4     0.30      0.10  -0.333   0.83

Results on the frozen scenarios in `data/quadruped_scenarios_v1.npz` (2026-10-31; journey 21):

    setting          split      reached        Wilson 95%    falls  qp_fail  solve/step
    T_ini=5,  N=12   tune  40   14/40 (35%)    [0.22, 0.50]  0      0         3.6 s
    T_ini=5,  N=12   eval  78   22/78 (28%)    [0.19, 0.39]  0      0         3.7 s
    T_ini=20, N=25   tune  40   28/40 (70%)    [0.55, 0.82]  0      0        16.9 s
    T_ini=20, N=25   eval  78   60/78 (77%)    [0.66, 0.85]  0      0        19.1 s

Solve times are with 10 workers on an M4 Pro. Re-measure (resumable; an eval
run counts as an evaluation look in data/quadruped_deepc/eval_looks.csv):
`.venv/bin/python scripts/run_quadruped_deepc.py --split eval --T-ini 20 --N 25 --out <csv>`
"""
from __future__ import annotations

import time

import numpy as np

from core.deepc import DeePC
from core.hankel import build_hankel
from core.selectdpc import SelectDPC, trajectory_bank

DEFAULT_LIBRARIES = "data/quadruped_libraries_v0.npz"

# Copied from QuadrupedGoalEnv defaults (quadruped/env.py); equality is asserted in tests.
_R = np.diag([3.3e-2, 3.3e-2, 1.1e-4])
_ACTION_BOUNDS = np.array([[-0.25, 0.6], [-0.2, 0.2], [-0.8, 0.8]])
_Q = np.diag([1.0, 1.0, 2.0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bearing_y_ref(y, goal) -> np.ndarray:
    """`(g_x, g_y, atan2(g_y - y, g_x - x))`, as in the unicycle's."""
    y = np.asarray(y, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return np.array([goal[0], goal[1], np.arctan2(goal[1] - y[1], goal[0] - y[0])])


class _YawBranch:
    """Mixin: unwrap yaw onto one branch before the QP sees it (see module docstring)."""

    def act(self, y_current, y_ref):
        y, ref = np.array(y_current, float), np.array(y_ref, float)
        c = (self.anchor_headings[self._select_index_for(y)] if self._n_lib > 1
             else _wrap(self._y_buf[-1, 2]))             # select: past window, re-centred on principal branch
        y[2] = c + _wrap(y[2] - c)
        self._y_buf[:, 2] = c + _wrap(self._y_buf[:, 2] - c)
        ref[2] = y[2] + _wrap(ref[2] - y[2])             # nearest-branch bearing
        return super().act(y, ref)


class QuadDeePC(_YawBranch, DeePC):
    pass


class QuadSelectDPC(_YawBranch, SelectDPC):
    pass


def build_canonical_quadruped_deepc(
    libraries_path: str = DEFAULT_LIBRARIES,
    T_ini: int = 20,
    N: int = 25,
    lambda_g: float = 2.0,
    lambda_y: float = 3e6,
    affine: bool = False,
    du_scale: float | None = 25.0,
    kind: str = "deepc",
    n_cols: int = 300,
):
    """Return `(controller, info)`. `du_scale=None` disables the rate limit."""
    if kind not in ("deepc", "select"):
        raise ValueError(f"kind must be 'deepc' or 'select', got {kind!r}")
    with np.load(libraries_path) as data:
        anchors = np.asarray(data["anchors"], dtype=np.float64)
        uy = [(data[f"u_{i}"], data[f"y_{i}"]) for i in range(len(anchors))]
    ab = _ACTION_BOUNDS.copy()
    du_max = None if du_scale is None else (ab[:, 1] - ab[:, 0]) / du_scale
    common = dict(Q=_Q.copy(), R=_R.copy(), T_ini=T_ini, N=N, lambda_g=lambda_g,
                  lambda_y=lambda_y, u_bounds=(ab[:, 0], ab[:, 1]),
                  du_max=du_max, affine=affine)
    if kind == "deepc":
        libs = [build_hankel(u, y, T_ini=T_ini, N=N) for u, y in uy]
        ctrl = QuadDeePC(libs, anchor_headings=anchors, **common)
    else:
        bank = trajectory_bank([u for u, _ in uy], [y for _, y in uy], T_ini, N)
        ctrl = QuadSelectDPC(bank, anchor_headings=np.zeros(1), n_cols=n_cols,
                             n_max=1, carry_prediction=False, **common)
    info = {
        "kind": kind, "anchors": anchors, "u_init": np.zeros(3),
        "action_bounds": ab, "Q": _Q.copy(), "R": _R.copy(), "T_ini": T_ini, "N": N,
        "lambda_g": lambda_g, "lambda_y": lambda_y, "affine": affine,
        "du_max": du_max, "n_cols": n_cols, "libraries_path": libraries_path,
    }
    return ctrl, info


def run_episode(ctrl, info, env, scenarios, i) -> dict:
    """Run scenario `i` to termination. A QP `RuntimeError` ends it with `qp_fail=True`."""
    from quadruped.scenarios import reset_to

    reset_to(env, scenarios, i)
    ctrl.reset(env.y, info["u_init"])
    reached = fell = truncated = qp_fail = False
    steps, times = 0, []
    while True:
        t0 = time.perf_counter()
        try:
            u = ctrl.act(env.y, bearing_y_ref(env.y, env.goal))
        except RuntimeError:
            qp_fail = True
            break
        times.append(time.perf_counter() - t0)
        _, _, term, trunc, inf = env.step(u)
        steps += 1
        reached, fell, truncated = bool(inf.get("reached", False)), bool(inf.get("fell", False)), bool(trunc)
        if term or trunc:
            break
    return {
        "id": int(i), "reached": reached, "fell": fell, "truncated": truncated,
        "qp_fail": qp_fail, "steps": steps,
        "final_dist": float(np.linalg.norm(env.y[:2] - env.goal)),
        "mean_solve_s": float(np.mean(times)) if times else 0.0,
    }
