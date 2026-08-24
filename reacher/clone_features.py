"""The behavioral clone's feature vector for Reacher.

The Reacher analogue of `two_wheel_robot/rl/features.py`, with one structural
difference in one case only: **whether a library one-hot is available**.

The unicycle clone was told which of four libraries DeePC had selected, and that
is load-bearing -- `select_library_index` is an `argmin`, so crossing a boundary
makes the controller's output jump, and without the index the network sees two
near-identical states with very different actions. `data/clone.pt` ships with
`n_lib = 4` and those columns protected from standardization.

Select-DPC has no such index: it re-selects 300 columns of ~17,760 every step, so
there is nothing discrete to hand over and `anchors=None` is correct.

`ReacherDeePC` DOES have one -- it picks its library by nearest anchor -- so the
fixed-anchor base gets the same treatment the unicycle got. The index here must
match `ReacherDeePC._select_index_for` exactly, or the clone is told the wrong
mode; `tests/test_reacher_clone_features.py` pins the two together.

Two conventions, both load-bearing:

* **joint0 enters as `(cos, sin)`.** It is unlimited and wraps, so `-pi` and
  `+pi` are the same configuration; a raw angle hands the network a
  discontinuity exactly there. joint1 is range-limited and enters raw.
* **The goal enters as `tip - goal`,** not as absolute `(g_x, g_y)`, matching
  `ReacherGoalEnv`'s observation for the same generalization reason: the
  controller's job depends on the relative displacement, not on where in the
  plane the pair happens to sit.

The fingertip *itself* stays absolute in the buffer blocks, and that asymmetry is
deliberate. The arm is anchored at the origin, so an absolute fingertip position
IS physically meaningful state -- it is the forward kinematics of `q`, and the
local dynamics differ between an extended and a folded arm at the same relative
goal. Making the whole vector translation-invariant would delete information the
clone needs. Only the goal is arbitrary in the plane, and only the goal is
relative.

**Buffer validity is the 43rd feature, and it is not cosmetic.** At `t = 0` there
is no history, so `rollout` primes the buffer with `u_ini = zeros` and
`y_ini = tile(y0)`. The first `T_ini` steps are therefore a structurally different
regime, and without this feature the clone cannot tell a part-primed buffer at
step 3 from real history that happens to look similar. Measured: adding it cuts
step-0 error by 13% (Select-DPC base) and 32% (fixed base). See
`.superpowers/sdd/2026-08-16-reacher-residual-rl/task-6-diagnosis.md` -- that
regime is also under-sampled 45:1 by construction, which the feature does NOT
fix; only more episodes do.

Layout at `T_ini = 5` (43 wide):

    [0 : 10]    u_ini    -- 5 rows of (tau_0, tau_1)
    [10 : 35]   y_ini    -- 5 rows of (cos q0, sin q0, q1, tip_x, tip_y)
    [35 : 40]   y_cur    -- (cos q0, sin q0, q1, tip_x, tip_y)
    [40 : 42]   tip - goal
    [42]        buffer validity, min(step_idx, T_ini) / T_ini
    [43 : 43+n] anchor one-hot, only when `anchors` is given
"""
from __future__ import annotations

import numpy as np

from reacher.model import NQ_ARM, config_distance

_Y_WIDTH = 5      # (cos q0, sin q0, q1, tip_x, tip_y) -- 4-D y expands by one


def feature_dim(T_ini: int, n_lib: int = 0) -> int:
    """Feature width, including buffer validity and any anchor one-hot."""
    return T_ini * NQ_ARM + T_ini * _Y_WIDTH + _Y_WIDTH + 2 + 1 + int(n_lib)


def anchor_index(q: np.ndarray, anchors: np.ndarray) -> int:
    """Which library `ReacherDeePC` would select for configuration `q`.

    Mirrors `ReacherDeePC._select_index_for` exactly -- wrapped joint-space
    distance, `argmin`. A test pins the two together, because handing the clone a
    different index than the controller used is worse than handing it none.
    """
    return int(np.argmin(config_distance(np.asarray(q)[:NQ_ARM], anchors)))


def expand_y(y: np.ndarray) -> np.ndarray:
    """`[q0, q1, tip_x, tip_y]` -> `[cos q0, sin q0, q1, tip_x, tip_y]`.

    Accepts `(4,)` or `(T, 4)` and returns `(5,)` or `(T, 5)`. Public because the
    feature layout is worth asserting against in tests: a silent change here
    shifts every downstream block.
    """
    y = np.asarray(y, dtype=np.float64)
    single = y.ndim == 1
    if single:
        y = y[None, :]
    out = np.column_stack([np.cos(y[:, 0]), np.sin(y[:, 0]), y[:, 1:]])
    return out[0] if single else out


def featurize(
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    y_current: np.ndarray,
    goal: np.ndarray,
    step_idx: int,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Build the clone feature vector. See the module docstring for the layout.

    Args:
        u_ini: `(T_ini, 2)` past torques, as APPLIED (post-clip).
        y_ini: `(T_ini, 4)` past outputs `[q; tip]`.
        y_current: `(4,)` current output `[q; tip]`.
        goal: `(2,)` target position.
        anchors: `(n_lib, 2)` library anchors, or None. When given, appends the
            one-hot of the library `ReacherDeePC` would select -- the same
            information the working unicycle clone received. Pass None for
            Select-DPC, which has no library index.
        step_idx: steps taken since `reset`. Becomes the buffer-validity
            feature `min(step_idx, T_ini) / T_ini`: 0 when the buffer is pure
            priming, 1 once it holds only real history.

    Returns:
        `(feature_dim(T_ini),)` float64.

    Raises:
        ValueError: if `u_ini` and `y_ini` disagree on `T_ini`, which would
            otherwise broadcast silently into a corrupt vector.
    """
    u_ini = np.asarray(u_ini, dtype=np.float64)
    y_ini = np.asarray(y_ini, dtype=np.float64)
    y_current = np.asarray(y_current, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    if u_ini.shape[0] != y_ini.shape[0]:
        raise ValueError(
            f"T_ini mismatch: u_ini has {u_ini.shape[0]} rows, "
            f"y_ini has {y_ini.shape[0]}"
        )
    T_ini = u_ini.shape[0]
    blocks = [
        u_ini.ravel(),
        expand_y(y_ini).ravel(),
        expand_y(y_current),
        y_current[NQ_ARM:] - goal,
        np.array([min(int(step_idx), T_ini) / T_ini]),
    ]
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=np.float64)
        onehot = np.zeros(anchors.shape[0])
        onehot[anchor_index(y_current, anchors)] = 1.0
        blocks.append(onehot)
    return np.concatenate(blocks)
