"""The behavioral clone's feature vector for PandaReach.

The Panda analogue of `reacher/clone_features.py`, with two differences, both
consequences of the robot rather than of taste:

* **No angle expansion.** Reacher's joint0 is unlimited and wraps, so it enters
  as `(cos, sin)` to avoid handing the network a discontinuity at +-pi. All
  seven Panda joints are range-limited -- `panda.model.safe_box` is built from
  those limits, and the widest span is 4.64 rad -- so none of them wraps and raw
  radians are correct. Expanding them here would widen the vector for nothing.
* **No library one-hot.** The unicycle clone was told which of four libraries
  DeePC had selected, and that was load-bearing: `select_library_index` is an
  argmin, so crossing a boundary makes the output jump. Select-DPC re-selects
  300 of ~180,000 columns every step, so there is nothing discrete to hand over
  and `n_lib = 0` everywhere.

`q` is in the vector and is MANDATORY, not stylistic. A 7-DoF arm has a 4-D
self-motion manifold: configurations agreeing on tip position to <1 mm sit a
median 3.59 rad apart in `q` and respond differently to the same input
(CLAUDE.md). A tip-only feature vector would therefore map one-to-many onto the
expert's label, and no feedforward network can fit a target its inputs do not
determine.

`u` is the expert's own input, an ABSOLUTE `q_des` joint target -- not the env's
delta action -- recorded POST-clip, i.e. the value the plant actually received.

Layout at `T_ini = 5` (99 wide):

    [0  : 35]   u_ini    -- 5 rows of applied ctrl (absolute q_des), 7 wide
    [35 : 85]   y_ini    -- 5 rows of y = [q (7); tip (3)]
    [85 : 95]   y_cur    -- [q (7); tip (3)]
    [95 : 98]   tip - goal
    [98]        buffer validity, min(step_idx, T_ini) / T_ini

Two conventions, both inherited from the working pipelines:

* **The goal enters relative, the state stays absolute.** `tip - goal` matches
  the env observation's rationale -- the controller's job depends on the
  displacement, not on where in the workspace the pair sits. The tip and `q`
  themselves stay ABSOLUTE, deliberately: the arm is anchored at the origin, so
  an absolute tip is the forward kinematics of `q` and is real state, and the
  local dynamics differ between an extended and a folded arm at the same
  relative goal. Making the whole vector translation-invariant would delete
  information the clone needs.
* **Buffer validity is not cosmetic.** At `t = 0` there is no history, so
  collection primes the buffer with `u_ini = tile(q0)` and `y_ini = tile(y0)`.
  The first `T_ini` steps are a structurally different regime, and without this
  feature the clone cannot tell a part-primed buffer at step 3 from real history
  that happens to look similar. Measured on Reacher: it cuts step-0 error by 13%
  on the Select-DPC base. It does NOT fix that regime being under-sampled ~30:1
  by construction (5 primed rows in a 150-step episode); only more episodes do.
"""
from __future__ import annotations

import numpy as np

NQ = 7          # Panda joints
Y_WIDTH = 10    # y = [q (7); tip (3)] -- part of the layout contract


def feature_dim(T_ini: int) -> int:
    """Feature width, including the buffer-validity scalar. 99 at T_ini = 5."""
    return T_ini * NQ + T_ini * Y_WIDTH + Y_WIDTH + 3 + 1


def q_slice(T_ini: int) -> slice:
    """Where `q` sits inside the feature vector -- `[85:92]` at `T_ini = 5`.

    `y_cur` follows the two buffers, and `q` is its first `NQ` entries. A delta
    target (`q_des - q`) is recoverable from the features alone because of
    this, so switching the clone between absolute and delta regression needs no
    re-collection.
    """
    start = T_ini * (NQ + Y_WIDTH)
    return slice(start, start + NQ)


def featurize(u_ini: np.ndarray, y_ini: np.ndarray, y_current: np.ndarray,
              goal: np.ndarray, step_idx: int) -> np.ndarray:
    """Build the clone feature vector. See the module docstring for the layout.

    Args:
        u_ini: `(T_ini, 7)` past inputs, as APPLIED (absolute q_des, post-clip).
        y_ini: `(T_ini, 10)` past outputs `[q; tip]`.
        y_current: `(10,)` current output `[q; tip]`.
        goal: `(3,)` target tip position.
        step_idx: steps since reset; becomes `min(step_idx, T_ini) / T_ini`.

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
    return np.concatenate([
        u_ini.ravel(),
        y_ini.ravel(),
        y_current,
        y_current[NQ:] - goal,
        np.array([min(int(step_idx), T_ini) / T_ini]),
    ])
