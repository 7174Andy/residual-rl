"""The canonical DeePC configuration for PandaReach-v0.

Centralized here for the same reason `two_wheel_robot/rl/deepc_setup.py` exists:
the clone dataset, the fidelity gate and the residual env must all be driven by
one configuration that cannot drift.

Keying is on **tip azimuth**, `atan2(y[1], y[0])` -- the structural analog of the
unicycle's heading key. Read the spec's "What azimuth does and does not capture"
before assuming it is the right axis: the unicycle's nonlinearity is in its
dynamics, so heading IS the nonlinearity, whereas the Panda's is in the output
map y = FK(q). Rotating q1 rotates the Jacobian but leaves its singular values
alone; extension changes its conditioning. Azimuth captures the former only.
"""
from __future__ import annotations

import numpy as np

from core.deepc import DeePC
from core.hankel import build_hankel
from panda.data_collection import LIBRARIES_PATH

# Ratio-matched to the unicycle's control/state cost balance (its Q*dbar^2 is
# 115.4 against the Panda's 0.640, a factor 5.55e-3), then swept. The paper's
# lambda_y = 3e6 is ~400x too strong against a metre-scale slack and measures
# 3.4x slower.
LAMBDA_G_DEFAULT = 5e-3
LAMBDA_Y_DEFAULT = 7.5e3


def azimuth_key(y: np.ndarray) -> float:
    """Tip azimuth about the base's vertical axis."""
    return float(np.arctan2(y[1], y[0]))


class DeePCPolicy:
    """Adapts DeePC to `eval.run_scenarios`' `policy(env) -> u` contract.

    Exposes `last_library_idx` so the harness can record library usage, which is
    what shows whether switching ever actually triggered -- the unicycle's
    library-switching writeup flags a no-op regime where episodes never cross a
    quadrant boundary.
    """

    def __init__(self, deepc: DeePC, info: dict):
        self.deepc = deepc
        self.info = info
        self._primed_for = None
        self.last_library_idx = -1
        # Older info dicts (built before the extended output existed) carry
        # neither key; those are tip-only by definition.
        self._y_attr = info.get("y_attr", "y")
        self._y_ref_attr = info.get("y_ref_attr", "y_ref")

    def __call__(self, env) -> np.ndarray:
        y = getattr(env, self._y_attr)
        if self._primed_for is not env or env.step_idx == 0:
            self.deepc.reset(y, u_initial=self.info["u_init"])
            self._primed_for = env
        u = self.deepc.act(y, getattr(env, self._y_ref_attr))
        self.last_library_idx = int(self.deepc.last_library_idx)
        return u


def build_canonical_panda_deepc(
    libraries_path: str = LIBRARIES_PATH,
    T_ini: int = 5,
    N: int = 12,
    lambda_g: float = LAMBDA_G_DEFAULT,
    lambda_y: float = LAMBDA_Y_DEFAULT,
    output: str = "tip",
) -> tuple[DeePC, dict]:
    """Construct the controller plus the info needed to run it.

    `output="tip"` identifies on `y = tip position` (3-D) -- the original
    configuration, and the default so existing results stay reproducible.

    `output="ext"` identifies on `y_ext = (tip, q_normalized)` (10-D), which is
    what `PandaReachEnv.y_ext`'s docstring argues for: tip position alone does not
    observe the state, so the past window maps one-to-many onto futures. Requires a
    libraries file carrying `yext_i` (v1 or later).

    Q is `diag(I_3, 0_7)` under `"ext"`, so the tracking cost is NUMERICALLY
    IDENTICAL to the tip-only case -- the extra outputs inform prediction via the
    `Yp`/`Yf` constraints without changing what is being optimized. `info["output"]`
    records which mode was built, and `info["y_attr"]`/`info["y_ref_attr"]` name the
    env properties a runner must read.
    """
    if output not in ("tip", "ext"):
        raise ValueError(f"output must be 'tip' or 'ext'; got {output!r}")
    key = "y" if output == "tip" else "yext"
    with np.load(libraries_path) as z:
        n_lib = int(z["anchor_q1"].shape[0])
        if f"{key}_0" not in z:
            raise KeyError(
                f"{libraries_path} has no '{key}_0'; it predates output={output!r}. "
                "Re-collect with scripts/collect_panda_data.py to get 'yext_i'."
            )
        uy = [(z[f"u_{i}"], z[f"{key}_{i}"]) for i in range(n_lib)]
        anchors = np.asarray(z["anchor_azimuths"], dtype=np.float64)
        delta_max = float(z["delta_max"])

    p_y = uy[0][1].shape[1]
    # Zero weight on everything past the tip block: identical cost, better model.
    Q = np.zeros((p_y, p_y))
    Q[:3, :3] = np.eye(3)
    R = 1.0e-2 * np.eye(7)
    u_bounds = (-delta_max * np.ones(7), delta_max * np.ones(7))

    libraries = [build_hankel(u, y, T_ini=T_ini, N=N) for (u, y) in uy]
    deepc = DeePC(
        libraries, anchor_headings=anchors, Q=Q, R=R, T_ini=T_ini, N=N,
        lambda_g=lambda_g, lambda_y=lambda_y, u_bounds=u_bounds,
        key_fn=azimuth_key, solver="SCS",
    )
    info = {
        "anchors": anchors,
        "Q": Q, "R": R,
        "T_ini": T_ini, "N": N,
        "lambda_g": lambda_g, "lambda_y": lambda_y,
        "delta_max": delta_max,
        "output": output,
        "p_y": p_y,
        # Which env properties a runner must read. DeePCPolicy consults these
        # rather than hardcoding `env.y`, so one policy class serves both modes
        # and cannot be paired with the wrong output by accident.
        "y_attr": "y" if output == "tip" else "y_ext",
        "y_ref_attr": "y_ref" if output == "tip" else "y_ref_ext",
        # u = 0 means "hold" under the delta interface, so a zero past-input
        # buffer is the natural synthetic past -- unlike the unicycle, whose
        # action space excludes zero and needs a midpoint.
        "u_init": np.zeros(7, dtype=np.float64),
    }
    return deepc, info
