"""Select-DPC on Reacher-v5: the `u = tau`, `y = [q; fingertip]` adapter.

The algorithm lives in `core/selectdpc.py` (system-agnostic, faithful to
Algorithm 1 + 2 of arXiv:2503.18845). This module only assembles Reacher's
`(u, y)` trajectories from a collection payload.

Reacher needs none of the Panda's corrections: its torque input is natively
bounded to `[-1, 1]`, so no rate limit is required, and `tau` stacks torque
(dimensionless), radians and metres whose numeric scales are close enough that the
paper's plain norm is reasonable as-is.
"""
from __future__ import annotations

from core.selectdpc import SelectDPC, select_predict
from core.selectdpc import trajectory_bank as _bank
from reacher.deepc_setup import outputs

__all__ = ["SelectDPC", "select_predict", "trajectory_bank"]


def trajectory_bank(payload: dict, T_ini: int, N: int, stride: int = 1) -> dict:
    """Pool a `reacher/deepc_setup.py` collection payload into a Select-DPC bank."""
    n = int(payload["anchors"].shape[0])
    u_list = [payload[f"u_{i}"] for i in range(n)]
    y_list = [outputs(payload[f"q_{i}"], payload[f"tip_{i}"]) for i in range(n)]
    return _bank(u_list, y_list, T_ini, N, stride=stride)
