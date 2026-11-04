import os

import numpy as np
import pytest

from core.deepc import DeePC
from core.hankel import build_hankel
from two_wheel_robot.rl.deepc_setup import (
    DEFAULT_LIBRARIES, bearing_y_ref, build_canonical_deepc,
)

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_LIBRARIES),
                                reason="data/libraries_v0.npz absent")

# From pre-change code: build_canonical_deepc, y=(0,0,.5), goal (3,4), midpoint u_init.
GOLDEN = np.array([12.2853702, 1.57079633])
Y = np.array([0.0, 0.0, 0.5])
GOAL = np.array([3.0, 4.0])


def test_default_is_bit_identical():
    d, info = build_canonical_deepc()
    d.reset(Y, info["u_init_midpoint"])
    np.testing.assert_allclose(d.act(Y, bearing_y_ref(Y, GOAL)), GOLDEN, atol=1e-8)


def test_affine_sums_to_one():
    d0, info = build_canonical_deepc()
    with np.load(DEFAULT_LIBRARIES) as data:
        libs = [build_hankel(data[f"u_{i}"], data[f"y_{i}"], T_ini=5, N=12) for i in range(4)]
    d = DeePC(libs, anchor_headings=info["anchors"], Q=info["Q"], R=info["R"], T_ini=5, N=12,
              lambda_g=2.0, lambda_y=3e6, affine=True,
              u_bounds=(info["action_bounds"][:, 0], info["action_bounds"][:, 1]))
    d.reset(Y, info["u_init_midpoint"])
    d.act(Y, bearing_y_ref(Y, GOAL))
    assert abs(d._g.value.sum() - 1.0) < 1e-3
