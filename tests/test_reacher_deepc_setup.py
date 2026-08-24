"""`reacher/deepc_setup.py`'s anchor grid — now public package API.

`anchor_grid` moved out of `scripts/run_reacher_deepc.py` into the package so
`reacher/clone_data.py` can use it without a package module importing from
`scripts/`. That promotion is what earns it a test: seven scripts and one package
module now depend on it, and it had none.
"""
from __future__ import annotations

import numpy as np

from reacher.deepc_setup import anchor_grid
from reacher.model import NQ_ARM, load_model, safe_box


def test_shape_is_the_product_of_the_grid():
    model, _ = load_model()
    for n0, n1 in ((6, 5), (3, 3), (1, 1)):
        assert anchor_grid(model, n0, n1).shape == (n0 * n1, NQ_ARM)


def test_joint0_excludes_the_duplicate_endpoint():
    """joint0 is periodic: -pi and +pi are the SAME configuration. Including both
    would silently duplicate a whole column of the grid, so every anchor budget
    would buy one column less coverage than it appears to."""
    model, _ = load_model()
    q0 = np.unique(anchor_grid(model, 6, 5)[:, 0])
    assert len(q0) == 6
    assert np.isclose(q0.min(), -np.pi)
    assert q0.max() < np.pi - 1e-9
    # Evenly spaced with no wrap-around collision.
    assert np.allclose(np.diff(q0), 2 * np.pi / 6)


def test_joint1_spans_the_safe_box_inclusively():
    """joint1 is range-limited, not periodic, so its endpoints are real
    configurations and must both be sampled — the fold limit is what lets the arm
    reach targets near the origin."""
    model, _ = load_model()
    lo, hi = safe_box(model)
    q1 = np.unique(anchor_grid(model, 6, 5)[:, 1])
    assert len(q1) == 5
    assert np.isclose(q1.min(), lo[1])
    assert np.isclose(q1.max(), hi[1])


def test_every_anchor_is_inside_the_safe_box():
    model, _ = load_model()
    lo, hi = safe_box(model)
    a = anchor_grid(model, 6, 5)
    # joint0 is unlimited, so only joint1 has a box to respect.
    assert (a[:, 1] >= lo[1] - 1e-12).all() and (a[:, 1] <= hi[1] + 1e-12).all()


def test_is_deterministic():
    model, _ = load_model()
    assert np.array_equal(anchor_grid(model, 6, 5), anchor_grid(model, 6, 5))
