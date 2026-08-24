"""The clone feature vector: width, layout, and the joint0 wrap treatment."""
from __future__ import annotations

import numpy as np
import pytest

from reacher.clone_features import expand_y, feature_dim, featurize


def _inputs(T_ini=5):
    rng = np.random.default_rng(0)
    return (rng.uniform(-1, 1, (T_ini, 2)),
            rng.uniform(-1, 1, (T_ini, 4)),
            rng.uniform(-1, 1, 4),
            rng.uniform(-0.2, 0.2, 2))


def test_width_is_43_at_T_ini_5():
    u, y, yc, g = _inputs()
    assert featurize(u, y, yc, g, 0).shape == (43,)
    assert feature_dim(5) == 43


def test_width_scales_with_T_ini():
    u, y, yc, g = _inputs(T_ini=3)
    assert featurize(u, y, yc, g, 0).shape == (feature_dim(3),)


def test_joint0_enters_as_cos_sin_so_the_wrap_is_continuous():
    """joint0 is unlimited. -pi and +pi are the same configuration; a raw angle
    would hand the clone a discontinuity exactly there."""
    u, y, yc, g = _inputs()
    y_lo, y_hi = y.copy(), y.copy()
    y_lo[:, 0] = -np.pi + 1e-9
    y_hi[:, 0] = np.pi - 1e-9
    yc_lo, yc_hi = yc.copy(), yc.copy()
    yc_lo[0], yc_hi[0] = -np.pi + 1e-9, np.pi - 1e-9
    assert np.allclose(featurize(u, y_lo, yc_lo, g, 0),
                       featurize(u, y_hi, yc_hi, g, 0), atol=1e-6)


def test_relative_goal_occupies_block_40_42():
    """Only the goal is relative. The fingertip stays ABSOLUTE in the buffer
    blocks on purpose: the arm is base-anchored, so tip position is the forward
    kinematics of `q` and carries real state. A fully translation-invariant
    encoding would delete that.

    Indexed absolutely, not from the end: the validity feature was appended after
    this block and negative slicing silently pointed at the wrong thing.
    """
    u, y, yc, g = _inputs()
    feat = featurize(u, y, yc, g, 0)
    assert np.allclose(feat[40:42], yc[2:] - g)
    # And the absolute tip really is preserved in the y_cur block, so the two
    # encodings coexist rather than one overwriting the other.
    assert np.allclose(feat[35:40], expand_y(yc))
    assert np.allclose(feat[38:40], yc[2:])


def test_moving_only_the_goal_changes_only_the_goal_block():
    """The converse of the above: the goal must not leak into any other block,
    and must not disturb the validity feature either."""
    u, y, yc, g = _inputs()
    a = featurize(u, y, yc, g, 0)
    b = featurize(u, y, yc, g + np.array([0.05, -0.03]), 0)
    assert np.allclose(a[:40], b[:40])
    assert not np.allclose(a[40:42], b[40:42])
    assert a[42] == b[42]


def test_layout_blocks_are_where_the_docstring_says():
    """All four documented blocks, so the name is not wider than the body. A
    silent width change anywhere shifts every block after it."""
    u, y, yc, g = _inputs()
    feat = featurize(u, y, yc, g, 0)
    assert np.allclose(feat[0:10], u.ravel())            # u_ini
    assert np.allclose(feat[10:35], expand_y(y).ravel())  # y_ini
    assert np.allclose(feat[35:40], expand_y(yc))        # y_cur
    assert np.allclose(feat[40:42], yc[2:] - g)          # tip - goal
    assert feat[42] == 0.0                                # validity, step 0
    assert feat.shape == (43,)                            # nothing past 43


def test_mismatched_buffer_lengths_raise():
    u, y, yc, g = _inputs()
    with pytest.raises(ValueError, match="T_ini"):
        featurize(u[:3], y, yc, g, 0)


def test_buffer_validity_ramps_from_zero_to_one_over_T_ini():
    """0 while the buffer is pure priming, 1 once it holds only real history, and
    saturated after. Without this the clone cannot tell a part-primed buffer from
    real history that happens to look similar -- measured worth 13-32% of step-0
    error."""
    u, y, yc, g = _inputs()
    assert featurize(u, y, yc, g, 0)[-1] == 0.0
    assert featurize(u, y, yc, g, 1)[-1] == pytest.approx(0.2)
    assert featurize(u, y, yc, g, 5)[-1] == 1.0
    assert featurize(u, y, yc, g, 49)[-1] == 1.0   # saturates, never exceeds 1


def test_validity_is_the_only_thing_step_idx_changes():
    u, y, yc, g = _inputs()
    a, b = featurize(u, y, yc, g, 0), featurize(u, y, yc, g, 3)
    assert np.allclose(a[:-1], b[:-1])
    assert a[-1] != b[-1]


def test_anchor_index_matches_the_controller_exactly():
    """The clone must be told the SAME library the controller actually used.

    A wrong index is worse than no index: the network would learn to associate a
    mode's actions with a different mode's flag. The unicycle pins
    `select_library_index` against `DeePC._select_index` for this reason; this is
    the Reacher counterpart.
    """
    import numpy as _np

    from reacher.clone_data import build_bank, build_fixed_controller
    from reacher.clone_features import anchor_index
    from reacher.model import load_model

    model, data = load_model()
    _bank, payload = build_bank(model, data, _np.random.default_rng(0),
                                grid=(3, 3), T=200, stride=4)
    anchors = payload["anchors"]
    ctrl = build_fixed_controller(payload)

    rng = _np.random.default_rng(1)
    for _ in range(40):
        q = rng.uniform(-np.pi, np.pi, 2)
        y = np.concatenate([q, rng.uniform(-0.2, 0.2, 2)])   # y = [q; tip]
        assert anchor_index(y, anchors) == ctrl._select_index_for(y), (
            f"clone would report a different library than the controller uses "
            f"at q={q}")


def test_one_hot_is_absent_without_anchors_and_present_with_them():
    u, y, yc, g = _inputs()
    anchors = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
    assert featurize(u, y, yc, g, 0).shape == (43,)
    f = featurize(u, y, yc, g, 0, anchors)
    assert f.shape == (46,)
    assert f[43:].sum() == 1.0 and set(f[43:]) <= {0.0, 1.0}
    assert feature_dim(5, n_lib=3) == 46
