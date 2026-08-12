"""DeePC.key_fn: a hook for library selection on systems whose keying
quantity is not a single component of `y`.

The unicycle keys on heading, which IS y[2]. The Panda keys on tip azimuth,
atan2(y[1], y[0]) -- a function of y, not a component of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.deepc import DeePC
from core.hankel import build_hankel


def _libs(n_lib: int, m_u: int = 2, p_y: int = 3, T_ini: int = 2, N: int = 3, T: int = 60):
    """n_lib distinct random (Up, Uf, Yp, Yf) tuples of consistent shape."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_lib):
        u = rng.normal(size=(T, m_u)) + i
        y = rng.normal(size=(T, p_y)) + i
        out.append(build_hankel(u, y, T_ini=T_ini, N=N))
    return out


def _deepc(n_lib, anchors, key_fn=None, heading_index=2):
    return DeePC(
        _libs(n_lib), anchor_headings=np.asarray(anchors, dtype=np.float64),
        Q=np.eye(3), R=np.eye(2), T_ini=2, N=3,
        key_fn=key_fn, heading_index=heading_index,
    )


ANCHORS = [-2.356, -0.785, 0.785, 2.356]


def _nearest_anchor_bruteforce(key: float, anchors) -> int:
    """Expected index, derived WITHOUT the modulo expression the code uses.

    Circular distance the long way round: fold |raw difference| into [0, pi] by
    taking min(d, 2*pi - d). Independent of `_select_index`'s
    `(x + pi) % (2*pi) - pi` formulation, so a shared wrap bug cannot hide.
    """
    best, best_d = 0, float("inf")
    for i, a in enumerate(anchors):
        d = abs(float(key) - float(a)) % (2 * np.pi)
        d = min(d, 2 * np.pi - d)
        if d < best_d:
            best, best_d = i, d
    return best


@pytest.mark.parametrize("heading", np.linspace(-np.pi, np.pi, 25))
def test_key_fn_none_reproduces_heading_index_selection(heading):
    """The default path must be byte-for-byte the old behaviour.

    Uses an independent oracle (_nearest_anchor_bruteforce) to avoid masking
    bugs that appear in both the code and the original parametrized test.
    """
    d = _deepc(4, ANCHORS, key_fn=None)
    y = np.array([0.0, 0.0, heading])
    expected_idx = _nearest_anchor_bruteforce(heading, ANCHORS)
    assert d._select_index_for(y) == expected_idx


def test_key_fn_can_read_a_function_of_y_not_a_component():
    """Azimuth keying: the key is atan2(y[1], y[0]), which is no component of y."""
    d = _deepc(4, ANCHORS, key_fn=lambda y: float(np.arctan2(y[1], y[0])))
    for anchor, expected in zip(ANCHORS, range(4)):
        # A tip lying on the anchor's azimuth ray, 0.5 m out.
        y = np.array([0.5 * np.cos(anchor), 0.5 * np.sin(anchor), 99.0])
        assert d._select_index_for(y) == expected
    # y[2] is deliberately absurd; with key_fn set it must be ignored entirely.


def test_key_fn_is_ignored_with_a_single_library():
    d = _deepc(1, [0.0], key_fn=lambda y: 1 / 0)  # would raise if ever called
    assert d._select_index_for(np.array([1.0, 2.0, 3.0])) == 0


def test_heading_index_not_validated_when_key_fn_given():
    """heading_index is meaningless under key_fn, so an out-of-range value must be
    accepted at construction -- and must never be used to index y, since y[99]
    would raise IndexError.

    The key is 0.785, which matches ANCHORS[2] exactly. Do not use 0.0 here: it is
    equidistant from ANCHORS[1] and ANCHORS[2] (both 0.785 away), so the result
    depends on argmin's tie-break and the test would be asserting an arbitrary
    convention rather than the behaviour it cares about.
    """
    d = _deepc(4, ANCHORS, key_fn=lambda y: 0.785, heading_index=99)
    assert d._select_index_for(np.zeros(3)) == 2


def test_heading_index_still_validated_without_key_fn():
    with pytest.raises(ValueError, match="heading_index"):
        _deepc(4, ANCHORS, key_fn=None, heading_index=99)
