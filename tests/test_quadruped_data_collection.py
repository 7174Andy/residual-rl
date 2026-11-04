import numpy as np

from quadruped import data_collection as dc

KEYS = {"u_0", "y_0", "u_1", "y_1", "sample_bounds", "anchors", "seeds", "retries",
        "n_wraps", "hold_s", "noise", "T"}


def test_anchor_headings_match_unicycle():
    want = (np.array([1, 3, 5, 7]) * np.pi / 4 + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(dc.anchor_headings(4), want)


def test_deterministic():
    a, b = dc.collect_libraries(T=100, seed=3), dc.collect_libraries(T=100, seed=3)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_npz_keys_and_shapes():
    d = dc.collect_libraries(T=100, n_anchors=2, seed=0)
    assert set(d) == KEYS
    for i in range(2):
        assert d[f"u_{i}"].shape == d[f"y_{i}"].shape == (100, 3)
        assert np.all(d[f"u_{i}"] >= dc.SAMPLE_BOUNDS[:, 0]) and np.all(d[f"u_{i}"] <= dc.SAMPLE_BOUNDS[:, 1])


def test_commands_held():
    u, _, fell, _ = dc.collect_trajectory(100, 0.0, np.random.default_rng(0), noise=(0, 0, 0))
    assert not fell
    change = np.flatnonzero(np.any(np.diff(u, axis=0) != 0, axis=1)) + 1
    assert len(change) > 0 and np.all(change % 25 == 0)


def test_fall_retries_next_seed(monkeypatch):
    from quadruped.model import Go1Sim
    n = [0]

    def fell(self):
        n[0] += 1
        return n[0] == 10

    monkeypatch.setattr(Go1Sim, "fell", fell)
    d = dc.collect_libraries(T=50, n_anchors=2, seed=7)
    assert list(d["seeds"]) == [8, 9] and list(d["retries"]) == [1, 0]


def test_unwrap_continuous():
    d = dc.collect_libraries(T=300, n_anchors=2, seed=0)
    for i in range(2):
        assert np.all(np.abs(np.diff(d[f"y_{i}"][:, 2])) < np.pi)
