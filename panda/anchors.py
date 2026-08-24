"""Task-relevant anchor selection for the `u = q_des` DeePC controller.

Implements sections 2-5, 7 and 10 of the anchor-selection plan: sample the goals
the task actually needs, get the configuration that reaches each one, cluster
those configurations, and key libraries on joint-space distance to the medoids.

Two deliberate deviations from the plan as written, both documented at the call
site that makes them:

1. **The IK stage is a choice, not a formality** (`sample_task_configs(ik=...)`).
   The plan samples Cartesian goals then solves IK back to `q`. Skipping it and
   keeping the FK-sampled configuration (`ik=None`) was tried first and is a trap:
   IK is one-to-many, so a *deterministic* IK policy collapses a 3-D goal space
   onto a ~3-D manifold, whereas FK from uniform configurations stays uniform.
   Measured with `ik=None`, `Q` matches a bare uniform safe-box draw to 2%
   (effective dimensionality 5.71/7, silhouette 0.114) -- k-medoids was clustering
   a box. Goals are drawn identically in every mode, so `ik` is the only variable.

2. **Clustering is on goal configurations, not visited ones.** An episode passes
   through configurations between start and goal, and those are not in `Q_task`.
   The plan specifies goal configurations, so that is what this does -- but a
   region that only ever appears mid-episode will not get an anchor, and that
   shows up as prediction error in `validate_anchors.py` rather than as a
   coverage warning here.

`core/` is untouched: `AnchorDeePC` (in `panda/qdes.py`) subclasses `DeePC` and
overrides library selection, because the plan's `y = [q; d_g]` puts `q` *in the
output*, so the keying quantity needs no `key_fn` at all.
"""
from __future__ import annotations

import mujoco
import numpy as np

from panda.model import MIN_TIP_Z, safe_box, sample_config, tip_id


def ik_solve(
    model, data, goal: np.ndarray, q_seed: np.ndarray, lo, hi, tip: int,
    iters: int = 200, damping: float = 0.05, tol: float = 1e-3,
    max_step: float = 0.2,
) -> np.ndarray | None:
    """Damped-least-squares IK: a configuration whose FK tip reaches `goal`.

    `dq = J^T (J J^T + damping^2 I)^-1 e`. Damping keeps the step finite through
    the rank-deficient configurations a 7-DoF arm passes near; the plain
    pseudo-inverse blows up there. `max_step` caps |dq| so a large initial error
    cannot throw the iterate across the workspace before the linearization it is
    built on stops holding.

    Returns `None` if it fails to converge, leaves the safe box, self-collides, or
    ends below the floor -- callers must handle rejection, unlike `sample_config`,
    which is rejection-sampled and always succeeds.
    """
    q = np.clip(np.asarray(q_seed, dtype=np.float64), lo, hi)
    goal = np.asarray(goal, dtype=np.float64)
    jacp = np.zeros((3, model.nv))
    for _ in range(iters):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        err = goal - data.site_xpos[tip]
        if np.linalg.norm(err) < tol:
            # Converged -- but the pose must also be one the robot can adopt.
            if data.ncon or data.site_xpos[tip][2] <= MIN_TIP_Z:
                return None
            return q
        mujoco.mj_jacSite(model, data, jacp, None, tip)
        dq = jacp.T @ np.linalg.solve(
            jacp @ jacp.T + damping**2 * np.eye(3), err
        )
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm
        q = np.clip(q + dq, lo, hi)
    return None


def sample_task_configs(
    model, data, rng: np.random.Generator, n: int, ik: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`n` task-relevant `(configuration, goal)` pairs. Plan sections 2 + 3.

    Returns `(Q, P)` with shapes `(n, nq)` and `(n, 3)`.

    Goals are ALWAYS drawn the way `PandaReachEnv` draws them -- FK of a
    rejection-sampled valid configuration -- so the goal distribution is identical
    across every `ik` setting and the only variable is which configuration is
    chosen to reach each goal.

    `ik` selects that choice, and it is a real design decision rather than a
    solver detail: it determines which slice of the 7-D configuration space the
    robot is asked to live in.

    * `None`   -- keep the configuration the goal was generated from. Cheap and
      exact, but the configurations inherit the uniform safe-box sampling, so
      `Q` has no structure to cluster (measured: matches a bare uniform draw to
      2%, effective dimensionality 5.71/7).
    * `"home"` -- solve IK from the `home` keyframe every time. A deterministic
      map from a 3-D goal space, so the image is a ~3-D manifold. This is the
      plan's section 3.
    * `"previous"` -- seed from the last solution, giving a continuous sheet
      rather than a star around `home`. Order-dependent by construction.
    * `"random"` -- seed from a fresh random valid configuration. The NULL
      CONTROL: IK with a random seed re-randomizes the choice, so this should
      reproduce the `None` result. If it does not, the concentration being
      measured is an artifact of the solver rather than of the policy.
    """
    lo, hi = safe_box(model)
    tip = tip_id(model)
    if ik not in (None, "home", "previous", "random"):
        raise ValueError(f"ik must be None, 'home', 'previous' or 'random'; got {ik!r}")

    Q = np.empty((n, model.nq))
    P = np.empty((n, 3))
    home = np.asarray(model.key_qpos[0], dtype=np.float64)
    prev = home.copy()
    i = attempts = 0
    while i < n:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError(
                f"IK ({ik!r}) converged for only {i}/{n} goals in {attempts} "
                "attempts; loosen --ik-tol or use --ik none"
            )
        q_fk, goal = sample_config(model, data, rng, lo, hi, tip)
        if ik is None:
            q = q_fk
        else:
            if ik == "home":
                seed = home
            elif ik == "previous":
                seed = prev
            else:
                seed = sample_config(model, data, rng, lo, hi, tip)[0]
            q = ik_solve(model, data, goal, seed, lo, hi, tip)
            if q is None:
                continue
            prev = q
        Q[i], P[i] = q, goal
        i += 1
    return Q, P


def _pairwise(Q: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """Weighted joint-space distance matrix, plan section 7's `d_i(q)`."""
    X = Q if w is None else Q * np.sqrt(w)
    d2 = np.maximum(
        (X**2).sum(1)[:, None] + (X**2).sum(1)[None, :] - 2 * X @ X.T, 0.0
    )
    return np.sqrt(d2)


def kmedoids(
    D: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """k-medoids by Voronoi iteration with k-means++ seeding on `D`.

    Medoids over means because the plan wants each anchor to be a configuration
    the robot can actually adopt -- a mean of valid configurations need not be
    valid (it can self-collide or drop below the floor), and data collection has
    to start *at* the anchor.

    Returns `(medoid_indices, labels, cost)` for the best of `n_init` restarts.
    """
    n = D.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}]; got {k}")
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        med = np.empty(k, dtype=int)
        med[0] = rng.integers(n)
        for j in range(1, k):                       # k-means++ seeding
            d2 = D[:, med[:j]].min(axis=1) ** 2
            total = d2.sum()
            med[j] = rng.integers(n) if total <= 0 else rng.choice(n, p=d2 / total)
        for _ in range(max_iter):
            labels = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for j in range(k):
                members = np.flatnonzero(labels == j)
                if members.size:                     # keep the medoid if empty
                    new[j] = members[np.argmin(D[np.ix_(members, members)].sum(1))]
            if np.array_equal(np.sort(new), np.sort(med)):
                break
            med = new
        labels = np.argmin(D[:, med], axis=1)
        cost = float(D[np.arange(n), med[labels]].sum())
        if best is None or cost < best[2]:
            best = (med.copy(), labels, cost)
    assert best is not None
    return best


def farthest_point(
    Q: np.ndarray, k: int, w: np.ndarray | None = None, start: int | None = None,
) -> dict:
    """Farthest-point sampling (greedy k-center). Returns a NESTED anchor sequence.

    `a_new = argmax_q min_i ||q - a_i||` -- repeatedly take the configuration
    worst served by the anchors chosen so far.

    This optimizes a different objective from `kmedoids`, and the difference is
    the point: k-medoids minimizes the MEAN distance to a medoid, which lets it
    ignore a sparse outlying region entirely; k-center minimizes the WORST
    distance, which is what decides whether any configuration is left outside
    every library's valid radius. Greedy FPS is the standard 2-approximation.

    The sequence is nested -- the first `j` indices are a valid `j`-anchor set for
    every `j <= k` -- so one pass yields the whole `r_K` curve. Because the
    running `d` vector *is* the nearest-anchor distance, `r_K = d.max()` and the
    mean come free at each step.

    Returns `idx` `(k,)`, plus `r` and `mean_nn` `(k,)` giving worst-case and mean
    nearest-anchor distance when the first `j+1` anchors are used.
    """
    X = Q if w is None else Q * np.sqrt(w)
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}]; got {k}")
    if start is None:
        # Most central point, so the sequence is deterministic and the first
        # anchor is a sensible single-anchor answer rather than an outlier.
        start = int(np.argmin(((X - X.mean(axis=0)) ** 2).sum(axis=1)))

    idx = [int(start)]
    d = np.linalg.norm(X - X[start], axis=1)
    r = [float(d.max())]
    mean_nn = [float(d.mean())]
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
        r.append(float(d.max()))
        mean_nn.append(float(d.mean()))
    return {"idx": np.array(idx), "r": np.array(r), "mean_nn": np.array(mean_nn)}


def coverage(Q: np.ndarray, anchors: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Worst-case and mean nearest-anchor distance for an ARBITRARY anchor set.

    Lets a k-medoids set and an FPS set be scored on the same k-center metric.
    """
    X = Q if w is None else Q * np.sqrt(w)
    A = anchors if w is None else anchors * np.sqrt(w)
    d = np.min(np.linalg.norm(X[:, None, :] - A[None, :, :], axis=2), axis=1)
    return {"r": float(d.max()), "mean_nn": float(d.mean()),
            "p95": float(np.quantile(d, 0.95))}


def assign(q: np.ndarray, anchors: np.ndarray, w: np.ndarray | None = None) -> int:
    """Nearest anchor by weighted joint-space distance. Plan section 7."""
    d = anchors - np.asarray(q, dtype=np.float64)
    if w is not None:
        d = d * np.sqrt(w)
    return int(np.argmin((d**2).sum(1)))


def select_anchors(
    Q: np.ndarray, k: int, rng: np.random.Generator,
    w: np.ndarray | None = None, n_init: int = 10,
) -> dict:
    """Cluster `Q` and return the anchors plus the diagnostics sections 4-5 want."""
    D = _pairwise(Q, w)
    med, labels, cost = kmedoids(D, k, rng, n_init=n_init)
    anchors = Q[med].copy()
    within = np.array([
        float(D[np.arange(len(Q)), med[labels]][labels == j].max(initial=0.0))
        for j in range(k)
    ])
    return {
        "anchors": anchors,
        "medoid_idx": med,
        "labels": labels,
        "cost": cost,
        "counts": np.bincount(labels, minlength=k),
        "radius": within,           # max member distance, i.e. the region's reach
        "diversity": diversity(anchors),
        "silhouette": silhouette(D, labels),
        "pca": pca(Q, w),
    }


def pca(Q: np.ndarray, w: np.ndarray | None = None) -> dict:
    """PCA of the configurations, in the SAME metric k-medoids clustered in.

    Deliberately NOT standardized per joint. k-medoids ran on raw radians (or on
    `sqrt(w)`-scaled radians), so standardizing here would rotate the data into
    axes the clustering never saw and the picture would not explain the labels it
    is drawn with. Pass the same `w` that produced the clusters.

    Returns `scores` (all 7 PCs), `explained` (variance ratio per PC), and
    `project`, which maps new configurations -- the anchors -- into the same axes.
    """
    X = Q if w is None else Q * np.sqrt(w)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    return {
        "scores": Xc @ Vt.T,
        "explained": var / var.sum(),
        "components": Vt,
        "project": lambda A: ((A if w is None else A * np.sqrt(w)) - mean) @ Vt.T,
    }


def silhouette(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over all points, from a precomputed distance matrix.

    Read this BEFORE reading a 2-D PCA scatter of the clusters. A projection can
    make well-separated clusters look merged whenever the leading two components
    carry little of the variance, so an apparent overlap in the picture is not
    evidence of a bad clustering. The silhouette is computed in the full 7-D
    metric and has no such failure mode: ~1 means tight and well separated, ~0
    means clusters touch, <0 means points sit closer to a neighbouring medoid
    than their own.
    """
    n = len(labels)
    k = int(labels.max()) + 1
    if k < 2:
        return 0.0
    a = np.zeros(n)
    b = np.full(n, np.inf)
    for j in range(k):
        m = labels == j
        if not m.any():
            continue
        col = D[:, m]
        own = labels == j
        size = int(m.sum())
        a[own] = col[own].sum(axis=1) / max(size - 1, 1)
        other = ~own
        b[other] = np.minimum(b[other], col[other].mean(axis=1))
    denom = np.maximum(a, b)
    return float(np.mean(np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1), 0.0)))


def diversity(anchors: np.ndarray) -> dict:
    """Plan section 5: are these genuinely different arm shapes, or one line?

    The plan's failure case is a set differing in only joint 1 -- the same arm
    shape rotated about the base. `effective_joints` is the participation ratio
    of the per-joint variance vector, `(sum v)^2 / sum(v^2)`: it reads ~1 when a
    single joint carries all the spread and ~nq when all joints contribute
    equally, so it detects that failure without a hand-tuned threshold.

    Joint 1 is called out separately because it is the one that provably does
    NOT change the dynamics -- the arm's joint-space dynamics are exactly
    invariant to it (rotation about the gravity axis), so spread concentrated
    there is spread that buys no model fidelity.
    """
    if len(anchors) < 2:
        return {"per_joint_std": np.zeros(anchors.shape[1]), "effective_joints": 0.0,
                "q1_variance_share": 0.0}
    v = anchors.var(axis=0)
    total = v.sum()
    return {
        "per_joint_std": np.sqrt(v),
        "effective_joints": float(total**2 / (v**2).sum()) if total > 0 else 0.0,
        "q1_variance_share": float(v[0] / total) if total > 0 else 0.0,
    }


def split_region(
    Q: np.ndarray, labels: np.ndarray, anchors: np.ndarray, region: int,
    rng: np.random.Generator, w: np.ndarray | None = None, n_init: int = 10,
) -> np.ndarray:
    """Plan section 10: split ONE region in two, leaving the others alone.

    Returns the new anchor set (length `len(anchors) + 1`). Re-clustering the
    whole set at `k+1` would move every anchor and invalidate every library
    already collected; this replaces exactly one and keeps the rest reusable.
    """
    members = np.flatnonzero(labels == region)
    if members.size < 2:
        raise ValueError(f"region {region} has {members.size} member(s); cannot split")
    sub = select_anchors(Q[members], 2, rng, w=w, n_init=n_init)
    keep = [a for j, a in enumerate(anchors) if j != region]
    return np.vstack([np.array(keep).reshape(-1, anchors.shape[1]), sub["anchors"]])
