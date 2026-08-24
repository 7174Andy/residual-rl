"""Clone-data collection: drive Select-DPC, record (features, applied action).

This is the ONLY place in the pipeline where a QP runs. Select-DPC costs 76.7
ms/step (`scripts/sweep_select_dpc.py`), so putting it in an RL loop would mean
~4.3 hours of solver time per 200k training steps -- the obstacle journey 08
named on the unicycle and removed the same way.

Two collection conventions, both inherited rather than invented:

* **`y_t` is recorded before `u_t` is applied**, so `y_{t+1}` is the response to
  `u_t`. Every collection in this repo uses this alignment, and the residual env
  slides its buffer the same way so the clone is deployed under exactly the
  labelling it was trained on.
* **Episodes run the full horizon, never stopping at first reach.** The dataset
  must contain the station-keeping regime, because that is what the residual is
  being asked to improve. Stopping early would remove those rows entirely.
"""
from __future__ import annotations

import numpy as np

from core.selectdpc import SelectDPC
from reacher.clone_features import featurize
from reacher.deepc_setup import (
    R_DEFAULT, TIP_WEIGHT, anchor_grid, collect_anchor, y_ref_for,
)
from reacher.model import NQ_ARM, load_model
from reacher.selectdpc import trajectory_bank


def build_bank(model, data, rng, grid=(6, 5), T: int = 1200, T_ini: int = 5,
               N: int = 12, stride: int = 2):
    """Collect the anchor grid and pool it into a Select-DPC bank.

    Returns `(bank, payload)`. The payload is kept so a caller can rebuild fixed
    libraries from the identical data -- the fallback base controller if the
    clone-fidelity gate fails (spec R1).
    """
    anchors = anchor_grid(model, *grid)
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
    return trajectory_bank(payload, T_ini, N, stride=stride), payload


def build_select_controller(bank: dict, T_ini: int = 5, N: int = 12,
                            n_cols: int = 300, n_max: int = 1,
                            lambda_g: float = 5e-3, lambda_y: float = 7.5e3,
                            carry_prediction: bool = True):
    """Select-DPC wired for Reacher, at journey 12's settings.

    `n_max = 1`: the sweep found the entire gain over fixed anchors comes from
    selecting the right data, not from Algorithm 1's loop, and `n_max = 3` costs
    3x for no reach-rate gain.

    Reacher needs none of the Panda's corrections -- its torque input is natively
    bounded so no rate limit is required, and `tau`'s blocks are numerically
    comparable so the paper's plain norm is reasonable as-is.
    """
    return SelectDPC(
        bank, anchor_headings=np.zeros(1),
        Q=np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2),
        R=R_DEFAULT * np.eye(NQ_ARM),
        T_ini=T_ini, N=N, lambda_g=lambda_g, lambda_y=lambda_y,
        u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), solver="SCS",
        n_cols=n_cols, n_max=n_max, carry_prediction=carry_prediction,
    )


def build_fixed_controller(payload: dict, T_ini: int = 5, N: int = 12,
                           lambda_g: float = 5e-3, lambda_y: float = 7.5e3):
    """The 30-anchor fixed-library controller -- spec R1's fallback base.

    Worse as a controller than Select-DPC (journey 12: 84/120 against 96/120) and
    ~4x more compute per step, but its control law is SMOOTH IN TIME: the library
    changes only when the nearest anchor changes, which is rare, where Select-DPC
    re-selects 300 of ~17,760 columns on every single step.

    That smoothness was the motivation for trying this base, and the attempt
    FAILED: it scored a 15.0-point gate drop against Select-DPC's 12.5, i.e. worse
    despite the smoother law. The discontinuity measurements that motivated it
    (error 3.8x higher on churning steps, `corr(error, action jump) = +0.73`) are
    real but were RETRACTED as the explanation -- removing the churn made the gate
    worse, not better. `docs/journey/13-reacher-residual.md` is the surviving
    account. Kept as a selectable base (`--base fixed`) for reproducibility, not
    as a recommendation.
    """
    from reacher.deepc_setup import make_controller

    ctrl, _info = make_controller(payload, T_ini=T_ini, N=N, lambda_g=lambda_g,
                                  lambda_y=lambda_y)
    return ctrl


def rollout(env, ctrl, seed: int, T_ini: int = 5, anchors=None) -> dict:
    """One full-horizon episode under `ctrl`, recording clone training rows.

    Raises `RuntimeError` (propagated from the solver) if the QP fails; the
    caller drops the episode rather than recording a truncated one, which would
    bias the dataset toward states the solver finds easy.
    """
    _obs, info = env.reset(seed=seed)
    base = env.unwrapped
    goal = base.goal
    y_ref = y_ref_for(goal)

    u_buf = np.zeros((T_ini, NQ_ARM))
    y_buf = np.tile(base.y, (T_ini, 1))
    ctrl.reset(base.y, u_initial=np.zeros(NQ_ARM))

    feats, acts = [], []
    best = float(info["dist"])
    reached = bool(info["reached"])
    for t in range(base.max_steps):
        y_pre = base.y
        feats.append(featurize(u_buf, y_buf, y_pre, goal, t, anchors))
        u = ctrl.act(y_pre, y_ref)
        acts.append(np.clip(u, -1.0, 1.0))

        _obs, _r, _term, trunc, info = env.step(u)
        u_buf = np.roll(u_buf, -1, axis=0)
        u_buf[-1] = info["action"]        # the APPLIED torque, post-clip
        y_buf = np.roll(y_buf, -1, axis=0)
        y_buf[-1] = y_pre                 # the PRE-step measurement
        best = min(best, float(info["dist"]))
        reached = reached or bool(info["reached"])
        if trunc:
            break

    return {"features": np.array(feats), "actions": np.array(acts),
            "reached": reached, "best": best}


def dagger_rollout(env, expert, policy, seed: int, T_ini: int = 5, anchors=None):
    """DAgger round: the STUDENT drives, the EXPERT labels.

    Plain behavioral cloning trains only on states the expert visits, so the
    student is accurate exactly where it will never be once it drives itself.
    Measured here: the clone's disagreement with its expert is **0.1025 at
    expert-visited states and 0.2815 at its own** -- 2.75x. DAgger (Ross, Gordon &
    Bagnell 2011) closes that by aggregating labels collected ON the student's
    distribution, which turns BC's `T^2 * eps` error growth into `T * eps`.

    THE SUBTLETY THAT SILENTLY BREAKS THIS. `DeePC.act` slides its own past
    buffer with the action IT computed (`core/deepc.py:86-89`). When the student
    drives, that action is NOT what the plant received, so from step 2 onward the
    expert would be answering questions about a trajectory that never happened --
    and nothing would raise. The expert's `_u_buf` is therefore corrected to the
    APPLIED action after every step.
    """
    _obs, info = env.reset(seed=seed)
    base = env.unwrapped
    goal = base.goal
    y_ref = y_ref_for(goal)

    u_buf = np.zeros((T_ini, NQ_ARM))
    y_buf = np.tile(base.y, (T_ini, 1))
    expert.reset(base.y, u_initial=np.zeros(NQ_ARM))

    feats, acts = [], []
    best = float(info["dist"])
    reached = bool(info["reached"])
    for t in range(base.max_steps):
        y_pre = base.y
        feats.append(featurize(u_buf, y_buf, y_pre, goal, t, anchors))
        # EXPERT labels this state ...
        acts.append(np.clip(expert.act(y_pre, y_ref), -1.0, 1.0))
        # ... but the STUDENT decides where we go next.
        u_apply = np.clip(np.asarray(policy(env, info), dtype=np.float64), -1.0, 1.0)

        _obs, _r, _term, trunc, info = env.step(u_apply)
        applied = info["action"]
        expert._u_buf[-1] = applied        # see the docstring: this is load-bearing
        u_buf = np.roll(u_buf, -1, axis=0)
        u_buf[-1] = applied
        y_buf = np.roll(y_buf, -1, axis=0)
        y_buf[-1] = y_pre
        best = min(best, float(info["dist"]))
        reached = reached or bool(info["reached"])
        if trunc:
            break

    return {"features": np.array(feats), "actions": np.array(acts),
            "reached": reached, "best": best}


def generate_clone_dataset(n_episodes: int = 200, seed: int = 0, grid=(6, 5),
                           T: int = 1200, T_ini: int = 5, N: int = 12,
                           n_cols: int = 300, n_max: int = 1,
                           stride: int = 2, base: str = "select",
                           carry_prediction: bool = True,
                           bank_seed: int | None = None,
                           episode_offset: int = 0) -> dict:
    """Collect `n_episodes` episodes of the base controller into a training set.

    `base` selects which controller is cloned:
      "select"  Select-DPC (better controller, discontinuous law -- gate FAILED)
      "fixed"   30-anchor fixed libraries (worse controller, smooth law)
    See `build_fixed_controller` for the measurement behind that trade.
    """
    if base not in ("select", "fixed"):
        raise ValueError(f"base must be 'select' or 'fixed', got {base!r}")
    import gymnasium as gym

    import reacher  # noqa: F401  registers the Gym ID

    model, data = load_model()
    # The bank is seeded SEPARATELY so a long collection can be split into chunks
    # that share one identical controller: the episodes must differ, the base
    # controller must not.
    rng = np.random.default_rng(seed if bank_seed is None else bank_seed)
    bank, _payload = build_bank(model, data, rng, grid=grid, T=T, T_ini=T_ini,
                                N=N, stride=stride)
    if base == "select":
        ctrl = build_select_controller(bank, T_ini=T_ini, N=N, n_cols=n_cols,
                                       n_max=n_max,
                                       carry_prediction=carry_prediction)
    else:
        ctrl = build_fixed_controller(_payload, T_ini=T_ini, N=N)
    # The fixed-anchor controller HAS a discrete library index, so hand it to the
    # clone exactly as the unicycle pipeline does. Select-DPC has none.
    anchors = _payload["anchors"] if base == "fixed" else None

    env = gym.make("ReacherGoal-v0")
    feats, acts, reached, dropped = [], [], 0, 0
    try:
        for ep in range(n_episodes):
            try:
                rec = rollout(env, ctrl,
                              seed=seed * 100_000 + episode_offset + ep,
                              T_ini=T_ini, anchors=anchors)
            except RuntimeError:
                dropped += 1
                continue
            feats.append(rec["features"])
            acts.append(rec["actions"])
            reached += int(rec["reached"])
    finally:
        env.close()

    if not feats:
        raise RuntimeError(f"every one of {n_episodes} episodes failed in the solver")

    return {
        "features": np.vstack(feats),
        "actions": np.vstack(acts),
        "meta": {"T_ini": T_ini, "N": N, "n_cols": n_cols, "n_max": n_max,
                 "grid": list(grid), "T": T, "stride": stride,
                 "n_episodes": n_episodes, "n_dropped": dropped,
                 "n_reached": reached, "seed": seed, "base": base, "carry_prediction": carry_prediction,
                 "n_lib": (0 if anchors is None else int(len(anchors))),
                 "bank_seed": (seed if bank_seed is None else bank_seed),
                 "episode_offset": episode_offset,
                 "bank_columns": int(bank["Up"].shape[1])},
    }
