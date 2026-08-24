"""One episode loop, shared by every row of the results table.

Every controller is scored by the same function so no row can differ by protocol
rather than by policy. The metrics and their definitions are
`docs/reference/metrics.md`; the two that matter most here:

* `best` -- closest approach at any point. The reach criterion.
* `final` -- distance at the LAST step. Journey 12 found every controller
  arrives at roughly half its final error and backs off by 2.1-2.3x, which is
  invisible under early stopping. This loop therefore ALWAYS runs the full
  horizon; there is no early-stop option, deliberately.

The two policy adapters live here rather than in a script because three callers
need them (the fidelity gate, the residual env's invariant test, and the final
evaluation), and a package module must not import from `scripts/`.
"""
from __future__ import annotations

import numpy as np

from reacher.clone_features import featurize
from reacher.deepc_setup import y_ref_for
from reacher.model import NQ_ARM


def run_episode(env, policy, qpos: np.ndarray, goal: np.ndarray,
                seed: int = 0) -> dict:
    """Run one scenario under `policy(env, info) -> action`.

    Returns `need`, `best`, `final`, `path`, `eff` (= path/net), `reached`,
    `steps` (first step inside tolerance, or None) and `steps_run`.
    """
    _obs, info = env.reset(seed=seed, options={"qpos": qpos, "goal": goal})
    base = env.unwrapped
    need = float(info["dist"])
    best = need
    reached_at = None

    steps_run = 0
    for t in range(base.max_steps):
        u = policy(env, info)
        _obs, _r, _term, trunc, info = env.step(u)
        steps_run += 1
        d = float(info["dist"])
        if d < best:
            best = d
        if reached_at is None and info["reached"]:
            reached_at = t + 1
        if trunc:
            break

    net = need - best
    return {
        "need": need,
        "best": best,
        "final": float(info["dist"]),
        "path": float(info["path"]),
        # net ~ 0 means no progress was made, so the ratio has no meaning --
        # NaN, not a huge number that would drag a median.
        "eff": float(info["path"] / net) if net > 1e-4 else float("nan"),
        "reached": reached_at is not None,
        "steps": reached_at,
        "steps_run": steps_run,
    }


class ControllerPolicy:
    """Any DeePC-family controller as a closed-loop policy.

    Works for Select-DPC and for the fixed-anchor `ReacherDeePC` alike, because
    both expose the same `reset(y_initial, u_initial)` / `act(y, y_ref)` contract.
    Named for the contract rather than for one implementation: the fidelity gate
    needs to run both through the identical loop.
    """

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def __call__(self, env, info):
        base = env.unwrapped
        if base.step_idx == 0:
            self.ctrl.reset(base.y, u_initial=np.zeros(NQ_ARM))
        return np.clip(self.ctrl.act(base.y, y_ref_for(base.goal)), -1.0, 1.0)


class ClonePolicy:
    """The clone as a closed-loop policy.

    Slides `(applied u, pre-step y)` exactly as `reacher/clone_data.py::rollout`
    did when labelling -- deploying under a different convention than the one it
    was trained on is a silent one-step shift, not an error.
    """

    def __init__(self, predictor, T_ini: int = 5, anchors=None):
        self.predictor = predictor
        self.T_ini = T_ini
        # MUST match what the dataset was built with, or the clone is told the
        # wrong mode -- worse than being told none.
        self.anchors = anchors
        self._u_buf = np.zeros((T_ini, NQ_ARM))
        self._y_buf = np.zeros((T_ini, NQ_ARM + 2))

    def __call__(self, env, info):
        base = env.unwrapped
        if base.step_idx == 0:                       # priming: a new episode
            self._u_buf = np.zeros((self.T_ini, NQ_ARM))
            self._y_buf = np.tile(base.y, (self.T_ini, 1))
        y_pre = base.y
        u = np.clip(self.predictor.predict(
            featurize(self._u_buf, self._y_buf, y_pre, base.goal,
                      base.step_idx, self.anchors)), -1.0, 1.0)
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = u
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_pre
        return u


class WarmStartClonePolicy:
    """The clone, with the BASE controller driving the first `T_ini` steps.

    Why this exists. The clone's error is ~7x higher at step 0 than mid-episode,
    and that is NOT a data-volume problem -- measured, see
    `.superpowers/sdd/2026-08-16-reacher-residual-rl/task-6-diagnosis.md`.
    Tripling episodes from 200 to 600 moved step-0 error by 0.0002. The reason is
    dimensional: while the buffer is synthetic priming, the only free inputs are
    `y0` (4) and `goal` (2), so the start regime is a ~6-D slice sampled ONCE per
    episode, and its label surface is 2.6x steeper than the history-rich regime.
    Halving nearest-neighbour distance in 6-D needs 2^6 = 64x the episodes.

    So instead of paying for coverage that cannot be bought, delete the regime:
    let the base controller run while the buffer fills with real history, then hand
    over. Costs `T_ini` QP solves per episode -- 5 of 50 steps here, a 10x speedup
    against the base rather than the pure clone's ~700x -- and the clone then only
    ever sees the regime it is good at.
    """

    def __init__(self, predictor, base_ctrl, T_ini: int = 5, anchors=None):
        self.clone = ClonePolicy(predictor, T_ini=T_ini, anchors=anchors)
        # The controller itself, not a ControllerPolicy wrapper: that wrapper
        # re-resets at step_idx == 0, which would fight this class's own priming.
        self.ctrl = base_ctrl
        self.T_ini = T_ini

    def __call__(self, env, info):
        base_env = env.unwrapped
        if base_env.step_idx == 0:                 # prime BOTH sub-policies
            self.clone._u_buf = np.zeros((self.T_ini, NQ_ARM))
            self.clone._y_buf = np.tile(base_env.y, (self.T_ini, 1))
            self.ctrl.reset(base_env.y, u_initial=np.zeros(NQ_ARM))
        if base_env.step_idx < self.T_ini:
            u = np.clip(self.ctrl.act(base_env.y,
                                      y_ref_for(base_env.goal)), -1.0, 1.0)
            # The clone's buffer must still slide, or it inherits a stale history
            # the moment it takes over.
            y_pre = base_env.y
            self.clone._u_buf = np.roll(self.clone._u_buf, -1, axis=0)
            self.clone._u_buf[-1] = u
            self.clone._y_buf = np.roll(self.clone._y_buf, -1, axis=0)
            self.clone._y_buf[-1] = y_pre
            return u
        return self.clone(env, info)
