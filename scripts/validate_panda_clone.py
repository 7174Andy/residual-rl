"""The Phase 2 clone gate: four layers, cheapest first.

    uv run python scripts/validate_panda_clone.py \
      --clone data/panda_clone_r2.pt \
      --data data/pcd_r0.npz data/pcd_r1.npz data/pcd_r2.npz --n-scen 24

    # layer 3 (expensive: the full frozen 78 through the gym plant)
    uv run python scripts/validate_panda_clone.py --clone data/panda_clone_r2.pt \
      --data ... --reach --method panda_clone_r2

Acceptance is STEERING FIDELITY, not reach parity: the reacher clone shipped at
82 against its expert's 96 and the residual made up the difference. The gate
exists to catch a clone that steers somewhere ELSE, not one that steers the same
way less precisely.

  1  held-out regression MSE, split by buffer regime (primed vs settled)
  2  closed-loop deviation from the expert, same scenarios, same starts
  3  paired reach on the frozen 78, McNemar (opt-in, --reach)
  4  disagreement ratio at expert-visited vs clone-visited states

Layer 4 is the one that says whether DAgger did its job. The unicycle's
off-policy signature was a 2.75x ratio; DAgger exists to flatten it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from panda import scenarios as sc
from panda.clone_data import rollout
from panda.clone_features import NQ, Y_WIDTH, featurize
from panda.model import load_model, safe_box
from panda.selectdpc import make_select_controller, panda_bank
from rl.clone import load_clone


def layer1_regression(pred, data_paths: list[str]) -> dict:
    """Held-out MSE, split by buffer regime. Uses the checkpoint's own split."""
    feats, acts = [], []
    for p in data_paths:
        with np.load(p, allow_pickle=True) as z:
            feats.append(z["features"])
            acts.append(z["actions"])
    X, Y = np.vstack(feats), np.vstack(acts)
    if pred.n_train_samples != X.shape[0]:
        raise SystemExit(
            f"dataset has {X.shape[0]} rows but the checkpoint was trained on "
            f"{pred.n_train_samples} -- pass the SAME --data training used, or "
            "val_idx points at different rows and layer 1 is meaningless")
    idx = pred.val_idx
    err = pred.predict(X[idx]) - Y[idx]
    # The validity feature is the last column; < 1.0 means a part-primed buffer.
    primed = X[idx, -1] < 1.0
    return {
        "n_val": int(idx.size),
        "val_mse": float((err ** 2).mean()),
        "val_mse_primed": (float((err[primed] ** 2).mean())
                           if primed.any() else None),
        "val_mse_settled": (float((err[~primed] ** 2).mean())
                            if (~primed).any() else None),
        "frac_primed": float(primed.mean()),
    }


def build_expert(libs: str, model, args):
    with np.load(libs) as z:
        payload = {k: z[k] for k in z.files}
    bank = panda_bank(payload, args.T_ini, args.N, stride=args.stride)
    del payload
    ctrl = make_select_controller(bank, model, T_ini=args.T_ini, N=args.N,
                                  n_cols=args.n_cols, n_max=args.n_max,
                                  du_max=args.du_max)
    ctrl.carry_prediction = args.carry
    return ctrl


def layers_2_and_4(pred, libs: str, args) -> dict:
    """Closed-loop deviation (layer 2) and the disagreement ratio (layer 4).

    Both come from the same two rollout families, so they are measured together:
    an EXPERT-driven episode and a CLONE-driven one from each identical start.
    `rollout` records the EXPERT's label at every visited state, so re-running
    the clone over those same recorded features gives its answer at identical
    states -- a true like-for-like disagreement on each driver's distribution.

    BC's failure is that the two differ; DAgger's job is to make them the same.
    """
    model, data = load_model()
    lo, hi = safe_box(model)
    ctrl = build_expert(libs, model, args)
    scen = sc.load(args.scenarios)

    def drive_with_clone(features, _q):
        return np.clip(pred.predict(features), lo, hi)

    dev = {"expert": [], "clone": []}
    for sid in range(args.n_scen):
        q0 = np.asarray(scen["qpos"][sid], float)
        goal = np.asarray(scen["goal"][sid], float)
        for driver in ("expert", "clone"):
            rec = rollout(model, data, ctrl, q0, goal, max_steps=args.steps,
                          tol=args.tol, T_ini=args.T_ini,
                          policy=None if driver == "expert" else drive_with_clone)
            clone_acts = np.clip(pred.predict(rec["features"]), lo, hi)
            dev[driver].append(
                np.linalg.norm(clone_acts - rec["actions"], axis=1))
        print(f"  scenario {sid}: done", flush=True)

    de, dc = np.concatenate(dev["expert"]), np.concatenate(dev["clone"])
    return {
        "deviation_expert_median": float(np.median(de)),
        "deviation_expert_p90": float(np.percentile(de, 90)),
        "deviation_clone_median": float(np.median(dc)),
        "deviation_clone_p90": float(np.percentile(dc, 90)),
        "disagreement_ratio": float(np.median(dc) / max(np.median(de), 1e-12)),
        "n_scen": args.n_scen,
    }


class ClonePolicy:
    """`policy(env) -> delta (7,)` for `panda/eval.py::run_scenarios`.

    The clone outputs an ABSOLUTE `q_des`; the env's action is a DELTA. The
    conversion here is the same one Phase 3's residual env will use, clip
    included, so this is the honest "clone as deployed" number.

    Stateful by necessity -- the clone needs the past `T_ini` window, which the
    env does not carry. `run_scenarios` resets the env between scenarios, so the
    buffer re-primes whenever `step_idx` is back at 0; without that, scenario k
    would inherit scenario k-1's history.
    """

    def __init__(self, pred, T_ini: int, delta_max: float):
        self.pred, self.T_ini, self.delta_max = pred, T_ini, delta_max
        self.u = np.zeros((T_ini, NQ))
        self.y = np.zeros((T_ini, Y_WIDTH))
        self.t = 0
        # Which quantity the net regresses. Checkpoints written before the flag
        # existed are absolute by construction. Getting this wrong is silent --
        # a delta read as absolute commands a tiny joint target near the origin
        # and the episode looks merely bad, not broken.
        # "increment" = step from the previous applied target (u_t - u_{t-1}).
        self.target = pred.stats.get("target", "absolute")
        if self.target not in ("delta", "absolute", "increment"):
            raise ValueError(f"unknown clone target {self.target!r}")

    def __call__(self, env):
        base = env.unwrapped
        q = base.state[: base.nq].copy()
        lo, hi = base.safe_box
        y = np.concatenate([q, base.y])
        if getattr(base, "step_idx", 0) == 0:
            self.u = np.tile(q, (self.T_ini, 1))
            self.y = np.tile(y, (self.T_ini, 1))
            self.t = 0
        f = featurize(self.u, self.y, y, base.goal, self.t)
        out = self.pred.predict(f) * self.pred.stats.get("out_scale", 1.0)
        base_q = {"delta": q, "increment": self.u[-1]}.get(self.target, 0.0)
        q_des = np.clip(base_q + out, lo, hi)
        delta = np.clip(q_des - q, -self.delta_max, self.delta_max)
        self.u = np.roll(self.u, -1, axis=0)
        self.u[-1] = np.clip(q + delta, lo, hi)
        self.y = np.roll(self.y, -1, axis=0)
        self.y[-1] = y
        self.t += 1
        return delta


def layer3_reach(pred, args) -> dict:
    """Paired reach on the frozen 78, through the gym (delta) plant."""
    from panda.env import PandaReachEnv
    from panda.eval import append_results, run_scenarios

    scen = sc.load(args.scenarios)
    # Construct the env directly, NOT via gym.make: `run_scenarios` calls
    # `scenarios.validate_against_env`, which reads `delta_max`/`max_steps` off
    # the env with getattr, and gym.make's OrderEnforcing wrapper does not
    # forward them. Every other caller in this repo does the same.
    env = PandaReachEnv()
    pol = ClonePolicy(pred, args.T_ini, args.delta_max)
    n = args.n_reach if args.n_reach > 0 else 78
    try:
        rows = run_scenarios(env, pol, list(range(n)), scen, args.method)
    finally:
        env.close()
    append_results(rows, args.results)
    k = sum(1 for r in rows if r["reached"])
    return {"method": args.method, "reached": k, "n": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clone", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--libs", default="data/panda_taskbank_20k_sdpc.npz")
    p.add_argument("--scenarios", default="data/panda_scenarios_v1.npz")
    p.add_argument("--n-scen", type=int, default=24)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--delta-max", type=float, default=0.2)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-max", type=int, default=1)
    p.add_argument("--carry", action="store_true")
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--reach", action="store_true",
                   help="also run layer 3 (the full frozen 78; expensive)")
    p.add_argument("--method", default="panda_clone")
    p.add_argument("--results", default="data/panda_results.csv",
                   help="where layer 3 appends its rows; point elsewhere for "
                        "smoke runs so the real results CSV stays clean")
    p.add_argument("--n-reach", type=int, default=0,
                   help="layer 3 scenario count; 0 = the full frozen 78")
    p.add_argument("--skip-closed-loop", action="store_true",
                   help="layer 1 only (seconds, no QP)")
    p.add_argument("--out", default="data/panda_clone_gate.json")
    args = p.parse_args()

    pred = load_clone(args.clone)
    out = {"clone": args.clone, "data": args.data}
    out["layer1"] = layer1_regression(pred, args.data)
    print("layer 1:", json.dumps(out["layer1"], indent=2), flush=True)

    if not args.skip_closed_loop:
        out["layer2_4"] = layers_2_and_4(pred, args.libs, args)
        print("layers 2+4:", json.dumps(out["layer2_4"], indent=2), flush=True)
    if args.reach:
        out["layer3"] = layer3_reach(pred, args)
        print("layer 3:", json.dumps(out["layer3"], indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
