"""THE GATE (spec R1): does the clone actually reproduce Select-DPC?

The unicycle clone worked because 4-library DeePC is piecewise smooth.
Select-DPC's action passes through an `argpartition` over thousands of columns,
so the state -> action map may be genuinely discontinuous at selection
boundaries. Measure it before spending on RL.

Two comparisons, on IDENTICAL frozen scenarios:

  open loop    regression error on held-out rows -- does the MLP fit the map?
  closed loop  reach rate and per-scenario `best` -- does the fit SURVIVE
               being put back in the loop, where errors compound?

The second is the one that decides. A clone can fit well and still diverge once
its own output feeds its next input.

PASS if closed-loop reach rate is within 10 points of Select-DPC's and the median
per-scenario `best` gap is under 5 mm. Otherwise report and stop: the fallback is
to clone the 30-anchor fixed controller (84/120) instead, trading 12 reaches of
baseline quality for a smoother map.

    uv run python scripts/validate_reacher_clone.py
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_data import (
    build_bank, build_fixed_controller, build_select_controller,
)
from reacher.eval import ClonePolicy, ControllerPolicy, run_episode
from reacher.model import load_model
from rl.clone import load_clone
from rl.stats import wilson_ci


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--dataset", default="data/dagger_r3.npz")
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--base", default="select", choices=["select", "fixed"],
                   help="which controller the clone was trained on")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    predictor = load_clone(args.clone, device="cpu")

    # --- open loop, on the checkpoint's own held-out split -------------------
    with np.load(args.dataset) as z:
        features, actions = z["features"], z["actions"]
        # Read the base's configuration FROM THE DATASET rather than taking it
        # from a flag. The gate compares the clone against the controller it was
        # trained on; rebuilding that controller with different settings measures
        # clone-vs-a-different-controller and silently invalidates the verdict.
        carry = bool(z["meta_carry_prediction"]) if "meta_carry_prediction" in z \
            else True
        ds_base = str(z["meta_base"]) if "meta_base" in z else args.base
        n_lib = int(z["meta_n_lib"]) if "meta_n_lib" in z else 0
    if ds_base != args.base:
        raise SystemExit(f"dataset was collected with base={ds_base!r} but --base "
                         f"is {args.base!r}; the gate would compare the clone "
                         f"against a controller it never saw.")
    print(f"[BASE] {ds_base}, carry_prediction={carry}, n_lib={n_lib} (read from the dataset)")
    val_idx = predictor.val_idx
    if val_idx is not None and predictor.n_train_samples == features.shape[0]:
        pred = predictor.predict(features[val_idx])
        err = np.linalg.norm(pred - actions[val_idx], axis=1)
        spread = np.linalg.norm(actions[val_idx] - actions[val_idx].mean(0), axis=1)
        print(f"[OPEN LOOP] {len(val_idx)} held-out rows")
        print(f"  median |u_clone - u_select|  {np.median(err):.4f}  "
              f"(torque units, box width 2.0)")
        print(f"  p95                          {np.percentile(err, 95):.4f}")
        print(f"  median |u_select - mean|     {np.median(spread):.4f}  "
              f"<- what a constant predictor would score")
    else:
        print("[OPEN LOOP] skipped: checkpoint's val split does not match this dataset")

    # --- closed loop, paired on identical scenarios --------------------------
    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(len(z["qpos"]))]
    eps = eps[: args.episodes]

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    bank, payload = build_bank(model, data, rng)
    base_ctrl = (build_select_controller(bank, n_cols=args.n_cols,
                                        carry_prediction=carry)
                 if args.base == "select" else build_fixed_controller(payload))
    select = ControllerPolicy(base_ctrl)
    clone = ClonePolicy(predictor,
                        anchors=payload["anchors"] if n_lib else None)

    env = gym.make("ReacherGoal-v0")
    rows = {"select": [], "clone": []}
    for q0, goal in eps:
        rows["select"].append(run_episode(env, select, q0, goal))
        rows["clone"].append(run_episode(env, clone, q0, goal))
    env.close()

    print(f"\n[CLOSED LOOP] {len(eps)} frozen scenarios, full horizon")
    print(f"  {'':<12}{'reached':>14}{'best':>10}{'final':>10}{'path/net':>10}")
    for key, label in (("select", args.base), ("clone", "clone")):
        r = rows[key]
        k, n = sum(x["reached"] for x in r), len(r)
        lo, hi = wilson_ci(k, n)
        print(f"  {label:<12}{k:>4}/{n:<3}[{100*lo:>3.0f}-{100*hi:<3.0f}%]"
              f"{np.median([x['best'] for x in r])*1e3:>9.1f}mm"
              f"{np.median([x['final'] for x in r])*1e3:>9.1f}mm"
              f"{np.nanmedian([x['eff'] for x in r]):>10.1f}")

    gap = np.median([c["best"] - s["best"]
                     for c, s in zip(rows["clone"], rows["select"])])
    k_s = sum(x["reached"] for x in rows["select"])
    k_c = sum(x["reached"] for x in rows["clone"])
    drop = 100.0 * (k_s - k_c) / len(eps)
    closer = sum(c["best"] < s["best"] for c, s in zip(rows["clone"], rows["select"]))
    print(f"\n  median per-scenario best gap (clone - select): {gap*1e3:+.1f} mm")
    print(f"  reach-rate drop: {drop:+.1f} points  ({k_s} -> {k_c})")
    print(f"  clone closer on {closer}/{len(eps)} scenarios (paired)")
    verdict = "PASS" if (drop <= 10.0 and abs(gap) < 0.005) else "FAIL"
    print(f"  GATE: {verdict}")
    if verdict == "FAIL":
        print("  -> spec R1 fallback: clone the 30-anchor fixed controller instead")


if __name__ == "__main__":
    main()
