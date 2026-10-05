"""Train the Panda behavioral clone on one or more collected rounds.

    uv run python scripts/train_panda_clone.py \
      --data data/pcd_r0.npz --out data/panda_clone_r0.pt

DAgger aggregates: pass every round collected so far, in order. Training on the
newest round alone is a different (and worse) algorithm -- the aggregation IS
DAgger.

    uv run python scripts/train_panda_clone.py \
      --data data/pcd_r0.npz data/pcd_r1.npz --out data/panda_clone_r1.pt

`n_lib = 0` always: Select-DPC re-selects 300 of ~180,000 columns every step, so
there is no discrete library index to protect from standardization (the unicycle
clone had one and it was load bearing; see panda/clone_features.py).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from panda.clone_features import q_slice
from panda.env import DELTA_MAX
from rl.clone import save_clone, train_clone


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--target", choices=("delta", "absolute"), default="delta",
                   help="what the net regresses. 'delta' (default) predicts "
                        "q_des - q and the deploy path adds q back; 'absolute' "
                        "predicts q_des directly. Measured 2026-09-30 on "
                        "pcd_r0: absolute scores +5.1%% skill over echoing the "
                        "current joints on an EPISODE-wise holdout, delta "
                        "+43.7%%, because q_des - q has std 0.07 rad inside a "
                        "target of std ~0.9 rad -- under 'absolute' the loss "
                        "spends itself re-encoding q, which the features "
                        "already carry verbatim at [85:92].")
    p.add_argument("--squash", action="store_true",
                   help="train under SAC's tanh head for warm-start RL: target "
                        "is clip(q_des - q, +-delta_max) / delta_max, in [-1, 1] "
                        "like the RL action. Needs --target delta.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="clone_panda")
    args = p.parse_args()

    feats, acts, metas = [], [], []
    for path in args.data:
        with np.load(path, allow_pickle=True) as z:
            feats.append(z["features"])
            acts.append(z["actions"])
            metas.append(json.loads(str(z["meta"])) if "meta" in z else {})
            print(f"{path}: {z['features'].shape[0]} rows")
    X, Y = np.vstack(feats), np.vstack(acts)
    print(f"total {X.shape[0]} rows, {X.shape[1]} features -> {Y.shape[1]} out")

    # A clone trained across rounds collected under DIFFERENT labeller settings
    # would be imitating two different experts; catch it here rather than in the
    # gate, where it would show up as unexplained disagreement.
    keys = {(m.get("n_max"), m.get("carry"), m.get("n_cols"), m.get("stride"),
             m.get("T_ini"), m.get("libs")) for m in metas if m}
    if len(keys) > 1:
        raise SystemExit(
            "the --data rounds were collected under different labeller "
            f"settings: {keys}. Re-collect, or train them separately.")

    # y_cur = [q (7); tip (3)] sits at [85:92] of the 99-D vector, so the delta
    # target needs no re-collection -- the labels on disk stay the absolute
    # q_des the plant actually received, which is the canonical record.
    T_ini = next((m["T_ini"] for m in metas if m.get("T_ini")), 5)
    targets = Y - X[:, q_slice(T_ini)] if args.target == "delta" else Y
    if args.squash:
        if args.target != "delta":
            p.error("--squash needs --target delta")
        targets = np.clip(targets, -DELTA_MAX, DELTA_MAX) / DELTA_MAX

    model, stats, history = train_clone(
        X, targets, n_lib=0, hidden=tuple(args.hidden), epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        seed=args.seed, squash=args.squash)
    # Load-bearing: the deploy path reads this to decide whether to add q back.
    # Reading a delta checkpoint as absolute (or vice versa) fails silently --
    # the same hazard class as the u_i delta-vs-absolute collision in journey 15.
    stats["target"] = args.target
    if args.squash:
        stats["out_scale"] = DELTA_MAX   # deploy multiplies back to radians
    save_clone(args.out, model, stats)
    best_val = min(history["val_mse"])
    print(f"wrote {args.out}: target={args.target} best val MSE {best_val:.6f} "
          f"after {len(history['val_mse'])} epochs")

    if args.wandb_project:
        from rl.wb import init_run
        run = init_run(args.wandb_project, args.wandb_name,
                       config={"data": args.data, "rows": int(X.shape[0]),
                               "hidden": args.hidden, "seed": args.seed,
                               "labeller": (list(keys)[0] if keys else None)},
                       tags=["panda", "clone"], group=args.wandb_group)
        if run is not None:
            for e, (tr, va) in enumerate(zip(history["train_mse"],
                                             history["val_mse"])):
                run.log({"panda/clone_train_mse": tr,
                         "panda/clone_val_mse": va, "panda/clone_epoch": e})
            run.log({"panda/clone_best_val_mse": best_val})
            run.finish()


if __name__ == "__main__":
    main()
