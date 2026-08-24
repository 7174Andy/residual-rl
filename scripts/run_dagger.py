"""DAgger: aggregate labels on the STUDENT's own state distribution.

Behavioral cloning here fails in the classic way -- the clone's disagreement with
its expert is 0.1025 at expert-visited states and 0.2815 at its own (2.75x), which
is exactly what DAgger exists to fix. Every dataset collected before this ran the
EXPERT and labelled with the EXPERT, so the clone's own states appear in training
data precisely never.

Each round: roll the current clone, ask the expert what it would have done at
every state the clone reached, aggregate, retrain, gate. The dataset only grows.

    uv run python scripts/run_dagger.py --rounds 3 --episodes 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_data import (
    build_bank, build_select_controller, dagger_rollout,
)
from reacher.eval import ClonePolicy
from reacher.model import load_model
from rl.clone import load_clone


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dataset", default="data/rcd_ml0.npz",
                   help="round-0 BC dataset (the expert-driven one)")
    p.add_argument("--seed-clone", default="data/reacher_clone_ml.pt")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=100, help="per round")
    p.add_argument("--out-prefix", default="data/dagger")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with np.load(args.seed_dataset) as z:
        X = z["features"]
        Y = z["actions"]
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
    carry = bool(meta.get("carry_prediction", True))
    n_lib = int(meta.get("n_lib", 0))
    print(f"round 0 (BC): {X.shape[0]} rows x {X.shape[1]} features, "
          f"n_lib={n_lib}, carry_prediction={carry}")

    model, data = load_model()
    bank, payload = build_bank(model, data, np.random.default_rng(0))
    expert = build_select_controller(bank, carry_prediction=carry)
    anchors = payload["anchors"] if n_lib else None

    clone_path = args.seed_clone
    for r in range(1, args.rounds + 1):
        # --- roll the CURRENT clone, label with the expert -------------------
        predictor = load_clone(clone_path, device="cpu")
        policy = ClonePolicy(predictor, anchors=anchors)
        env = gym.make("ReacherGoal-v0")
        feats, acts, reached, gap = [], [], 0, []
        try:
            for ep in range(args.episodes):
                rec = dagger_rollout(env, expert, policy,
                                     seed=800_000 + r * 10_000 + ep,
                                     anchors=anchors)
                feats.append(rec["features"])
                acts.append(rec["actions"])
                reached += int(rec["reached"])
                # How far off was the clone on the states it chose? This is the
                # quantity DAgger is supposed to shrink round over round.
                gap.append(np.median(np.linalg.norm(
                    predictor.predict(rec["features"]) - rec["actions"], axis=1)))
        finally:
            env.close()

        newX, newY = np.vstack(feats), np.vstack(acts)
        print(f"\nround {r}: collected {newX.shape[0]} rows on the CLONE's "
              f"distribution ({reached}/{args.episodes} of those episodes reached)")
        print(f"  clone-vs-expert error on its OWN states: "
              f"{np.median(gap):.4f}   <- DAgger should shrink this")

        X, Y = np.vstack([X, newX]), np.vstack([Y, newY])
        ds = f"{args.out_prefix}_r{r}.npz"
        np.savez(ds, features=X, actions=Y,
                 **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        print(f"  aggregated dataset: {X.shape[0]} rows -> {ds}")

        clone_path = f"{args.out_prefix}_clone_r{r}.pt"
        subprocess.run([sys.executable.replace("python", "python"),
                        "scripts/train_reacher_clone.py",
                        "--dataset", ds, "--out", clone_path], check=True)
        subprocess.run([sys.executable, "scripts/validate_reacher_clone.py",
                        "--clone", clone_path, "--dataset", ds,
                        "--base", str(meta.get("base", "select"))], check=True)

    print(f"\nfinal clone: {clone_path}")


if __name__ == "__main__":
    main()
