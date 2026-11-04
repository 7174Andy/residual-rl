"""DAgger for the unicycle clone: aggregate DeePC labels on the CLONE's own states.

The unicycle clone (journey 07) was trained only on states DeePC itself visits (plus
synthetic pasts), so it is accurate exactly where it will not be once it drives --
the coverage failure journey 14 measured on the Reacher. Mirrors `run_dagger.py`
(Reacher) and the Panda's box rounds: each round rolls the current clone, labels
every visited state with canonical DeePC, aggregates, retrains. After the last
round a squash clone is trained on the final dataset for the warm-start arm.

    OMP_NUM_THREADS=1 uv run python -u scripts/run_unicycle_dagger.py --rounds 3 --episodes 100

Each DeePC label is a QP solve (~0.3 s, up to ~1 s on the clone's states), so a
round of ~20k labels is spread over `--workers` processes, one DeePC each.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys

import numpy as np

from rl.clone import load_clone
from two_wheel_robot.rl.clone_data import _reset_deepc_solver_state, dagger_rollout
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv, WarmDeePCEnv

EVAL_SEEDS = [4104626029 + i for i in range(78)]   # the canonical 78-seed sweep


def reach(env, act) -> int:
    """Deterministic reach count over the 78 eval seeds, `act(obs) -> action`."""
    n = 0
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s)
        term = trunc = False
        info: dict = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(act(obs))
        n += bool(info.get("reached", False))
    return n


_W: dict = {}


def _init_worker(clone_path: str, libraries: str) -> None:
    deepc, _info = build_canonical_deepc(libraries_path=libraries)
    _reset_deepc_solver_state(deepc)
    _W["deepc"], _W["env"] = deepc, ResidualDeePCEnv(clone_path, libraries)


def _rollout(seed: int):
    return dagger_rollout(_W["deepc"], _W["env"], seed=seed)


def train(ds: str, out: str, *extra: str) -> None:
    subprocess.run([sys.executable, "scripts/train/train_clone.py", "--data", ds, "--out", out,
                    "--device", "cpu", *extra], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dataset", default="data/clone_dataset.npz",
                   help="round-0 BC dataset (expert-driven + synthetic)")
    p.add_argument("--seed-clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=100, help="per round")
    p.add_argument("--out-prefix", default="data/unicycle_dagger")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    with np.load(args.seed_dataset, allow_pickle=True) as z:
        ds = {k: z[k] for k in z.files}

    clone_path = args.seed_clone
    env = ResidualDeePCEnv(clone_path, args.libraries)
    print(f"round 0 (BC): {len(ds['targets'])} rows, clone reach "
          f"{reach(env, lambda _o: np.zeros(2))}/78")
    for r in range(1, args.rounds + 1):
        # --- roll the CURRENT clone, label with DeePC ------------------------
        predictor = load_clone(clone_path, device="cpu")
        feats, targs, idxs, reached = [], [], [], 0
        seeds = [800_000 + r * 10_000 + ep for ep in range(args.episodes)]
        # Sorted back into seed order so the dataset does not depend on scheduling.
        with mp.Pool(args.workers, _init_worker, (clone_path, args.libraries)) as pool:
            done = []
            for k, res in enumerate(pool.imap(_rollout, seeds), 1):
                done.append(res)
                if k % 10 == 0:
                    print(f"  round {r}: {k}/{args.episodes} episodes labeled", flush=True)
        for f, t, i, ok in done:
            feats += f
            targs += t
            idxs += i
            reached += ok
        X, Y = np.asarray(feats), np.asarray(targs)
        # The quantity DAgger is supposed to shrink round over round.
        gap = np.median(np.abs(predictor.predict(X) - Y), axis=0)
        print(f"\nround {r}: {len(Y)} rows on the CLONE's distribution "
              f"({reached}/{args.episodes} episodes reached); clone-vs-DeePC median "
              f"|err| on its own states: v {gap[0]:.3f}, w {gap[1]:.3f}")

        ds["features"] = np.vstack([ds["features"], X])
        ds["targets"] = np.vstack([ds["targets"], Y])
        ds["library_idx"] = np.concatenate([ds["library_idx"], idxs])
        ds["regime"] = np.concatenate([ds["regime"], [f"dagger_r{r}"] * len(Y)])
        path = f"{args.out_prefix}_r{r}.npz"
        np.savez(path, **ds)

        clone_path = f"{args.out_prefix}_clone_r{r}.pt"
        train(path, clone_path)
        env.close()
        env = ResidualDeePCEnv(clone_path, args.libraries)
        print(f"  aggregated {len(ds['targets'])} rows -> {path}; {clone_path} reach "
              f"{reach(env, lambda _o: np.zeros(2))}/78")
    env.close()

    squash = f"{clone_path[:-3]}_squash.pt"
    train(path, squash, "--squash", "--libraries", args.libraries)
    sq = load_clone(squash, device="cpu")
    warm = WarmDeePCEnv(squash, args.libraries)
    print(f"\nfinal clone {clone_path}; squash {squash} reach "
          f"{reach(warm, lambda o: sq.predict(o.astype(np.float64)))}/78")
    warm.close()


if __name__ == "__main__":
    main()
