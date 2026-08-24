"""Merge chunked clone datasets into one training set.

Collection is chunked because a single 600-episode run takes ~30 min and this
host has killed long-running jobs; three ~10 min chunks complete reliably. The
chunks must share ONE base controller, so `gen_reacher_clone_data.py` seeds the
anchor bank (`--bank-seed`) separately from the episodes (`--episode-offset`).

This script enforces that: it refuses to merge chunks whose `bank_seed`, `base`,
feature width, or controller hyperparameters disagree. Silently concatenating
chunks from two different controllers would train the clone on a mixture and the
resulting fidelity number would mean nothing.

    uv run python scripts/merge_reacher_clone_data.py data/rcd_c*.npz \\
        --out data/reacher_clone_dataset_600.npz
"""
from __future__ import annotations

import argparse

import numpy as np

# Every chunk must agree on these or the merge is unsound.
MUST_MATCH = ("bank_seed", "base", "T_ini", "N", "n_cols", "n_max", "T",
              "stride", "bank_columns", "grid", "seed", "n_lib",
              "carry_prediction")
# `grid` matters even though `bank_columns` is checked: (6,5) and (10,3) both give
# 30 anchors and identical bank_columns from DIFFERENT anchors -- exactly the
# two-controller mixture this script exists to refuse. `seed` is provenance: the
# merged file records ref["seed"] as its own, so a mismatch would make that a lie.


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", nargs="+")
    p.add_argument("--out", default="data/reacher_clone_dataset_600.npz")
    args = p.parse_args()

    feats, acts, metas, offsets = [], [], [], []
    for path in args.chunks:
        with np.load(path) as z:
            feats.append(z["features"])
            acts.append(z["actions"])
            metas.append({k[5:]: z[k] for k in z.files if k.startswith("meta_")})
        offsets.append(int(metas[-1]["episode_offset"]))
        print(f"  {path}: {feats[-1].shape[0]} rows x {feats[-1].shape[1]} "
              f"features, offset {offsets[-1]}, "
              f"{int(metas[-1]['n_reached'])} reached")

    ref = metas[0]
    for path, m in zip(args.chunks[1:], metas[1:]):
        for k in MUST_MATCH:
            if str(ref[k]) != str(m[k]):
                raise SystemExit(
                    f"REFUSING TO MERGE: {path} disagrees on {k!r} "
                    f"({m[k]} vs {ref[k]}). Chunks must share one base "
                    f"controller; merging different ones trains the clone on a "
                    f"mixture and makes the fidelity gate meaningless."
                )
    # Distinct offsets are NOT enough: episode seeds run [offset, offset+n), so
    # offsets 0 and 100 with n=200 each overlap on 100 identical episodes while
    # passing a mere uniqueness check.
    spans = sorted((int(m["episode_offset"]),
                    int(m["episode_offset"]) + int(m["n_episodes"]))
                   for m in metas)
    for (lo_a, hi_a), (lo_b, hi_b) in zip(spans, spans[1:]):
        if lo_b < hi_a:
            raise SystemExit(
                f"REFUSING TO MERGE: episode ranges overlap -- [{lo_a},{hi_a}) "
                f"and [{lo_b},{hi_b}) share {hi_a - lo_b} episodes. Offsets must "
                f"be spaced by at least --episodes.")
    widths = {f.shape[1] for f in feats}
    if len(widths) != 1:
        raise SystemExit(f"REFUSING TO MERGE: feature widths differ: {widths}")

    X, Y = np.vstack(feats), np.vstack(acts)
    n_ep = sum(int(m["n_episodes"]) - int(m["n_dropped"]) for m in metas)
    if n_ep == 0:
        raise SystemExit("REFUSING TO MERGE: every episode was dropped.")
    reached = sum(int(m["n_reached"]) for m in metas)
    out_meta = {k: ref[k] for k in MUST_MATCH}
    # `n_episodes` in the merged file means KEPT, and n_dropped is folded in --
    # recorded explicitly so the meaning change across the merge is not silent.
    out_meta.update({"n_episodes": n_ep, "n_dropped": 0,
                     "n_dropped_upstream": sum(int(m["n_dropped"]) for m in metas), "n_reached": reached,
                     "seed": ref["seed"], "grid": ref["grid"],
                     "episode_offset": np.array(sorted(offsets)),
                     "n_chunks": len(args.chunks)})
    np.savez(args.out, features=X, actions=Y,
             **{f"meta_{k}": np.asarray(v) for k, v in out_meta.items()})
    print(f"\nwrote {args.out}")
    print(f"  {X.shape[0]} rows x {X.shape[1]} features from {n_ep} episodes")
    print(f"  {reached}/{n_ep} episodes reached ({100 * reached / n_ep:.0f}%)")
    print(f"  distinct episode STARTS: {n_ep}  <- the quantity the clone was "
          f"short of; each episode contributes exactly one step-0 example")


if __name__ == "__main__":
    main()
