#!/usr/bin/env python
"""Generate the committed per-seed traces behind the seed-showcase figures.

Writes full per-step (x, y, heading, v, w) closed-loop traces for one or more
showcase seeds, for both clone and residual, to
`<outdir>/traj_<seed>_{clone,residual}.csv`, for `scripts/plot_seed_traces.py`.
Both closed loops are DeePC-free (clone/residual only -- no QP), so this runs
in well under a second per seed. Defaults to `--trace-model
data/residual_td3.zip` (the 200k checkpoint) to match the checkpoint used to
record that seed's embedded video in docs/journey/10-residual-rl.md.
"""
from __future__ import annotations

import argparse

from two_wheel_robot.rl.showcase_trace import generate_trace_pair
from two_wheel_robot.rl.trace_io import clone_trace_path, residual_trace_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--outdir", default="docs/journey/figures")
    p.add_argument("--device", default="cpu")

    p.add_argument("--trace-model", default="data/residual_td3.zip")
    p.add_argument("--trace-seeds", default="4104626029,4104626034")
    args = p.parse_args()

    for seed in (int(s) for s in args.trace_seeds.split(",")):
        clone_trace, residual_trace = generate_trace_pair(
            seed, args.outdir,
            clone_path=args.clone, residual_model_path=args.trace_model,
            libraries_path=args.libraries, device=args.device,
        )
        print(f"wrote {clone_trace_path(args.outdir, seed)}  "
              f"({len(clone_trace['step']) - 1} steps)")
        print(f"wrote {residual_trace_path(args.outdir, seed)}  "
              f"({len(residual_trace['step']) - 1} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
