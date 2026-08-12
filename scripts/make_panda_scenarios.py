"""Generate and freeze the PandaReach-v0 evaluation scenario set.

    uv run python scripts/make_panda_scenarios.py

Writes data/panda_scenarios_v1.npz and prints its checksum. Run this ONCE. The
file is frozen the moment any result is recorded against it; regenerating means
bumping to v2 and accepting that cross-version numbers are not comparable.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from panda import scenarios as sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=sc.SCENARIOS_PATH)
    ap.add_argument("--n", type=int, default=sc.N_SCENARIOS)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file (invalidates recorded results)")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        try:
            existing = sc.load(args.out)
        except Exception as e:
            print(f"{args.out} already exists but is unreadable ({e!r}); "
                  "inspect or delete it, or pass --force.")
            return
        print(f"{args.out} already exists; checksum {sc.checksum(existing)}")
        print("Refusing to overwrite. Pass --force only if you accept that every "
              "already-recorded result against this file becomes incomparable.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = sc.generate(args.n)
    sc.save(args.out, data)
    print(f"wrote {args.out}  ({args.n} scenarios)")
    print(f"checksum {sc.checksum(data)}")
    d = np.linalg.norm(data["goal"], axis=1)
    print(f"goal radius {d.min():.3f} .. {d.max():.3f} m   (mean {d.mean():.3f})")


if __name__ == "__main__":
    main()
