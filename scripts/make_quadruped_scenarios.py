"""Generate the frozen QuadrupedGoal-v0 scenario file."""
import argparse
import os

import numpy as np

from quadruped import scenarios as S


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=S.SCENARIOS_PATH)
    a = ap.parse_args()
    d = S.generate()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    S.save(a.out, d)
    dist = np.linalg.norm(d["state"][:, :2] - d["goal"], axis=1).mean()
    print(f"n={len(d['seed'])} checksum={S.checksum(d)} mean_start_goal_dist={dist:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
