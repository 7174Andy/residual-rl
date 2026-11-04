"""Does each quadruped library stay near its anchor heading, and does reach follow it?

    .venv/bin/python scripts/quadruped_library_coverage.py --csv data/quadruped_deepc/rung1_t20n25_eval.csv
Prints each library's share of samples within pi/4 of its anchor, then (with --csv)
reached / far-miss (> 2.5 m) counts grouped by the start->goal bearing's nearest anchor.
"""
import argparse
import csv

import numpy as np

from quadruped import scenarios as sc
from quadruped.deepc_setup import DEFAULT_LIBRARIES


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--libraries", default=DEFAULT_LIBRARIES)
    p.add_argument("--scenarios", default=sc.SCENARIOS_PATH)
    p.add_argument("--csv", default=None)
    a = p.parse_args(argv)
    with np.load(a.libraries) as d:
        anchors = d["anchors"]
        yaws = [d[f"y_{i}"][:, 2] for i in range(len(anchors))]
    print("lib  anchor  in-region  mean|yaw-anchor|  yaw range")
    for i, (an, y) in enumerate(zip(anchors, yaws)):
        e = np.abs(_wrap(y - an))
        print(f"{i:3d}  {an:6.2f}  {np.mean(e <= np.pi / 4):9.2f}  {e.mean():16.2f}  [{y.min():.2f}, {y.max():.2f}]")
    if a.csv:
        s = sc.load(a.scenarios)
        tot = np.zeros((len(anchors), 3), int)  # reached, n, far misses
        for r in csv.DictReader(open(a.csv)):
            i = int(r["id"])
            st, g = s["state"][i], s["goal"][i]
            k = np.argmin(np.abs(_wrap(np.arctan2(g[1] - st[1], g[0] - st[0]) - anchors)))
            hit = r["reached"] == "True"
            tot[k] += (hit, 1, not hit and float(r["final_dist"]) > 2.5)
        print("bearing region  reached  far misses")
        for k, (h, n, far) in enumerate(tot):
            print(f"lib {k} ({anchors[k]:5.2f})   {h:3d}/{n:<3d}  {far}")


if __name__ == "__main__":
    main()
