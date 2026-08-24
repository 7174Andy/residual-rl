#!/usr/bin/env python
"""Plot the Panda's seven joints: labelled side view + travel limits.

    uv run python scripts/plot_panda_joints.py

A reference figure for `docs/reference/mujoco-primer.md`. Every number is read
live out of `MjModel`/`MjData` at the `home` keyframe rather than transcribed, so
the figure cannot drift from `panda_nohand.xml` after a `robot_descriptions`
upgrade -- if the model changes, re-running this is the whole update.

Two things the readout makes visible that the XML does not:

* `model.jnt_axis` is all `[0, 0, 1]` -- it is expressed in each joint's own body
  frame, where every Panda joint is a Z-rotation. The world-frame axes drawn here
  come from `data.xaxis`, i.e. AFTER `mj_forward` resolves the body transforms.
  They are properties of the pose, true at `home` and nowhere else.
* `joint4` and `joint6` do not straddle zero (the elbow cannot straighten), so
  the +-symmetric limits one would assume for a 7-DoF arm are wrong for two of
  the seven.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from panda.model import load_model, safe_box, tip_id  # noqa: E402

# Nested magnitude (the safe box sits inside the model limit), so: one hue, two
# steps -- reusing the ordinal ramp already established in plot_reach_rates.py
# rather than introducing a second blue. Re-validated for this figure:
#   node <dataviz>/scripts/validate_palette.js "#3987e5,#184f95" --mode light --ordinal
#   -> ALL CHECKS PASS (monotone L, dL >= 0.06, light end 3.54:1 on surface)
_LIMIT = "#3987e5"   # full model range
_SAFE = "#184f95"    # the box apply_delta clips into
_TIP = "#2f7d47"     # the tracked output site, kept distinct from the ramp
_INK = "#52514e"
_GRID = "#e1e0d9"
_GRAY = "#898781"

# Which world axis each joint turns about, at the home pose. Derived, not assumed
# -- see `_axis_kind`.
_YAW, _PITCH, _ROLL = "yaw", "pitch", "roll"


def _axis_kind(axis: np.ndarray) -> str:
    """Classify a world-frame joint axis for drawing.

    The side view is the world X-Z plane, so a Y axis points out of the screen and
    cannot be drawn as a line -- it gets the axis-toward-viewer glyph instead.
    """
    ax = np.abs(axis)
    if ax[2] > 0.9:
        return _YAW
    if ax[1] > 0.9:
        return _PITCH
    return _ROLL


def read_geometry() -> dict:
    """Joint names, world anchors/axes, limits and the tip site at `home`."""
    model, data = load_model()
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    lo, hi = safe_box(model)
    joints = []
    for i in range(model.njnt):
        joints.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
            "anchor": data.xanchor[i].copy(),
            "kind": _axis_kind(data.xaxis[i]),
            "range": model.jnt_range[i].copy(),
            "safe": (float(lo[i]), float(hi[i])),
            "home": float(model.key_qpos[0][i]),
        })
    tip = data.site_xpos[tip_id(model)].copy()

    # The side view is only honest if the arm is planar at this pose. If a model
    # update breaks that, an X-Z projection would quietly foreshorten the links
    # and every length in the figure would be wrong -- fail instead.
    off = max(abs(float(j["anchor"][1])) for j in joints) + abs(float(tip[1]))
    assert off < 1e-6, (
        f"joints are not in the y=0 plane at the home keyframe (max |y| = {off:.2e} m); "
        "the side view would be a foreshortened projection, not a true elevation"
    )
    return {"joints": joints, "tip": tip}


def _draw_side_view(ax, geo: dict) -> None:
    joints, tip = geo["joints"], geo["tip"]

    # Link path: base, then each distinct anchor in chain order, then the tip.
    path = [(0.0, 0.0)]
    for j in joints:
        xz = (float(j["anchor"][0]), float(j["anchor"][2]))
        if xz != path[-1]:
            path.append(xz)
    path.append((float(tip[0]), float(tip[2])))
    px, pz = zip(*path)

    ax.axhline(0.0, color=_GRID, lw=1.4, zorder=1)
    ax.plot(px, pz, "-", color=_LIMIT, lw=7, alpha=0.30, solid_capstyle="round", zorder=2)
    ax.plot(px, pz, "-", color=_SAFE, lw=1.6, solid_capstyle="round", zorder=3)

    # Labels collide two different ways: co-located pairs share an anchor exactly
    # (shoulder, wrist), and same-height neighbours sit only ~80 mm apart. Both are
    # solved by clustering on z and stacking within a cluster -- placing every label
    # at a fixed offset overprints three of the seven.
    zs = sorted({round(float(j["anchor"][2]), 3) for j in joints})
    rows: dict[float, list[str]] = {z: [] for z in zs}
    for j in joints:
        rows[round(float(j["anchor"][2]), 3)].append(j["name"])
    top_z = zs[-1]

    for j in joints:
        x, z = float(j["anchor"][0]), float(j["anchor"][2])
        row = rows[round(z, 3)]
        n = row.index(j["name"])

        if j["kind"] is _PITCH:            # axis out of the page
            ax.plot([x], [z], "o", ms=11, mfc="white", mec=_INK, mew=1.5, zorder=4)
            ax.plot([x], [z], "o", ms=3.2, color=_INK, zorder=5)
        elif j["kind"] is _YAW:            # axis vertical in view
            ax.plot([x, x], [z - 0.052, z + 0.052], ":", color=_INK, lw=1.2, zorder=4)
            ax.plot([x], [z], "o", ms=8, mfc="white", mec=_INK, mew=1.5, zorder=5)
        else:                              # axis horizontal in view
            ax.plot([x - 0.058, x + 0.058], [z, z], ":", color=_INK, lw=1.2, zorder=4)
            ax.plot([x], [z], "o", ms=8, mfc="white", mec=_INK, mew=1.5, zorder=5)

        # The topmost cluster has the chain arriving from its left, so labels there
        # stack upward into clear sky; lower clusters stack leftward.
        if round(z, 3) == top_z:
            off, ha = (0, 16 + 14 * n), "center"
        else:
            off, ha = (-16, 7 - 15 * n), "right"
        ax.annotate(f"{j['name']}  ({j['kind']})",
                    xy=(x, z), xytext=off,
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=8.5, color=_SAFE, fontweight="bold",
                    fontfamily="monospace")

    ax.plot([tip[0]], [tip[2]], "o", ms=8, color=_TIP, zorder=6)
    tip_y = 0.0 if abs(float(tip[1])) < 1e-9 else float(tip[1])
    ax.annotate(f"attachment_site\ny = ({tip[0]:.3f}, {tip_y:.0f}, {tip[2]:.3f})",
                xy=(tip[0], tip[2]), xytext=(14, -14), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=_TIP,
                fontfamily="monospace")

    ax.set_title("Side elevation at the $\\tt home$ keyframe (world X–Z plane)",
                 fontsize=10, color=_INK, pad=10)
    ax.set_xlabel("world x  [m]", fontsize=9, color=_INK)
    ax.set_ylabel("world z  [m]", fontsize=9, color=_INK)
    ax.set_aspect("equal")
    ax.set_xlim(-0.30, 0.90)
    ax.set_ylim(-0.06, 0.98)   # headroom for the top cluster's stacked labels
    ax.grid(True, color=_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_GRAY, labelsize=8)

    # The glyph encoding repeats across seven joints, so it needs a key.
    handles = [
        plt.Line2D([], [], ls=":", color=_INK, lw=1.2, marker="o", ms=7,
                   mfc="white", mec=_INK, label="yaw / roll — axis in view plane"),
        plt.Line2D([], [], ls="none", marker="o", ms=9, mfc="white", mec=_INK,
                   mew=1.5, label="pitch — axis out of screen (world Y)"),
        plt.Line2D([], [], ls="none", marker="o", ms=7, color=_TIP,
                   label="tracked output site (tip)"),
    ]
    # Lower left is the only reliably empty quadrant -- the arm occupies up-and-right.
    ax.legend(handles=handles, loc="lower left", fontsize=7.6, frameon=False,
              labelcolor=_INK, handletextpad=0.7)


def _draw_limits(ax, geo: dict) -> None:
    joints = geo["joints"]
    y = np.arange(len(joints))[::-1]     # joint1 on top

    for yi, j in zip(y, joints):
        r0, r1 = float(j["range"][0]), float(j["range"][1])
        s0, s1 = j["safe"]
        ax.barh(yi, r1 - r0, left=r0, height=0.52, color=_LIMIT, alpha=0.35,
                zorder=2, label="_")
        ax.barh(yi, s1 - s0, left=s0, height=0.30, color=_SAFE, zorder=3, label="_")
        ax.plot([j["home"]], [yi], marker="|", ms=13, mew=2.0, color=_INK, zorder=4)
        # The assumption this panel exists to break: that a 7-DoF arm's joints are
        # symmetric about zero. Keyed on the SAFE box rather than the model range,
        # because that is what the env can actually command -- joint6's raw range
        # clears zero by 0.018 rad while its commanded box starts at +0.359.
        # Written into whichever half of the row the bar leaves empty.
        if not (s0 <= 0.0 <= s1):
            right = s1 < 0.0
            ax.annotate("commanded box excludes 0", xy=(0.0, yi),
                        xytext=(9 if right else -9, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center",
                        fontsize=7.2, color=_GRAY, style="italic")

    ax.axvline(0.0, color=_GRAY, lw=1.0, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([j["name"] for j in joints], fontsize=8.5,
                       fontfamily="monospace", color=_SAFE)
    ticks = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"], fontsize=8)
    ax.set_xlim(-3.35, 4.0)
    ax.set_xlabel("joint angle  [rad]", fontsize=9, color=_INK)
    ax.set_title("Travel: model limit vs commanded box", fontsize=10, color=_INK, pad=10)
    ax.grid(True, axis="x", color=_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_GRAY, labelsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_LIMIT, alpha=0.35, label="model jnt_range"),
        plt.Rectangle((0, 0), 1, 1, color=_SAFE, label="safe box (10% trim/end)"),
        plt.Line2D([], [], ls="none", marker="|", ms=11, mew=2.0, color=_INK,
                   label="home keyframe"),
    ]
    # Below the axes: every in-axes corner is occupied by a bar on some row.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, fontsize=7.6, frameon=False, labelcolor=_INK,
              handletextpad=0.7, columnspacing=1.8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/reference/panda_joints.png")
    args = ap.parse_args()

    geo = read_geometry()
    fig, (ax_side, ax_lim) = plt.subplots(1, 2, figsize=(12.4, 5.4),
                                         gridspec_kw={"width_ratios": [1.25, 1]})
    _draw_side_view(ax_side, geo)
    _draw_limits(ax_lim, geo)
    fig.suptitle("Franka Panda (panda_nohand.xml) — 7 joints, world frame",
                 fontsize=12, color=_INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(f"wrote {args.out}")
    for j in geo["joints"]:
        print(f"  {j['name']}  {j['kind']:5s}  anchor=({j['anchor'][0]:.3f}, "
              f"{j['anchor'][2]:.3f})  home={j['home']:+.3f}")


if __name__ == "__main__":
    main()
