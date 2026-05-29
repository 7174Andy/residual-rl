# 08. Stopping at the goal — the deceleration / overshoot problem

!!! warning "Open problem"
This entry records a **known, unresolved** failure mode rather than a settled
decision. No fix is committed yet; the candidate fixes below are notes for later.

## Problem (one line)

With forward-only speed bounds (`v_min > 0`), the controller drives the robot to
the goal's neighborhood but **cannot decelerate to land inside the tolerance** —
it skims past the goal circle and misses by a small margin, then loops back.

## Context

After [03 action bounds](03-action-bounds.md) and the bounds sweep, `data/libraries.npz`
is collected with **hybrid** PE bounds `v ∈ [10, 20]`, `w ∈ [±π/2]` (paper-style
forward-only `v`, but a fast turn rate). This removes the spin-in-place stall from
[06](06-single-library-fails.md): forcing `v ≥ 10` makes "do nothing" infeasible.

But the paper's task is continuous heart-curve _tracking_ — the robot never stops.
Ours is point-to-point _reaching_, which requires stopping within `goal_tolerance`
(`0.5`). Forcing `v ≥ 10` means the robot moves a minimum of `v_min · Δt = 10 · 0.025
= 0.25` units **every** step and can never slow down. On a tangential approach it
steps _over_ the 0.5-radius circle instead of landing in it.

## What it looks like

Closed-loop run on the hybrid data (6 seeds, `Q_heading=2`, bearing reference):

| seeds                                                                      | result |
| -------------------------------------------------------------------------- | ------ |
| reach the goal                                                             | 2 / 6  |
| **near-miss overshoot** (reached `min_dist` 0.68 / 0.88, then sailed past) | 2 / 6  |
| never converge                                                             | 2 / 6  |

Two observed symptoms, one root cause:

1. **`v` barely changes; only `w` does** — with `v_min=10` and a tiny control weight
   (`R = 1.3e-3`), cruising near minimum speed and steering with `w` is cost-optimal.
   Not a bug; it's the _cause_ of the next symptom.
2. **Misses the goal by ~1 cm** — the constant ~0.25 units/step stride can't be
   shortened, so the robot overshoots the tolerance circle.
