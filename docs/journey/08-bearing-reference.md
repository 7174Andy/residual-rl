# 08. Bearing-aware reference + nonzero `Q[2,2]`

## Decision

For goal-reaching, set $y_{\text{ref}}[2] = \mathrm{atan2}(g_y - z_y, g_x - z_x)$ (bearing from robot to goal) and use $Q[2,2] = 1.0$ by default in `scripts/run_deepc.py`.

The env's underlying `y_ref` stays `(g_x, g_y, 0)` (paper-faithful, "heading don't-care"). The bearing override is a **controller-side strategy** applied at runtime.

## Context

The paper's `Q = diag(1, 1, 0)` means the **heading is in `y` but unpenalized in cost**. This works fine for trajectory tracking: the reference's `r_δ` is meaningful but doesn't directly drive cost; the position part of the cost drives the controller, and the heading evolves naturally as a consequence.

For goal-reaching, this fails:

1. With $Q[2,2] = 0$, the QP has **zero direct gradient on $w$**.
2. $R[1,1] = 1.3 \cdot 10^{-3}$ makes $w$ essentially free.
3. The QP saturates $w$ at the bound — bang-bang at $\pm w_{\max}$ — because that's the cheapest action that perturbs the (noisy) predicted position.

You see this in the action stats: `w` has `std = 0` for the entire episode, locked at the bound.

## Why bearing-aware reference helps

If we set $y_{\text{ref}}[2] = \text{bearing-to-goal}$ and give the QP a small heading cost ($Q[2,2] > 0$), the controller now has a **direct, well-grounded reason to turn**: heading deviation from the bearing costs reward.

As the robot moves, the bearing updates each step (it depends on the current robot position). The controller's "target heading" continuously tracks the direction of the goal. Combined with the position cost on `(x, y)`, this gives the QP enough information to both *turn toward the goal* and *drive forward toward it*.

## Considered

1. **Keep `Q[2,2] = 0`, `y_ref[2] = 0`** (paper-faithful). Fails on goal-reaching as described.
2. **`Q[2,2] = 0`, bearing-aware `y_ref[2]`**. No effect — `Q[2,2] = 0` means heading deviation has no cost.
3. **`Q[2,2] > 0`, `y_ref[2] = 0`**. Penalizes heading away from 0. Biases the robot to always face east; wrong for arbitrary goals.
4. **`Q[2,2] > 0`, bearing-aware `y_ref[2]`** (chosen). Penalizes heading away from the goal direction.

A tuning sweep over `Q_heading ∈ {0.05, 0.1, 0.3, 1.0}` showed `1.0` works well for our default scales. CLI flag `--Q_heading` lets you adjust.

## Outcome

The combined fix:

| Setting | Effect |
|---|---|
| `Q_heading = 1.0` | Heading deviation has noticeable cost gradient. |
| Bearing-aware `y_ref[2]` per step | Target heading points at the goal at all times. |
| Library switching (previous entry) | Each library's predictor is locally accurate for its quadrant. |

Together, these three changes are what makes the DeePC controller actually navigate to the goal. Each is necessary; none is sufficient on its own.

`scripts/run_deepc.py` enables both by default. Use `--no_bearing_ref` and `--Q_heading 0` to reproduce the paper-faithful (and broken-for-goal-reaching) baseline.

## Caveat

Bearing reference has a sign / wrap issue near the boundary: if the robot's heading is $+\pi - 0.1$ and the bearing is $-\pi + 0.1$, the shortest angular distance is $0.2$, but the cost sees $-2\pi + 0.2$ (a near-$2\pi$ error). Most of the time the bearing and the heading are close enough that this doesn't trigger, but it's a known edge case. A clean fix is to lift heading to $(\sin \delta, \cos \delta)$ representation — a path discussed but not taken (see also [env design entry](02-env-design.md)).
