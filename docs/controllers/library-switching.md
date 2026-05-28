# Library switching

`LibrarySwitchingDeePC` is a wrapper around `N` `DeePC` instances + a shared past-`(u, y)` buffer. Each step, it picks the instance whose anchor heading is closest to the robot's current heading and runs that controller's QP.

This page is the **reference**. The "why" lives in two journey entries:

- [Journey 06 — why one library isn't enough](../journey/06-single-library-fails.md): bilinear dynamics defeat a single linear predictor.
- [Journey 07 — library switching](../journey/07-library-switching.md): how piecewise local linearization fixes it.

## How it works

`LibrarySwitchingDeePC` wraps `N` `DeePC` instances plus a single shared past-`(u, y)` buffer. On every `act()`:

1. Read the robot's current heading from `y_current[heading_index]` (default index 2).
2. Compute the **closest anchor** by shortest signed angular distance: `argmin |wrap(heading − anchor_i)|`.
3. Copy the shared buffer into the chosen `DeePC`'s buffer.
4. Run that controller's QP via its own `act()`.
5. Copy the updated buffer back into the shared one.

Selecting by *closest anchor* is equivalent to selecting by *quadrant* when anchors are quadrant midpoints (the paper's choice: `π/4, 3π/4, -3π/4, -π/4`). Closest-anchor degrades gracefully if anchors aren't midpoints.

## Anchors

The four libraries map to four heading quadrants:

| Library | Anchor (wrapped to `[-π, π]`) | Quadrant |
|---|---|---|
| 0 | `π/4` | `[0, π/2)` |
| 1 | `3π/4` | `[π/2, π)` |
| 2 | `-3π/4` (was `5π/4`) | `[-π, -π/2)` |
| 3 | `-π/4` (was `7π/4`) | `[-π/2, 0)` |

These come from `controllers/data_collection.PAPER_INIT_HEADINGS` after the env's `reset()` wraps headings to `[-π, π]`.

## Usage

```python
import numpy as np
from two_wheel_robot.controllers.deepc import DeePC, LibrarySwitchingDeePC
from two_wheel_robot.controllers.hankel import build_hankel

# Build 4 DeePCs, one per library
controllers = []
for i in range(4):
    Up, Uf, Yp, Yf = build_hankel(data[f"u_{i}"], data[f"y_{i}"], T_ini=5, N=12)
    controllers.append(DeePC(Up, Uf, Yp, Yf, Q=Q, R=R, T_ini=5, N=12, u_bounds=u_bounds))

# Wrap with the switcher
anchors = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]
switcher = LibrarySwitchingDeePC(controllers, anchors)

switcher.reset(env.unwrapped.y, u_initial=midpoint)
for _ in range(max_steps):
    u_t = switcher.act(env.unwrapped.y, env.unwrapped.y_ref)
    env.step(u_t)
    # switcher.last_library_idx tells you which one fired
```

`scripts/run_deepc.py` does exactly this by default. Use `--single_library N` to fall back to one library and see the contrast.

## Diagnostic: library usage histogram

`scripts/run_deepc.py` prints library usage per episode:

```text
episode 1: REACHED   after 107 steps  return=-11998.2  final_dist=0.43
  library usage: [0, 0, 17, 90]
```

This episode started in quadrant Q3 (library 3, 17 steps), then switched to Q2 (library 2, 90 steps). Useful for sanity-checking that switching actually triggers — narrow `w` collections give episodes where the robot stays in one quadrant (`[0, 0, 200, 0]`-style usage) and switching becomes a no-op.

## Public API

::: two_wheel_robot.controllers.deepc.LibrarySwitchingDeePC
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - reset
        - act
