# Library switching

Orientation-keyed library switching is built **into** `DeePC`: a single
controller holds `N` data libraries (one per heading region) and, each step,
feeds the library whose anchor heading is closest to the robot's current
heading into one cached QP. There is no separate switcher class.

This page is the **reference**. The "why" lives in two journey entries:

- [Journey 06 — why one library isn't enough](../journey/06-single-library-fails.md): bilinear dynamics defeat a single linear predictor.
- [Journey 07 — library switching](../journey/07-library-switching.md): how piecewise local linearization fixes it.

## How it works

`DeePC` holds `N` libraries' Hankel matrices and a single shared past-`(u, y)`
buffer. The Hankel matrices of the *active* library live in `cp.Parameter`s, so
switching libraries means swapping parameter values into the one compiled
problem — not recompiling. On every `act()`:

1. Read the robot's current heading from `y_current[heading_index]` (default index 2).
2. Compute the **closest anchor** by shortest signed angular distance: `argmin |wrap(heading − anchor_i)|`.
3. If the selected library changed since the previous step, clear the warm-start `g` (its columns indexed the previous library and are meaningless under the new one).
4. Write the selected library's `(Up, Uf, Yp, Yf)` into the Hankel parameters and solve.
5. Slide the single shared buffer with the applied `(u_t, y_current)`.

Selecting by *closest anchor* is equivalent to selecting by *quadrant* when anchors are quadrant midpoints (the paper's choice: `π/4, 3π/4, -3π/4, -π/4`). Closest-anchor degrades gracefully if anchors aren't midpoints.

With a single library (`len(libraries) == 1`), selection is trivially index 0 and `anchor_headings` / `heading_index` are not consulted — this is the plain single-library DeePC case.

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
from two_wheel_robot.controllers.deepc import DeePC
from two_wheel_robot.controllers.hankel import build_hankel

# One Hankel tuple per library
libraries = [
    build_hankel(data[f"u_{i}"], data[f"y_{i}"], T_ini=5, N=12)
    for i in range(4)
]
anchors = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]

controller = DeePC(libraries, anchor_headings=anchors, Q=Q, R=R,
                   T_ini=5, N=12, u_bounds=u_bounds)

controller.reset(env.unwrapped.y, u_initial=midpoint)
for _ in range(max_steps):
    u_t = controller.act(env.unwrapped.y, env.unwrapped.y_ref)
    env.step(u_t)
    # controller.last_library_idx tells you which library fired
```

`scripts/run_deepc.py` does exactly this by default. Use `--single_library N` to pass a one-element library list and see the single-library contrast.

## Diagnostic: library usage histogram

`scripts/run_deepc.py` prints library usage per episode:

```text
episode 1: REACHED   after  94 steps  return=-11208.0  final_dist=0.46
  library usage: [0, 0, 17, 77]
```

This episode started in quadrant Q3 (library 2, 17 steps), then switched to Q4 (library 3, 77 steps). Useful for sanity-checking that switching actually triggers — narrow `w` collections give episodes where the robot stays in one quadrant (`[0, 0, 200, 0]`-style usage) and switching becomes a no-op.

## Public API

Library switching is part of `DeePC` — see the [DeePC reference](deepc.md#public-api) for the full constructor (`libraries`, `anchor_headings`, `heading_index`) and the `last_library_idx` / `last_warm_started` diagnostics.
