# two-wheel-exp

A Gymnasium environment and controller benchmark for a **kinematic unicycle (two-wheel) robot** navigating to a goal in a continuous 2D workspace, with a data-driven predictive controller (DeePC) implementing orientation-keyed library switching.

The repo is built around three things:

1. **A clean Gym env** (`TwoWheelGoal-v0`) — goal-reaching, configurable, deterministic, well-tested.
2. **A DeePC controller** with the regularized hybrid form from [Pai, Shang, Qian, Zheng (arXiv:2603.07395)](https://arxiv.org/abs/2603.07395), built on top of the original DeePC formulation by [Coulson, Lygeros, Dörfler (arXiv:1811.05890)](https://arxiv.org/abs/1811.05890).
3. **A decision log** capturing the design tradeoffs — why goal-reaching instead of trajectory tracking, why broad action bounds, why library switching is essential, etc.

## What to read first

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Getting started](getting-started.md)__

    ---

    Install, generate offline data, watch a controller drive the robot in pygame.

-   :material-robot: __[Environment](environment/index.md)__

    ---

    State, action, observation, reward, episode mechanics, and the DeePC-compatible interface.

-   :material-cog-outline: __[Controllers](controllers/index.md)__

    ---

    DeePC, orientation-keyed library switching, and offline data collection.

-   :material-history: __[Journey](journey/index.md)__

    ---

    Decision log. Why each big choice was made, what was tried, what was ruled out.

</div>

## At a glance

- **System**: discrete-time kinematic unicycle, `Δt = 0.025` s.
- **Task**: random start, random goal in a continuous 2D workspace `[-10, 10]²`. Episode terminates on `‖p − g‖ < 0.5` or truncates at 200 steps.
- **Action**: `u = (v, w)`. Broad bounds `v ∈ [0, 20]`, `w ∈ [-π/2, π/2]` by default; paper-faithful bounds available.
- **Observation**: body-frame relative `(distance, sin(bearing_rel), cos(bearing_rel), v_prev, w_prev)` — for RL training.
- **DeePC output**: `y = (x, y, δ)` (dim 3); reference `y_ref = (g_x, g_y, 0)` (or bearing-aware at runtime). Q is 3×3.
- **Controllers**: `DeePC` (single library), `LibrarySwitchingDeePC` (4 libraries keyed on heading quadrant).

## Status snapshot

| Component | State |
|---|---|
| Gym env | done, 69 tests |
| DeePC controller (hybrid L1/L2) | done, 15 tests |
| Library switching | done, 8 tests |
| Visualization (`scripts/run_deepc.py`) | done, pygame renderer |
| RL baselines (SB3) | not yet — wrapper module in place |
