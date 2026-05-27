# Two-wheel goal-reaching Gym environment — design

Status: approved 2026-05-27.

## Task

Continuous 2D navigation: a kinematic unicycle starts at a random pose, must reach a randomly-placed goal point in a bounded workspace. Episode terminates on reaching the goal or times out.

This deviates from the paper (arXiv:2603.07395 Appendix D), which does heart-curve trajectory tracking. The dynamics are the same; the task is not.

## Dynamics

Kinematic unicycle, forward Euler, `Δt = 0.025` s:

```
z_{t+1} = z_t + Δt · [cos(δ_t) v_t, sin(δ_t) v_t, w_t]
```

`z = (x, y, δ) ∈ ℝ³`, `u = (v, w) ∈ ℝ²`. Heading wrapped to `[-π, π]` in stored state.

## Gym ID

`TwoWheelGoal-v0`, registered on `import two_wheel_robot.env`.

## Spaces

- **Action**: `Box(low=[0, -π/2], high=[20, π/2], dtype=float32)`. Broadened from paper's `[10, 20] × [-π/6, π/6]` so the robot can stop and pivot. Configurable via `action_bounds`.
- **Observation** (`Box`, shape `(5,)`, float32):
  `[distance_to_goal, sin(bearing_rel), cos(bearing_rel), v_prev, w_prev]`
  where `bearing_rel = atan2(g_y − y, g_x − x) − δ`, wrapped to `[-π, π]`.

Body-frame relative encoding makes the policy translation- and rotation-invariant.

## Reward

```
r_t = -(p_t - g)ᵀ Q (p_t - g) - u_tᵀ R u_t + reach_bonus · [reached this step]
```

Defaults: `Q = diag(1, 1)` (position-only; no heading penalty), `R = 1.3e-3 · I₂` (kept from paper), `reach_bonus = 100`.

## Episode

- `reset(seed, options)`:
  1. Sample `(x, y)` uniform in workspace `[-10, 10]²`, `δ` uniform in `[-π, π]`.
  2. Sample goal uniform in workspace; rejection-sample until `‖start − goal‖ ≥ min_start_goal_dist` (default `2.0`, max 100 attempts).
  3. `options = {"state": ..., "goal": ...}` overrides sampling for reproducible eval.
- `step(action)`:
  1. Clip action to bounds.
  2. Integrate dynamics; clip `(x, y)` to `workspace_bounds` (wall-like); wrap heading to `[-π, π]`.
  3. `terminated = True` iff `‖p − g‖ < goal_tolerance` (default `0.5`).
  4. `truncated = True` at `step_idx == max_steps` (default `200`) if not already terminated.

## Public attributes (for classical controllers)

`env.unwrapped` exposes:
- `state` — `(x, y, δ)` ndarray
- `goal` — `(g_x, g_y)` ndarray
- `step_idx` — int
- `last_action` — `(v, w)` ndarray (clipped action from the previous step; zeros after `reset`)

Predictive controllers should read these directly rather than parsing the body-frame observation.

## Layout

```
two_wheel_robot/env/
    __init__.py     # gym.register("TwoWheelGoal-v0", ...)
    dynamics.py     # step_unicycle, wrap_to_pi  ── pure numpy
    env.py          # UnicycleGoalEnv(gym.Env)
    rendering.py    # PygameRenderer (handles both "human" and "rgb_array")
```

## Deviation from the brainstorm

Brainstorm proposed matplotlib for `rgb_array`. Implementation uses pygame's offscreen `Surface` + `surfarray.array3d` for both modes — one renderer, one dependency, identical visuals across modes.

## Testing

Visualization smoke test (`scripts/visualize_random.py`) is the primary gate for this milestone. Unit tests (env_checker, dynamics, reset reproducibility) deferred to a follow-up.
