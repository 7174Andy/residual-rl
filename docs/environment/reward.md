# Reward & episode

## Stage cost (DeePC form)

The reward is the negative DeePC stage cost plus a reach bonus:

$$
r_t = -(y_t - y_{\text{ref}})^\top Q (y_t - y_{\text{ref}}) - u_t^\top R u_t + b \cdot \mathbb{1}[\text{reached this step}]
$$

with defaults:

- $y = (x, y, \delta)$ — full 3-D output.
- $y_{\text{ref}} = (g_x, g_y, 0)$ — goal position, heading don't-care by default.
- $Q = \mathrm{diag}(1, 1, 0)$ — heading is in $y$ for the behavioral predictor, but unpenalized in cost.
- $R = 1.3 \cdot 10^{-3} I_2$ — paper value.
- $b = 100$ — bonus on the step where the goal is reached.

The 3×3 `Q` matches the [paper](https://arxiv.org/abs/2603.07395)'s convention exactly. With the third diagonal at zero, position dominates the cost, but the predictor still sees heading.

## Termination vs truncation

- **`terminated = True`** when $\lVert (x, y) - (g_x, g_y) \rVert < \text{goal\_tolerance}$ (default `0.5`). Only the position part of the error matters.
- **`truncated = True`** at `step_idx == max_steps` (default `200`), and only if not already terminated. The env never reports both flags `True` in the same step.

## Reset

On `reset(seed, options)`:

1. RNG is seeded.
2. `state ← (x, y, δ)` sampled uniformly in `workspace_bounds × [-π, π]` (heading is wrapped on entry).
3. `goal ← (g_x, g_y)` sampled uniformly in `workspace_bounds`; resampled (up to 100 attempts) until `‖start − goal‖ ≥ min_start_goal_dist`.
4. `last_action ← (0, 0)`, `step_idx ← 0`.

`options` can override either field:

```python
env.reset(seed=0, options={"state": [0.0, 0.0, np.pi/4], "goal": [5.0, 5.0]})
```

Useful for reproducible evaluation and for [offline data collection](../controllers/data-collection.md), which places the goal far outside the workspace so termination never fires mid-trajectory.

## Step

`step(action)`:

1. Action is clipped to `action_bounds`. The clipped value becomes `last_action`.
2. Dynamics integrated forward one step. Position is wall-clipped; heading is wrapped.
3. Distance to goal computed; `reached = distance < goal_tolerance`.
4. Reward computed using the **post-step** state.
5. `terminated = reached`; `truncated = (not reached) and (step_idx >= max_steps)`.

## `info` dict

Every `step()` and `reset()` returns an `info` dict with:

| Key | Type | Notes |
|---|---|---|
| `state` | `ndarray (3,)` | copy of internal state |
| `goal` | `ndarray (2,)` | copy of goal |
| `y` | `ndarray (3,)` | DeePC output measurement (alias of state) |
| `y_ref` | `ndarray (3,)` | DeePC reference `(g_x, g_y, 0)` |
| `pos_error` | `ndarray (2,)` | `state[:2] - goal` |
| `distance` | `float` | `‖pos_error‖` |
| `action` | `ndarray (2,)` | clipped action that was just applied (zeros after reset) |
| `step_idx` | `int` | current step counter |
| `reached` | `bool` | whether goal-tolerance is satisfied this step |
