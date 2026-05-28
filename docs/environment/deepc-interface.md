# DeePC-compatible interface

The env exposes a fixed set of accessors that DeePC (and any other data-driven predictive controller) needs. The env is **stateless** from the controller's perspective — the controller maintains its own past-trajectory buffer; the env just hands over `y` each step.

## The contract

| Quantity | Symbol | Where it lives | Shape |
|---|---|---|---|
| Control input | $u$ | `env.action_space` (passed into `step`) | `(2,)` |
| Output measurement | $y$ | `env.unwrapped.y` | `(3,)` |
| Reference | $y_{\text{ref}}$ | `env.unwrapped.y_ref` | `(3,)` |
| Stage cost matrices | $Q$, $R$ | `env.unwrapped.Q`, `env.unwrapped.R` | `(3, 3)`, `(2, 2)` |
| Action bounds | $u_{\min}, u_{\max}$ | `env.unwrapped.action_bounds` | `(2, 2)` |
| No disturbance $e$ | — | — | — |

`y` and `y_ref` are dim 3 — the paper's choice — so the behavioral predictor sees heading even though `Q[2, 2] = 0` keeps it out of the runtime cost.

## What's *not* in this interface

- No past-trajectory buffer. DeePC manages its own.
- No Hankel matrices. Those are built once at controller construction (`controllers/hankel.py`) from offline data.
- No disturbance signal `e`. The unicycle goal-reaching task has no exogenous input; this is one of the differences from mixed-traffic Deep-LCC. See the [DeePC formulation journey entry](../journey/04-deepc-formulation.md).

## Reading and writing from a controller

```python
import gymnasium as gym
import two_wheel_robot.env  # noqa: F401

env = gym.make("TwoWheelGoal-v0", render_mode="human")
base = env.unwrapped

env.reset(seed=0)
controller.reset(base.y, u_initial=midpoint_of_action_bounds)

terminated = truncated = False
while not (terminated or truncated):
    u_t = controller.act(base.y, base.y_ref)   # read y, decide u
    _, _, terminated, truncated, _ = env.step(u_t)
```

Everything the controller needs is reachable from `env.unwrapped`. The body-frame `obs` returned by `step()` is for RL agents — DeePC ignores it.

## Reference handling at runtime

The default `y_ref` is `(g_x, g_y, 0)` — heading 0 is a don't-care because `Q[2, 2] = 0`. For goal-reaching with `Q[2, 2] > 0`, you typically want **bearing-aware** reference instead:

```python
dx = base.goal[0] - base.state[0]
dy = base.goal[1] - base.state[1]
bearing = float(np.arctan2(dy, dx))
y_ref_step = np.array([base.goal[0], base.goal[1], bearing])
u_t = controller.act(base.y, y_ref_step)
```

`scripts/run_deepc.py` does this by default. See the [bearing reference journey entry](../journey/08-bearing-reference.md) for why.
