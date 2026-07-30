# Dynamics & spaces

## Continuous-time model

The kinematic unicycle:

$$
\dot{x} = v \cos(\delta), \quad \dot{y} = v \sin(\delta), \quad \dot{\delta} = w
$$

where $(x, y)$ is position, $\delta$ is heading (radians), $v$ is tangential velocity, $w$ is angular velocity.

## Discrete-time model

Forward Euler with $\Delta t = 0.025$ s:

$$
\begin{aligned}
x_{t+1} &= x_t + \Delta t \cdot \cos(\delta_t) \cdot v_t \\
y_{t+1} &= y_t + \Delta t \cdot \sin(\delta_t) \cdot v_t \\
\delta_{t+1} &= \mathrm{wrap}\bigl(\delta_t + \Delta t \cdot w_t\bigr)
\end{aligned}
$$

After the integration step, position is **wall-clipped** to `workspace_bounds` and heading is wrapped to $[-\pi, \pi]$. The pure-numpy `step_unicycle` is in `two_wheel_robot.env.dynamics` and has no Gym dependency, so controllers and notebooks can import it directly.

## State

Internal state $z \in \mathbb{R}^3$:

| Index | Symbol | Meaning | Units | Range |
|---|---|---|---|---|
| 0 | $z_x$ | x position | workspace units | `[xmin, xmax]` |
| 1 | $z_y$ | y position | workspace units | `[ymin, ymax]` |
| 2 | $z_\delta$ | heading | radians | wrapped to $[-\pi, \pi]$ |

Accessed via `env.unwrapped.state` (read/write copy).

## Goal

$g = (g_x, g_y) \in \mathbb{R}^2$ — sampled uniformly inside `workspace_bounds` on `reset()`, with rejection sampling to ensure `‖start − goal‖ ≥ min_start_goal_dist` (default `2.0`).

Accessed via `env.unwrapped.goal`.

## Action

$u = (v, w) \in \mathbb{R}^2$. Clipped to `action_bounds` on every `step()`.

| Index | Symbol | Units | Default bounds |
|---|---|---|---|
| 0 | $v$ | units/s (tangential) | `[0, 20]` |
| 1 | $w$ | rad/s (angular) | `[-\pi/2, \pi/2]` |

The clipped value is exposed via `env.unwrapped.last_action`. It's also what gets serialized into the post-step `info` dict and into the observation's `v_prev`/`w_prev` slots.

!!! note "Paper bounds"
    Appendix D of [arXiv:2603.07395](https://arxiv.org/abs/2603.07395) uses `v ∈ [10, 20]`, `w ∈ [-π/6, π/6]`. These work for trajectory tracking but make goal-reaching infeasible because the robot cannot stop.

## Observation

`Box(shape=(5,), dtype=float32)`, body-frame relative — rotation/translation invariant:

| Index | Component | Range |
|---|---|---|
| 0 | distance to goal | `[0, workspace_diagonal]` |
| 1 | $\sin(\text{bearing}_{\text{rel}})$ | `[-1, 1]` |
| 2 | $\cos(\text{bearing}_{\text{rel}})$ | `[-1, 1]` |
| 3 | $v_{\text{prev}}$ (last applied tangential velocity) | `[\min(0, v_{\min}), v_{\max}]` |
| 4 | $w_{\text{prev}}$ (last applied angular velocity) | `[\min(0, w_{\min}), w_{\max}]` |

with $\text{bearing}_{\text{rel}} = \mathrm{wrap}\bigl(\mathrm{atan2}(g_y - z_y,\ g_x - z_x) - z_\delta\bigr)$.

The `min(0, ·)` widening on the action components ensures that the post-`reset()` zero-init of `v_prev`/`w_prev` always lies inside the obs space, even when `action_bounds` excludes zero.

!!! info "State vs observation"
    `env.unwrapped.state` (world frame) and the returned `obs` (body frame) are different views of the system. RL agents consume `obs`; classical controllers consume `state` directly via `env.unwrapped`. The rationale lives in [journey 02 — env design](../journey/02-env-design.md).
