# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Gymnasium environment and controller benchmark for a **kinematic unicycle (two-wheel) robot** navigating to a goal point in a continuous 2D workspace.

The underlying dynamics are adapted from Appendix D of Pai, Shang, Qian, Zheng, *"Online Tracking with Predictions for Nonlinear Systems with Koopman Linear Embedding"* (arXiv:2603.07395), but the task here is **point-to-point goal-reaching**, not the paper's heart-curve trajectory tracking. See `docs/superpowers/specs/` for design rationale.

Repo plan:

1. Gymnasium env for goal-reaching (`TwoWheelGoal-v0`).
2. Classical controller baselines — **DeePC** (data-EnablEd predictive control, Coulson/Lygeros/Dörfler 2019; same family as the paper's DDPC).
3. RL baselines via **stable-baselines3**.

The env is the product; controllers are baselines that consume it.

## Domain facts (do not re-derive these)

Discrete-time kinematic unicycle, `Δt = 0.025` s:

```
z_x,t+1 = z_x,t + Δt · cos(z_δ,t) · v_t
z_y,t+1 = z_y,t + Δt · sin(z_δ,t) · v_t
z_δ,t+1 = z_δ,t + Δt · w_t
```

- State `z = (z_x, z_y, z_δ)` — position and heading (radians, wrapped to `[-π, π]`).
- Action `u = (v, w)` — tangential and angular velocity.
- Default action bounds (broadened from paper): `v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`. Paper's bounds `v ∈ [10, 20]`, `w ∈ [-π/6, π/6]` are forward-only and unsuitable for point-to-point reaching; configurable via `action_bounds`.
- Workspace: continuous box, default `[-10, 10]²`. Position is wall-clipped to the box on every step (heading is not affected).

The unicycle is *not* globally Koopman-linearizable. DeePC built here should expect orientation-keyed local-linear data libraries (the paper's strategy).

## Stage cost / reward (DeePC form)

```
r_t = -(y_t - y_ref)ᵀ Q (y_t - y_ref) - u_tᵀ R u_t + reach_bonus · [reached this step]
```

where `y = (x, y, δ)` is the 3-D output and `y_ref = (g_x, g_y, 0)`. Defaults: `Q = diag(1, 1, 0)` (heading not penalized, but still in `y` so behavioral predictors see it), `R = 1.3e-3 · I₂` (paper value), `reach_bonus = 100`.

Termination uses position-only error (`‖p − g‖ < goal_tolerance`); heading is irrelevant to "reached".

## Planned layout (Option A — flat by role)

```
two_wheel_robot/
    env/
        __init__.py        # registers Gym ID "TwoWheelGoal-v0"
        dynamics.py        # pure unicycle step, no Gym deps
        env.py             # UnicycleGoalEnv(gym.Env)
        rendering.py       # PygameRenderer (handles "human" and "rgb_array")
    controllers/
        base.py            # Controller protocol: reset(obs), act(obs) -> u
        deepc.py           # DeePC data-driven predictive controller (CVXPY QP)
        data_collection.py # offline rollouts to build Hankel data libraries
    rl/
        train_sb3.py       # PPO / SAC training entrypoint
        wrappers.py        # observation normalization, action rescale, etc.
    eval/
        run.py             # CLI: pick a controller, run it, save metrics + plots
        metrics.py         # success rate, time-to-goal, control effort, etc.
configs/                   # YAML configs
scripts/                   # standalone entrypoints (e.g. visualize_random.py)
tests/                     # pytest
docs/superpowers/specs/    # design docs
```

Boundary rules:

- `env/dynamics.py` imports only `numpy` — usable from controllers and tests without Gym.
- `controllers/` is RL-library-agnostic. A controller implements the `Controller` protocol in `controllers/base.py`. Do not import `gym` from inside a controller.
- `rl/` is the only place that imports `stable_baselines3`.
- `eval/run.py` is the single CLI surface — runs any controller (classical or RL checkpoint) on the env and emits comparable metrics.

## Public env attributes (consumed by classical controllers)

`env.unwrapped` exposes:

- `state` — `(z_x, z_y, z_δ)` ndarray, current pose.
- `goal` — `(g_x, g_y)` ndarray.
- `step_idx` — int step counter.
- `last_action` — clipped action from the previous step (zeros after `reset`).
- `y` — DeePC output measurement, `(x, y, δ)`, dim 3. For this fully-observed unicycle, `y == state`.
- `y_ref` — DeePC reference, `(g_x, g_y, 0)`, dim 3.
- `Q`, `R` — cost matrices (3×3 and 2×2).

DeePC contract: `u = action_space` (dim 2), `y` (dim 3), no disturbance `e`. The controller maintains its own past-trajectory buffer `(u_ini, y_ini)` of length `T_ini` and Hankel matrices built from offline data; the env stays stateless from the controller's perspective beyond exposing `y` each step.

Predictive controllers should read these directly rather than reverse-engineering from the body-frame observation.

## Tooling

- Package manager: **uv** (lockfile committed: `uv.lock`). Python `>=3.12`.
- Add deps with `uv add <pkg>`; sync with `uv sync`.
- Run anything via `uv run <cmd>` so the project venv is used.

## Conventions

- **Numpy first.** Don't introduce torch outside `rl/`.
- **State vs observation.** Internal state `z` is world-frame; the agent observation is body-frame relative `(distance, sin(bearing_rel), cos(bearing_rel), v_prev, w_prev)` for RL generalization. Classical controllers should pull from `env.unwrapped`, not the observation.
- **Determinism.** Seed via `reset(seed=...)`; controllers that need randomness take an explicit `rng: np.random.Generator`.
- **Angles in radians**, wrapped to `[-π, π]` in stored state.
- **Units.** Positions in workspace units; velocities in units/s.
