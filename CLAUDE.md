# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Gymnasium environment and controller benchmark for a **kinematic unicycle (two-wheel) robot** navigating to a goal point in a continuous 2D workspace.

The underlying dynamics are adapted from Appendix D of Pai, Shang, Qian, Zheng, *"Online Tracking with Predictions for Nonlinear Systems with Koopman Linear Embedding"* (arXiv:2603.07395), but the task here is **point-to-point goal-reaching**, not the paper's heart-curve trajectory tracking. Design rationale lives in `docs/journey/` (published decision log); `docs/superpowers/specs/` also has design notes but is local dev scratch (gitignored — not present in a fresh clone).

Repo status:

1. Gymnasium env for goal-reaching (`TwoWheelGoal-v0`) — done.
2. Classical controller baseline — **DeePC** (data-EnablEd predictive control, Coulson/Lygeros/Dörfler 2019; same family as the paper's DDPC) with orientation-keyed library switching — done.
3. Imitation-learned clone of DeePC (fast MLP surrogate) plus **TD3/SAC residual RL** correction on top, via stable-baselines3 — done; the residual lifts reach rate from ~39% (DeePC/clone) to 87–95% depending on training length (see `docs/journey/09-imitation-learning.md`, `docs/journey/10-residual-rl.md`).

The env is the product; controllers/policies are baselines that consume it.

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

where `y = (x, y, δ)` is the 3-D output and `y_ref = (g_x, g_y, 0)`. Defaults: `Q = diag(1, 1, 2)` (heading weighted, matching the paper's `Q_z` in arXiv:2603.07395 Appendix D; the full 3-D `y` also keeps heading visible to behavioral predictors), `R = 1.3e-3 · I₂` (paper value), `reach_bonus = 100`.

Termination uses position-only error (`‖p − g‖ < goal_tolerance`); heading is irrelevant to "reached".

## Repo layout (flat by role)

```
two_wheel_robot/
    env/
        __init__.py        # registers Gym ID "TwoWheelGoal-v0"
        dynamics.py         # pure unicycle step, no Gym deps
        env.py              # UnicycleGoalEnv(gym.Env)
        rendering.py        # PygameRenderer (handles "human" and "rgb_array")
    controllers/
        data_collection.py  # offline PE rollouts to build Hankel data libraries
        hankel.py           # build_hankel() past/future block-Hankel matrices
        deepc.py            # DeePC: orientation-keyed library-switching QP controller
    rl/
        clone.py             # imitation-learning clone MLP (train/save/load)
        clone_data.py         # hybrid synthetic + on-policy dataset generation
        clone_eval.py          # fidelity-gate stats (regression, McNemar, paired outcomes)
        deepc_setup.py          # canonical DeePC config shared by clone/residual code
        device.py                # cuda -> mps -> cpu device selection
        features.py               # featurize() for the clone
        residual_env.py            # ResidualDeePCEnv: clone + TD3/SAC residual correction
        residual_eval.py            # 3-way benchmark: DeePC vs clone vs clone+residual
        showcase_trace.py            # per-seed closed-loop trace cache generation
        trace_io.py                   # CSV read/write for traces (no gym/torch deps)
        trace_reward.py                 # recompute_reward() from a CSV trace
        train_sb3.py                     # TD3/SAC residual training entrypoint
        video_encoding.py                 # shared MP4 encoder (imageio)
        wrappers.py                        # rescale_action_symmetric() for SB3 compatibility
scripts/                   # 14 CLI entrypoints: data collection, DeePC/clone/residual run+train+eval, plotting, video
tests/                     # pytest
docs/superpowers/specs/    # local dev-scratch design notes (gitignored, not part of the published repo)
```

There is no `two_wheel_robot/eval/` package and no `configs/` directory. Evaluation/metrics logic lives in `rl/clone_eval.py`, `rl/residual_eval.py`, and `rl/trace_reward.py`; every script takes CLI flags (argparse, hardcoded defaults) instead of YAML configs.

Boundary rules:

- `env/dynamics.py` imports only `numpy` — usable from controllers and tests without Gym.
- `controllers/` is RL-library-agnostic and has no `gym` import. There's no formal `Controller` base class yet — `DeePC` (`controllers/deepc.py`) exposes `reset(y_initial, u_initial=None)` / `act(y_current, y_ref)` as its de facto contract.
- `rl/` is the only place that imports `stable_baselines3` (and the only place that imports `torch`, via the clone MLP).
- `scripts/*.py` is the CLI surface — each script is a standalone `argparse` entrypoint over some combination of the env, a controller, and/or a trained checkpoint. See `docs/reference/cli.md` for the full flag reference.

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
