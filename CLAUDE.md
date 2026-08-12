# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Gymnasium environment and controller benchmark for a **kinematic unicycle (two-wheel) robot** navigating to a goal point in a continuous 2D workspace.

The underlying dynamics are adapted from Appendix D of Pai, Shang, Qian, Zheng, *"Online Tracking with Predictions for Nonlinear Systems with Koopman Linear Embedding"* (arXiv:2603.07395), but the task here is **point-to-point goal-reaching**, not the paper's heart-curve trajectory tracking. Design rationale lives in `docs/journey/` (published decision log); `docs/superpowers/specs/` also has design notes but is local dev scratch (gitignored — not present in a fresh clone).

Repo status:

1. Gymnasium env for goal-reaching (`TwoWheelGoal-v0`) — done.
2. Classical controller baseline — **DeePC** (data-EnablEd predictive control, Coulson/Lygeros/Dörfler 2019; same family as the paper's DDPC) with orientation-keyed library switching — done.
3. Imitation-learned clone of DeePC (fast MLP surrogate) plus **TD3/SAC residual RL** correction on top, via stable-baselines3 — done; the residual lifts reach rate from ~39% (DeePC/clone) to 87–95% depending on training length (see `docs/journey/07-imitation-learning.md`, `docs/journey/08-residual-rl.md`).
4. A second, structurally different env — **`PandaReach-v0`**, a 7-DoF Franka Panda end-effector reaching env built on MuJoCo (`panda/`) — done, to test whether the DeePC → clone → residual pipeline generalizes past the unicycle. A DeePC-on-Panda spec is next; see [the MuJoCo primer](docs/reference/mujoco-primer.md).

The env is the product; controllers/policies are baselines that consume it. That statement now covers two, structurally different envs — don't assume the whole repo is 2-D, heading-based, or numpy-only-dynamics; `PandaReach-v0` is none of those.

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

## PandaReach-v0 domain facts (do not re-derive these either)

7-DoF Franka Panda (`panda_nohand.xml`, no floor geom), driven by MuJoCo, 50 Hz control.
Model source: `franka_emika_panda/panda_nohand.xml` from google-deepmind/mujoco_menagerie, Apache-2.0, derived from Franka Emika's `franka_description` URDF; fetched by `robot_descriptions` into `~/.cache/` (not vendored). All constants below were measured against menagerie revision `feadf76` (2026-03-18) — see `panda/model.py`'s docstring.


- Action `u = Δq ∈ [−0.2, 0.2]⁷` — a **delta** joint target, applied as `ctrl = clip(q_current + u, safe_box)` (`panda/model.py::apply_delta`). The PD-servo actuators treat `ctrl` as an absolute joint-angle target, not a torque. **The plant's true input is `ctrl`, not `u`** — the two differ whenever the box clip fires, which happens on a large fraction of steps under random or far-goal excitation (measured 24–48% depending on policy). A system-identification or DeePC stage on this env should collect and identify `ctrl → y`, not `delta → y`; `info["ctrl"]` from `env.step()` is what to record.
- `frame_skip = 10` physics steps (`opt.timestep = 0.002 s`) per control step — `dt_ctrl = 0.02 s`, i.e. 50 Hz.
- `y` = end-effector position `(x, y, z)`, 3-D, read from MuJoCo site `attachment_site`. `y_ref` = the goal position directly — no heading/bearing component, unlike the unicycle. **`y` is the reward's output and stays 3-D.**
- **`y` does not observe the state, and this is measured, not suspected.** A 7-DoF arm has a 4-D self-motion manifold: 132 sampled configuration pairs agree on tip position to <1 mm while sitting a median 3.59 rad apart in `q`, and because their Jacobians differ, driving both with an *identical* delta sequence separates their tips by a median 63 mm at a 12-step horizon — past the 50 mm `goal_tolerance` in 57% of pairs. So `(u_ini, y_ini)` maps **one-to-many** onto futures, violating the precondition Willems' lemma (hence DeePC) needs from the past window. This is *not* fixable by more data or finer library keying. Contrast the unicycle, where `y = (x, y, δ)` **is** the full state `z` — zero hidden dimensions. That asymmetry, not nonlinearity, is the main structural difference between the two systems.
- `y_ext` = `(tip, q_normalized)`, 10-D, is the additive remedy: `q` normalized to `[-1, 1]` over the safe box (raw radians against metres would make `λ_y‖σ_y‖²` sum incommensurable units). Tip occupies the first 3 components so `azimuth_key` works on it unchanged. `y_ref_ext` = `(goal, zeros(7))`. Paired with `Q = diag(I₃, 0₇)` in `deepc_setup(output="ext")`, so the tracking cost is **numerically identical** to tip-only — the extra outputs inform prediction through the `Yp`/`Yf` constraints without changing what is optimized. `y`/`y_ref`/`Q`/the reward are untouched; a controller opts in by reading `y_ext`.
- `Q = I₃`, `R = 1.0e-2 · I₇` — ratio-matched to the unicycle's control/state cost balance, not copied (`|u|` here is ~30× smaller, so copying the unicycle's `R` would make control effectively free).
- `goal_tolerance = 0.05` m, `max_steps = 150`, `reach_bonus = 100`.
- Goals are sampled by forward kinematics from a random valid (collision-free, above-floor) configuration, so every goal is guaranteed reachable — a Cartesian box sample would not guarantee this.
- Solvability bounds, with provenance — only the lower one is live: random actions reach **0/20** seeds (still reproducible via `scripts/record_panda_video.py`). A DLS-IK oracle once measured **0.90** over 60 seeds during design on 2026-08-10, establishing the task was achievable, before being removed as out-of-scope (not part of the QP → NN → residual pipeline under study); that number is not reproducible from this tree — it's preserved verbatim in `docs/superpowers/specs/2026-08-10-panda-reach-env-design.md`'s appendix if it ever needs re-measuring.

## Layout rules

There is no `two_wheel_robot/eval/` package and no `configs/` directory. Evaluation/metrics logic lives in `rl/clone_eval.py`, `rl/residual_eval.py`, and `rl/trace_reward.py`; every script takes CLI flags (argparse, hardcoded defaults) instead of YAML configs.

Boundary rules:

- `core/` imports nothing from `two_wheel_robot/` or `panda/` (or `gymnasium`) — it's the dependency sink shared by both systems' controllers.
- `env/dynamics.py` imports only `numpy` — usable from controllers and tests without Gym.
- `controllers/` is RL-library-agnostic and has no `gym` import. There's no formal `Controller` base class yet — `DeePC` (`core/deepc.py`) exposes `reset(y_initial, u_initial=None)` / `act(y_current, y_ref)` as its de facto contract.
- `rl/` is the only place that imports `stable_baselines3` (and the only place that imports `torch`, via the clone MLP).
- `panda/model.py` imports `mujoco` and `numpy` only, no `gymnasium` — mirrors the `env/dynamics.py` rule above.
- `scripts/*.py` is the CLI surface — each script is a standalone `argparse` entrypoint over some combination of the env, a controller, and/or a trained checkpoint. See `docs/reference/cli.md` for the full flag reference.

## Public env attributes (consumed by classical controllers)

This section is `TwoWheelGoal-v0`-specific (2-D action, 3-D `y` with a heading component). `PandaReach-v0` exposes the analogous `state`/`y`/`y_ref`/`tip_site_id`/`safe_box` accessors with different shapes — see `panda/env.py`'s own docstrings, not this section.

DeePC contract: `u = action_space` (dim 2), `y` (dim 3), no disturbance `e`. The controller maintains its own past-trajectory buffer `(u_ini, y_ini)` of length `T_ini` and Hankel matrices built from offline data; the env stays stateless from the controller's perspective beyond exposing `y` each step.

Predictive controllers should read `env.unwrapped` directly rather than reverse-engineering from the body-frame observation.

## Tooling

- `mujoco` and `robot-descriptions` are deps added for `panda/` only. `robot_descriptions` lazily shallow-clones `mujoco_menagerie` into `~/.cache/robot_descriptions/` the first time anything resolves the Panda model path, so the first run after a fresh clone needs network access; every call after reads from the cache.

## Conventions

- **Numpy first.** Don't introduce torch outside `rl/`.
- **State vs observation.** Internal state `z` is world-frame; the agent observation is body-frame relative `(distance, sin(bearing_rel), cos(bearing_rel), v_prev, w_prev)` for RL generalization. Classical controllers should pull from `env.unwrapped`, not the observation.
- **Determinism.** Seed via `reset(seed=...)`; controllers that need randomness take an explicit `rng: np.random.Generator`.
- **Angles in radians**, wrapped to `[-π, π]` in stored state.
- **Units.** Positions in workspace units; velocities in units/s.
