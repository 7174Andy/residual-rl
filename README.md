# two-wheel-exp

A [Gymnasium](https://gymnasium.farama.org/) environment and controller benchmark for a
**kinematic unicycle (two-wheel) robot** navigating to a goal point in a continuous 2D
workspace — plus a **DeePC** (data-driven predictive control) baseline and a **TD3 residual
RL** policy trained on top of it.

📖 **Full docs:** <https://7174andy.github.io/two-wheeled-experiment/> — install guide, API
reference, CLI reference, and the [decision log](https://7174andy.github.io/two-wheeled-experiment/prod/journey/)
behind every design choice (action bounds, DeePC formulation, library switching, imitation
learning, residual RL).

The dynamics are adapted from Appendix D of Pai, Shang, Qian, Zheng, _"Online Tracking with
Predictions for Nonlinear Systems with Koopman Linear Embedding"_ ([arXiv:2603.07395](https://arxiv.org/abs/2603.07395)),
but the task here is **point-to-point goal-reaching**, not the paper's heart-curve trajectory
tracking.

A second, structurally different environment — **`PandaReach-v0`**, a 7-DoF Franka Emika Panda
driving its end-effector to a 3-D goal in MuJoCo — exists to test whether that
DeePC → clone → residual pipeline *generalizes* past the unicycle, or is an artifact of its
particular structure. See the [MuJoCo primer](docs/reference/mujoco-primer.md).

## Install

Managed with [uv](https://docs.astral.sh/uv/). Python `>=3.12`.

```bash
git clone https://github.com/7174Andy/two-wheel-exp
cd two-wheel-exp
uv sync
```

## Quickstart

```bash
# Smoke-test the env
uv run pytest tests/

# Generate offline data and run the DeePC controller in closed loop
uv run python scripts/collect_data.py --v_min 0 --w_abs_max 1.5708 --out data/libraries.npz
uv run python scripts/run_deepc.py --episodes 5 --seed 42

# Run the trained TD3 residual policy (RL + MPC) over the frozen clone
uv run python scripts/run_residual.py --seeds 4104626029
```

See the [getting started guide](https://7174andy.github.io/two-wheeled-experiment/prod/getting-started/)
and [CLI reference](https://7174andy.github.io/two-wheeled-experiment/prod/reference/cli/) for every
script and flag.

## Repo layout

```text
core/               # system-agnostic: DeePC solver, Hankel matrices, trace/video IO
two_wheel_robot/    # the unicycle system
    env/            # Gymnasium env, dynamics, pygame rendering
    controllers/    # DeePC config, offline data collection
    rl/             # imitation clone, TD3/SAC residual, SB3 wrappers
panda/              # the Franka Panda system (MuJoCo)
    model.py        # load, safe joint box, FK, delta-control law, sampling
    env.py          # PandaReach-v0
    data_collection.py, deepc_setup.py, scenarios.py, eval.py, rendering.py
scripts/            # CLI entrypoints (visualize, collect data, train/run/eval/record)
tests/              # pytest
docs/               # mkdocs source (guides, API reference, decision log)
data/               # trajectory libraries, scenario sets, checkpoints (git-ignored)
```

`core/` is the dependency sink shared by both systems: it imports nothing from
`two_wheel_robot/` or `panda/`, and no `gymnasium`. One `DeePC` class serves both
robots — the unicycle keys its library switching on heading, the Panda on tip
azimuth, via a `key_fn` hook rather than a subclass.

`env/dynamics.py` and `panda/model.py` have no Gym dependency and are usable
standalone from controllers and tests. `controllers/` is RL-library-agnostic; `rl/`
is the only place that imports `stable_baselines3` or `torch`.

### Robot model

`PandaReach-v0` uses the **Franka Emika Panda** MJCF from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
— specifically `franka_emika_panda/panda_nohand.xml`, the no-gripper variant, which gives a
clean `nq = nv = nu = 7`. That model is licensed **Apache-2.0** (see
`franka_emika_panda/LICENSE` in menagerie) and is itself derived from Franka Emika's publicly
available [`franka_description`](https://github.com/frankaemika/franka_ros/tree/develop/franka_description)
URDF, converted to MJCF by the menagerie authors.

The model is **not vendored here**. [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py)
fetches menagerie into `~/.cache/robot_descriptions/` the first time anything imports the
Panda, so `uv sync` alone does not download it and the first run needs network access.

Every measured constant under `panda/` — joint ranges, the PD servo gains, workspace extents,
the 0.4% self-collision rate — is a property of menagerie revision
[`feadf76`](https://github.com/google-deepmind/mujoco_menagerie/commit/feadf76d42f8a2162426f7d226a3b539556b3bf5)
(2026-03-18). `uv run python scripts/mujoco_hello.py` reprints all of them, so a model
update can be *checked* rather than assumed.

## References

> Pai, C., Shang, X., Qian, J., & Zheng, Y. _Online Tracking with Predictions for Nonlinear
> Systems with Koopman Linear Embedding._ [arXiv:2603.07395](https://arxiv.org/abs/2603.07395).
>
> Coulson, J., Lygeros, J., & Dörfler, F. _Data-Enabled Predictive Control: In the Shallows of
> the DeePC._ [arXiv:1811.05890](https://arxiv.org/abs/1811.05890).
>
> RL + MPC residual architecture: [arXiv:2510.03354](https://arxiv.org/abs/2510.03354), Eq. 18.
