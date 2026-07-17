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

# Watch a random policy in a pygame window
uv run python scripts/visualize_random.py --episodes 5 --seed 42

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
two_wheel_robot/
    env/            # Gymnasium env, dynamics, pygame rendering
    controllers/    # DeePC controller, Hankel matrices, offline data collection
    rl/             # imitation clone, TD3/SAC residual, SB3 wrappers
scripts/            # CLI entrypoints (visualize, collect data, train/run/eval)
tests/              # pytest
docs/               # mkdocs source (guides, API reference, decision log)
data/               # offline trajectory libraries, trained checkpoints (git-ignored)
```

`env/dynamics.py` has no Gym dependency and is usable standalone from controllers and tests.
`controllers/` is RL-library-agnostic; `rl/` is the only place that imports `stable_baselines3`.

## References

> Pai, C., Shang, X., Qian, J., & Zheng, Y. _Online Tracking with Predictions for Nonlinear
> Systems with Koopman Linear Embedding._ [arXiv:2603.07395](https://arxiv.org/abs/2603.07395).
>
> Coulson, J., Lygeros, J., & Dörfler, F. _Data-Enabled Predictive Control: In the Shallows of
> the DeePC._ [arXiv:1811.05890](https://arxiv.org/abs/1811.05890).
>
> RL + MPC residual architecture: [arXiv:2510.03354](https://arxiv.org/abs/2510.03354), Eq. 18.
