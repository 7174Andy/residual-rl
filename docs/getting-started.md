# Getting started

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). Python `>= 3.12`.

```bash
git clone https://github.com/7174Andy/two-wheel-exp
cd two-wheel-exp
uv sync
```

This installs runtime deps (`gymnasium`, `numpy`, `cvxpy`, `pygame`) and dev deps (`pytest`, docs tooling).

## Smoke-test the env

```bash
uv run pytest tests/
```

You should see all tests green (~70+ tests).

## Generate offline data for DeePC

The DeePC controller needs an offline `(u, y)` trajectory to build its Hankel matrices. Generate one library per heading quadrant (paper's setup) with **broad** PE bounds so the data includes stopping, pivoting, and the full action range:

```bash
uv run python scripts/collect_data.py \
    --v_min 0 --w_abs_max 1.5708 \
    --out data/libraries.npz
```

For paper-faithful narrow PE bounds (suitable for trajectory tracking, less so for goal-reaching), pass the narrow bounds explicitly — the bare defaults are now the **broad** bounds above:

```bash
uv run python scripts/collect_data.py \
    --v_min 10 --w_abs_max 0.5236 \
    --out data/libraries.npz   # v∈[10,20], w∈[-π/6,π/6]
```

The sample bounds are saved inside the `.npz` under `sample_bounds` so the run script reconstructs the env consistently.

## Run DeePC in closed-loop

```bash
uv run python scripts/run_deepc.py                       # 3 episodes, library switching on
uv run python scripts/run_deepc.py --episodes 5 --seed 42
```

By default the script uses **all 4 libraries** with orientation-keyed switching (see [library switching](controllers/library-switching.md)). Each episode prints:

```text
episode 1: REACHED   after 107 steps  return=  -11998.2  final_dist=0.43
  v: min=+0.572 max=+13.230 mean=+5.774 std=3.269
  w: min=-1.371 max=+1.571 mean=+0.497 std=0.927
  library usage: [0, 0, 17, 90]
```

The `library usage` row shows how many steps each of the 4 controllers was active.

To compare against the single-library version:

```bash
uv run python scripts/run_deepc.py --single_library 0
```

(Expect worse results — see [why single library fails](journey/05-library-switching.md).)

## Build the docs locally

```bash
uv run mkdocs serve
```

Open <http://127.0.0.1:8000>. Hot-reloads on file changes.
