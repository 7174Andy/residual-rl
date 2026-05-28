# CLI reference

All scripts run via `uv run python …`. They share the same Python venv as the rest of the project.

## `scripts/visualize_random.py`

Runs the env with `env.action_space.sample()` as the policy. Opens a pygame window. Useful for smoke-checking the env after changes.

```bash
uv run python scripts/visualize_random.py
uv run python scripts/visualize_random.py --episodes 5 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 3 | Number of episodes |
| `--seed` | 0 | RNG seed for `env.reset(seed=)` |

## `scripts/collect_data.py`

Generates four offline `(u, y)` trajectories (one per heading quadrant) and saves them to `.npz` alongside the sample bounds used.

```bash
# Paper-faithful (narrow PE, v ∈ [10, 20], w ∈ [-π/6, π/6])
uv run python scripts/collect_data.py --out data/libraries.npz

# Broad (goal-reaching ready, v ∈ [0, 20], w ∈ [-π/2, π/2])
uv run python scripts/collect_data.py --v_min 0 --w_abs_max 1.5708 --out data/libraries.npz
```

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `data/libraries.npz` | Output path |
| `--T` | 1500 | Trajectory length |
| `--seed` | 42 | RNG seed |
| `--v_min` | 10.0 | Min tangential velocity for PE sampling |
| `--v_max` | 20.0 | Max tangential velocity for PE sampling |
| `--w_abs_max` | π/6 ≈ 0.524 | Absolute upper bound on `w`; sampled in `[-w_abs_max, w_abs_max]` |

**Saved keys**: `u_0, y_0, …, u_3, y_3` and `sample_bounds (shape (2, 2))`.

## `scripts/run_deepc.py`

Closed-loop DeePC on `TwoWheelGoal-v0` with pygame rendering. Default mode uses **all 4 libraries** with orientation-keyed switching.

```bash
uv run python scripts/run_deepc.py                          # 3 episodes, defaults
uv run python scripts/run_deepc.py --episodes 5 --seed 42
uv run python scripts/run_deepc.py --single_library 0       # disable switching
uv run python scripts/run_deepc.py --no_bearing_ref --Q_heading 0    # paper-faithful baseline (will fail)
```

| Flag | Default | Meaning |
|---|---|---|
| `--libraries` | `data/libraries.npz` | Path to saved offline data |
| `--single_library` | `None` (use switching) | If set to 0–3, use only that one library and skip switching |
| `--T_ini` | 5 | Past-window length |
| `--N` | 12 | Future / prediction horizon length |
| `--episodes` | 3 | Number of episodes |
| `--seed` | 0 | Per-episode seed offset |
| `--lambda_g` | 2.0 | L1 weight on `g` (paper default) |
| `--lambda_y` | 3e6 | L2 weight on `σ_y` (paper default) |
| `--Q_heading` | 1.0 | `Q[2,2]` weight on heading deviation; `0` reproduces paper Q |
| `--no_bearing_ref` | off | Use `y_ref = (g_x, g_y, 0)` instead of bearing-to-goal |

Per-episode output looks like:

```text
episode 1: REACHED   after 107 steps  return=  -11998.2  final_dist=0.43
  v: min=+0.572 max=+13.230 mean=+5.774 std=3.269
  w: min=-1.371 max=+1.571 mean=+0.497 std=0.927
  library usage: [0, 0, 17, 90]
```

The `library usage` line shows how many steps each library was active. Only present in switching mode.

## `tests/`

Pytest, no script wrapper:

```bash
uv run pytest tests/
uv run pytest tests/test_deepc.py -v
uv run pytest tests/ -k library_switch     # just switcher tests
```

Test count: ~80+ across env, controllers, wrappers, dynamics.
