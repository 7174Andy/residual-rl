# CLI reference

All scripts run via `uv run python …`. They share the same Python venv as the rest of the project.

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
| `--v_min` | 0.0 | Min tangential velocity for PE sampling (paper: 10) |
| `--v_max` | 20.0 | Max tangential velocity for PE sampling |
| `--w_abs_max` | π/2 ≈ 1.571 | Absolute upper bound on `w`; sampled in `[-w_abs_max, w_abs_max]` (paper: π/6) |

**Saved keys**: `u_0, y_0, …, u_3, y_3` and `sample_bounds (shape (2, 2))`.

## `scripts/run_deepc.py`

Closed-loop DeePC on `TwoWheelGoal-v0` with pygame rendering. Default mode uses **all 4 libraries** with orientation-keyed switching.

```bash
uv run python scripts/run_deepc.py                          # 3 episodes, defaults
uv run python scripts/run_deepc.py --episodes 5 --seed 42
uv run python scripts/run_deepc.py --single_library 0       # disable switching
uv run python scripts/run_deepc.py --no_bearing_ref --Q_heading 0    # heading-don't-care baseline (will fail)
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
| `--Q_heading` | 2.0 | `Q[2,2]` weight on heading deviation; default `2.0` matches the paper's `Q_z = diag(1,1,2)`, `0` gives a heading-don't-care `Q` |
| `--no_bearing_ref` | off | Use `y_ref = (g_x, g_y, 0)` instead of bearing-to-goal |

Per-episode output looks like:

```text
episode 1: REACHED   after 107 steps  return=  -11998.2  final_dist=0.43
  v: min=+0.572 max=+13.230 mean=+5.774 std=3.269
  w: min=-1.371 max=+1.571 mean=+0.497 std=0.927
  library usage: [0, 0, 17, 90]
```

The `library usage` line shows how many steps each library was active. Only present in switching mode.

## `scripts/gen_clone_data.py`

Generates the hybrid clone-training dataset (synthetic sampled inputs + on-policy DeePC rollouts) and caches it to `.npz`.

```bash
uv run python scripts/gen_clone_data.py --out data/clone_dataset.npz \
    --n_synthetic 20000 --p_degenerate 0.25 --n_onpolicy 100 --seed 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `data/clone_dataset.npz` | Output path |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries used to build the canonical config |
| `--n_synthetic` | 20000 | Number of synthetic (random-buffer) samples |
| `--p_degenerate` | 0.25 | Fraction of synthetic samples drawn from degenerate (near-zero-velocity) buffers |
| `--n_onpolicy` | 100 | Number of on-policy DeePC episodes to roll out for additional samples |
| `--seed` | 0 | RNG seed |

Prints sample count, feature dim, and per-regime counts, e.g. `wrote data/clone_dataset.npz: 28345 samples, dim 40` / `regimes: {'synthetic': 20000, 'onpolicy': 8345}`.

## `scripts/train_clone.py`

Trains the imitation-learning clone MLP (`f_θ`) on a generated dataset and saves the checkpoint.

```bash
uv run python scripts/train_clone.py --data data/clone_dataset.npz \
    --out data/clone.pt --device auto
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/clone_dataset.npz` | Input dataset from `gen_clone_data.py` |
| `--out` | `data/clone.pt` | Checkpoint output path |
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 512 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--device` | `auto` | `cuda`→`mps`→`cpu` fallback (`rl/device.py`) |
| `--seed` | 0 | RNG seed |

Prints final val MSE, e.g. `trained on 28345 samples (dim 40); final val MSE 0.00412 over 200 epochs -> data/clone.pt`.

## `scripts/validate_clone.py`

Runs the layered fidelity gate on a trained clone: (1) regime-conditioned open-loop regression on the held-out split, (2)/(3) closed-loop paired outcomes vs DeePC over a seed sweep.

```bash
uv run python scripts/validate_clone.py --clone data/clone.pt \
    --data data/clone_dataset.npz --n_seeds 78 --base_seed 4104626029
```

| Flag | Default | Meaning |
|---|---|---|
| `--clone` | `data/clone.pt` | Trained clone checkpoint |
| `--data` | `data/clone_dataset.npz` | Dataset (used for the held-out regression split) |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries |
| `--n_seeds` | 78 | Number of seeds in the closed-loop sweep |
| `--base_seed` | 4104626029 | First seed; sweep uses `base_seed .. base_seed+n_seeds-1` |
| `--device` | `auto` | Device for the clone |

Prints regression-by-regime MAE/RMSE, then the confusion matrix, agreement rate, McNemar p-value, DeePC/clone reach rates with Wilson CIs, and trajectory-deviation stats. The gate passes only if trajectory deviation is bounded **and** agreement is high — reach-rate parity alone isn't sufficient.

## `scripts/run_clone.py`

Closed-loop run of the trained clone — the mirror of `run_deepc.py`, with the QP replaced by the amortized NN clone. Loads the canonical config so action bounds/anchors/buffer priming match how the clone was trained and validated.

```bash
# live pygame window
uv run python scripts/run_clone.py --episodes 3 --seed 0

# record MP4s headless (no display needed)
uv run python scripts/run_clone.py --record videos/clone --seeds 4104626029,4104626034

# aggregate performance, no window
uv run python scripts/run_clone.py --headless --episodes 78 --seed 4104626029
```

| Flag | Default | Meaning |
|---|---|---|
| `--clone` | `data/clone.pt` | Trained clone checkpoint |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries (canonical config only) |
| `--device` | `auto` | Device for the clone |
| `--episodes` | 3 | Number of episodes (base seed + episode index) |
| `--seed` | 0 | Base seed |
| `--seeds` | `None` | Comma-separated exact seeds; overrides `--episodes`/`--seed` |
| `--random` | off | Draw the base seed from OS entropy (printed for reproducibility) |
| `--headless` | off | No pygame window; run as fast as possible (for aggregate metrics) |
| `--record` | `None` | Record each episode to `DIR/episode_<seed>.mp4` (forces `rgb_array`) |

Per-episode output mirrors `run_deepc.py`'s (`seed N: REACHED after S steps return=... final_dist=...`, plus `v`/`w` mean±std), with a final `reached: k/n` summary line across all episodes run.

## `scripts/train_residual.py`

Trains the TD3 (default) or SAC residual over the frozen clone — RL correction on top of the imitation-learned base policy (`two_wheel_robot.rl.residual_env.ResidualDeePCEnv`).

```bash
uv run python scripts/train_residual.py --clone data/clone.pt \
    --out data/residual_td3.zip --timesteps 200000
```

| Flag | Default | Meaning |
|---|---|---|
| `--clone` | `data/clone.pt` | Frozen clone checkpoint the residual corrects |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries (canonical config) |
| `--out` | `data/residual_td3.zip` | SB3 model output path |
| `--algo` | `td3` | `td3` or `sac` (SAC as a fallback for the hard collapse regime) |
| `--timesteps` | 200000 | Total training timesteps |
| `--residual-frac` | 1.0 | Scale of the residual correction added to the clone's base action |
| `--noise-sigma` | 0.1 | Action-noise sigma for exploration |
| `--lr` | 1e-3 | Learning rate |
| `--device` | `cpu` | Training device |
| `--seed` | 0 | RNG seed |
| `--monitor-out` | `None` | Persist per-episode returns to `<path>.monitor.csv` (feed to `plot_training_return.py --monitor`) |

Prints `saved -> <out>` on completion.

## `scripts/run_residual.py`

Closed-loop run of the clone+residual policy — mirrors `run_clone.py`, stepping a `ResidualDeePCEnv` driven by the trained TD3/SAC model.

```bash
uv run python scripts/run_residual.py --seeds 4104626029
uv run python scripts/run_residual.py --record docs/journey/videos --seeds 4104626061
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `data/residual_td3.zip` | Trained residual checkpoint |
| `--clone` | `data/clone.pt` | Frozen clone checkpoint |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries |
| `--seeds` | `"4104626029"` | Comma-separated seeds to run |
| `--residual-frac` | 1.0 | Residual-correction scale (must match training) |
| `--record` | `None` | Record each episode to `DIR/episode_<seed>.mp4` (else opens a live pygame window) |
| `--algo` | `td3` | `td3` or `sac`, must match the loaded checkpoint |
| `--device` | `cpu` | Inference device |

Per-seed output: `seed N: REACHED after S steps final_dist=D.DD`.

## `scripts/eval_residual.py`

Three-way benchmark: DeePC (QP) vs clone-only vs clone+residual, over the same seed sweep used everywhere else in the project.

```bash
uv run python scripts/eval_residual.py --model data/residual_td3.zip \
    --clone data/clone.pt --n_seeds 78 --base_seed 4104626029
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `data/residual_td3.zip` | Trained residual checkpoint |
| `--clone` | `data/clone.pt` | Frozen clone checkpoint |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries |
| `--n_seeds` | 78 | Sweep size |
| `--base_seed` | 4104626029 | First seed |
| `--residual-frac` | 1.0 | Residual-correction scale (must match training) |
| `--algo` | `td3` | `td3` or `sac` |
| `--device` | `cpu` | Inference device |

Prints DeePC/clone/residual reach counts + rates + Wilson CIs, residual-vs-clone McNemar p-value, `rescued`/`regressions` counts, and median trajectory deviation vs clone — this is the source of the reach-rate numbers in the [residual RL journey entry](../journey/08-residual-rl.md).

## `scripts/eval_seed_showcase.py`

Writes per-step closed-loop trace CSVs (`traj_<seed>_{clone,residual}.csv`) for showcase seeds, consumed by `plot_seed_traces.py` and `render_dashboard_video.py`. Both closed loops are DeePC-free (clone/residual inference only, no QP), so this runs in well under a second per seed.

```bash
uv run python scripts/eval_seed_showcase.py --trace-seeds 4104626029,4104626034
```

| Flag | Default | Meaning |
|---|---|---|
| `--clone` | `data/clone.pt` | Frozen clone checkpoint |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries |
| `--outdir` | `docs/journey/figures` | Output directory for the trace CSVs |
| `--device` | `cpu` | Inference device |
| `--trace-model` | `data/residual_td3.zip` | Residual checkpoint — defaults to the **200k** checkpoint, matching the video embedded in [journey 08](../journey/08-residual-rl.md) |
| `--trace-seeds` | `"4104626029,4104626034"` | Comma-separated seeds to trace |

Prints one `wrote <path> (N steps)` line per seed per controller.

## `scripts/render_dashboard_video.py`

Side-by-side clone vs clone+TD3-residual dashboard video: an animated trajectory panel plus `v(t)`/`w(t)`/cumulative-reward sparklines per column, at `fps=40` (matching `Δt=0.025s`). Reuses `eval_seed_showcase.py`'s trace cache, generating it on demand if a seed hasn't been showcased yet.

```bash
uv run python scripts/render_dashboard_video.py --seeds 4104626029,4104626034
```

| Flag | Default | Meaning |
|---|---|---|
| `--seeds` | *(required)* | Comma-separated seeds |
| `--clone` | `data/clone.pt` | Frozen clone checkpoint |
| `--residual-model` | `data/residual_td3.zip` | Residual checkpoint |
| `--algo` | `td3` | `td3` or `sac` |
| `--libraries` | `data/libraries_v0.npz` | DeePC offline data libraries |
| `--figdir` | `docs/journey/figures` | Trace-CSV cache directory (read/written) |
| `--outdir` | `docs/journey/videos` | Output directory for `dashboard-<seed>.mp4` |
| `--fps` | 40 | Playback frame rate |
| `--device` | `cpu` | Inference device |

## `scripts/plot_reach_rates.py`

Bar chart of DeePC vs clone vs clone+TD3-residual (200k, 400k) reach rates with Wilson 95% CIs. Default input is the committed benchmark CSV, so the figure regenerates without re-running the (QP-bound, minutes-per-seed) benchmark.

```bash
uv run python scripts/plot_reach_rates.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `docs/journey/figures/reach_rates.csv` | Input CSV (`label,k,n` columns) |
| `--out` | `docs/journey/figures/reach_rates.png` | Output PNG path |

Prints `wrote <out> (N bars)`.

## `scripts/plot_seed_traces.py`

Trajectory + forward-velocity (`v(t)`) companion figure for one showcase seed, from the trace CSVs `eval_seed_showcase.py` writes.

```bash
uv run python scripts/plot_seed_traces.py --seed 4104626029
```

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | *(required)* | Seed to plot |
| `--figdir` | `docs/journey/figures` | Directory containing `traj_<seed>_{clone,residual}.csv` |
| `--residual-label` | `"clone + TD3 (200k)"` | Legend label for the residual curve |
| `--residual-color` | `#3987e5` | Residual curve color |
| `--out` | `<figdir>/seed_<seed>_metrics.png` | Output PNG path |

Prints `wrote <out>`.

## `scripts/plot_training_return.py`

Plots the TD3 residual's training-return curve (mean episode return vs. episode). Default input is a committed curve CSV; `--monitor` derives the curve fresh from an SB3 Monitor CSV instead; `--compare` overlays multiple committed curves (used for the 200k-vs-400k comparison).

```bash
uv run python scripts/plot_training_return.py
uv run python scripts/plot_training_return.py \
    --compare 200k:docs/journey/figures/residual_return.csv \
              400k:docs/journey/figures/residual_return_400k.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `docs/journey/figures/residual_return.csv` | Committed curve CSV (`episode,ep_rew_mean`) |
| `--monitor` | `None` | SB3 Monitor CSV of raw returns; overrides `--input`, smoothed with a rolling mean |
| `--out` | `docs/journey/figures/residual_return.png` | Output PNG path |
| `--window` | 100 | Rolling-mean window when reading `--monitor` |
| `--save-curve` | `None` | Also write the (possibly `--monitor`-derived) curve to this CSV path, for committing |
| `--compare` | `None` | One or more `LABEL:PATH` pairs to overlay instead of a single curve |

Prints `wrote <out> (...)`.

## `tests/`

Pytest, no script wrapper:

```bash
uv run pytest tests/
uv run pytest tests/test_deepc.py -v
uv run pytest tests/ -k library_switch     # just switcher tests
```

Test count: 135 across env, controllers, dynamics, wrappers, imitation clone, and residual RL (`uv run pytest tests/ --collect-only -q` to recount as the suite grows). Tests marked `integration` need real checkpoints in `data/` (git-ignored) and are skipped otherwise.
