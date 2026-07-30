# Offline data collection

DeePC consumes one (or more) **offline trajectories** of length `T` per library. Each trajectory is a sequence of aligned `(u_t, y_t)` pairs collected by rolling out the env with **persistently exciting** (PE) inputs — random samples from a uniform distribution over a bounded action region.

## The function

```python
from two_wheel_robot.controllers.data_collection import collect_trajectory

u_traj, y_traj = collect_trajectory(
    env, T=1500,
    init_state=np.array([0.0, 0.0, np.pi/4]),
    rng=np.random.default_rng(0),
    sample_bounds=np.array([[0.0, 20.0], [-np.pi/2, np.pi/2]]),
)
# u_traj.shape == (T, 2)   — actually-applied (post-clip) actions
# y_traj.shape == (T, 3)   — y observed BEFORE each action is applied
```

The pair `(u_traj[t], y_traj[t])` is aligned: `y_traj[t]` is the system's output at the moment `u_traj[t]` is decided. So `y_traj[t+1]` is the response to applying `u_traj[t]` from `y_traj[t]`.

## What happens during collection

1. Env is reset with `options={"state": init_state, "goal": (100.0, 100.0)}`. Placing the goal far outside the workspace prevents accidental termination mid-trajectory.
2. At each step:
    - Record `y_traj[t] = env.unwrapped.y`.
    - Sample `u ~ Uniform(sample_bounds)`.
    - Apply: `env.step(u)`.
    - Record `u_traj[t] = env.unwrapped.last_action` (the *post-clip* action, so the data reflects what the dynamics actually saw).
3. The env's passive `obs-not-in-space` warning is suppressed inside the collector since we knowingly put the env in an out-of-spec state by placing the goal far away; the obs is never consumed during collection.

## Multi-library collection (the paper's setup)

```python
from two_wheel_robot.controllers.data_collection import (
    collect_libraries, paper_init_states, PAPER_SAMPLE_BOUNDS,
)

libraries = collect_libraries(
    env, T=1500,
    init_states=paper_init_states(),   # 4 states at headings π/4, 3π/4, 5π/4, 7π/4
    rng=np.random.default_rng(42),
    sample_bounds=PAPER_SAMPLE_BOUNDS, # v ∈ [10, 20], w ∈ [-π/6, π/6]
)
# libraries[i] = (u_traj_i, y_traj_i)
```

The CLI script `scripts/collect_data.py` wraps this and saves to `.npz`:

```bash
uv run python scripts/collect_data.py \
    --v_min 0 --w_abs_max 1.5708 \
    --out data/libraries.npz
```

The sample bounds are stored inside the file under key `sample_bounds`, so `scripts/run_deepc.py` reconstructs the env with matching bounds automatically.

## Choosing the PE bounds

Two canonical configurations:

- **`v ∈ [10, 20]`, `w ∈ [-π/6, π/6]`** — paper-faithful; for trajectory tracking.
- **`v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`** — broad; for goal-reaching (allows stop and pivot).

Pick based on the task.

## Hankel construction

Once `(u, y)` is in hand, build the past/future block-Hankels:

```python
from two_wheel_robot.controllers.hankel import build_hankel

Up, Uf, Yp, Yf = build_hankel(u_traj, y_traj, T_ini=5, N=12)
# Up: (T_ini · m_u, n_cols)
# Uf: (N · m_u, n_cols)
# Yp: (T_ini · p_y, n_cols)
# Yf: (N · p_y, n_cols)
# where n_cols = T - (T_ini + N) + 1
```

`build_hankel` is dimension-agnostic: it works for any `m_u`, `p_y`. There's no disturbance `e` in this implementation — the original Deep-LCC has `Ep` / `Ef` for the head-vehicle velocity, but that has no analog in the two-wheel goal-reaching task.

## API

::: two_wheel_robot.controllers.data_collection

::: two_wheel_robot.controllers.hankel
