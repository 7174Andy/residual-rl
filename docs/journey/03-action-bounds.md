# 03. Action bounds — paper-faithful vs broad

## Decision

Default to **broad bounds** for both runtime *and* offline data collection: `v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`. Paper bounds (`v ∈ [10, 20]`, `w ∈ [-π/6, π/6]`) remain a CLI override.

## Context

The paper uses `v ∈ [10, 20]`, `w ∈ [-π/6, π/6]` to sample its PE inputs during data collection. These bounds make sense for the paper's task — **trajectory tracking**, where the robot is always moving forward along a smooth curve at moderate speed.

For our goal-reaching task, the same bounds are pathological:

- **`v ≥ 10`**: the robot **cannot stop**. Once it gets close to the goal it overshoots.
- **`w ∈ [-π/6, π/6]`**: turn rate ≤ 30 °/s. A 180° turn takes 6 seconds = 240 steps. Episode length is 200. The robot can't reverse direction within an episode without using forward-curve maneuvering.

The first run with paper bounds showed it clearly: the controller **drove the robot into a wall and used wall-clipping as a free brake**. From the controller's perspective this was a perfectly rational way to minimize position cost — but it's not the kind of behavior we want.

## Considered

| Option | Status |
|---|---|
| Keep paper bounds for paper-faithfulness | Rejected — task isn't tracking. |
| Switch to broad bounds (`v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`) | Chosen as the default. |
| Allow reverse (`v ∈ [-20, 20]`) | Considered. Kinematically valid but a step away from the paper's "forward-only" robot. Not needed for goal-reaching. |
| Use paper bounds AND remove wall clipping (with a soft penalty for leaving the workspace) | Rejected — adds complexity, doesn't fix the underlying "can't stop" issue. |

## Outcome

- `scripts/collect_data.py` accepts `--v_min --v_max --w_abs_max` and saves the chosen bounds inside the `.npz` under key `sample_bounds`.
- `scripts/run_deepc.py` reads `sample_bounds` from the data file and constructs the env with matching `action_bounds` automatically. This guarantees DeePC stays inside its empirical data envelope (no extrapolation to unseen action regions).
- The env's `action_bounds` constructor parameter is fully configurable, so paper bounds are one flag away if needed for paper-reproduction work.

## Side effect — the wall-pinning bug as a feature

The original "drive into wall to stop" behavior is a real failure mode of forward-only action spaces on environments with wall clipping. It's worth flagging because it's a *system-level* bug (not a controller bug): the data-driven controller did the optimal thing given the constraints. Removing wall-clipping and using a soft penalty would change the failure mode but not solve the underlying mismatch between forward-only kinematics and a stationary goal.
