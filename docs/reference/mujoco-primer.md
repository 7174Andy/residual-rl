# MuJoCo primer

A from-scratch walkthrough of the MuJoCo Python bindings, written against the exact model this repo drives: `panda_nohand.xml`, a 7-DoF Franka Panda with the hand removed, wrapped as `PandaReach-v0` in `panda/env.py`. If you've never touched MuJoCo before, start here — the rest of the reference docs assume you already know what `qpos` means; this page does not.

Every number below is copied from a real run of `scripts/mujoco_hello.py` or `scripts/record_panda_video.py`, or from a test-suite docstring cited inline. Nothing here is estimated — run the commands yourself as you read; each takes seconds.

!!! note "First run downloads the model"
    `panda/model.py`'s `model_path()` resolves through `robot_descriptions`, which shallow-clones `mujoco_menagerie` into `~/.cache/robot_descriptions/` the first time anything imports the Panda. That first call needs network access and takes a while; every call after is instant, reading from the cache. The model itself is never the reason a Panda test carries `@pytest.mark.integration` — this repo's `integration` marker means "loads real data files from `data/`" (declared-dependency assets like the Panda model don't count). As of this tree there are zero Panda tests marked `integration` (`uv run pytest --collect-only -m integration -q` collects 11, all pre-existing unicycle tests) — the two that used to carry it, pinning aggregate reach-rate statistics for the DLS-IK oracle, were removed with that oracle.

!!! info "Where the robot model comes from"

    Franka Emika Panda, MJCF from
    [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
    — `franka_emika_panda/panda_nohand.xml`, the no-gripper variant, which is why
    `nq = nv = nu = 7` with no extra finger joints to reason about.

    | | |
    |---|---|
    | license | **Apache-2.0** (`franka_emika_panda/LICENSE` in menagerie) |
    | derived from | Franka Emika's [`franka_description`](https://github.com/frankaemika/franka_ros/tree/develop/franka_description) URDF |
    | revision | menagerie [`feadf76`](https://github.com/google-deepmind/mujoco_menagerie/commit/feadf76d42f8a2162426f7d226a3b539556b3bf5) (2026-03-18) |
    | vendored? | no — `robot_descriptions` caches it under `~/.cache/` |

    The revision is worth pinning because **every number on this page is a property of
    that model revision**, not of MuJoCo or of Panda arms in general. If a menagerie
    update changes a joint range or a servo gain, `scripts/mujoco_hello.py` reprints
    everything and the difference shows up immediately — which is the whole reason that
    script is in the repo rather than having been a throwaway.

!!! info "Where the *dynamics* come from — and what to read"

    The link masses and inertias in `panda_nohand.xml` are **not** Franka's spec sheet.
    Traced back through the URDF this MJCF was converted from, they are identified
    parameters from a peer-reviewed paper. `franka_description`'s own header says so:

    > This file does not contain official inertial properties of panda robot. The values
    > are from the identification results published in: [...] by Claudio Gaz, Marco
    > Cognetti, Alexander Oliva, Paolo Robuffo Giordano, Alessandro de Luca
    > — [`franka_description/robots/common/inertial.yaml`](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/inertial.yaml)

    The link-1..7 masses in that file (4.970684, 0.646926, 3.228604, 3.587895, 1.225946,
    1.666555, 0.735522 kg) appear verbatim in `panda_nohand.xml` — those seven links
    *are* the no-hand arm, so the citation covers exactly this model.

    **Primary source.** C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca,
    *"Dynamic Identification of the Franka Emika Panda Robot With Retrieval of Feasible
    Parameters Using Penalty-Based Optimization"*, IEEE RA-L 4(4):4147–4154, 2019.
    It gives the DH table, the identified dynamic coefficients, a joint friction model,
    and — the part that makes it simulator-usable rather than regressor-only — a
    *physically feasible* (mass, CoM, inertia) set.

    | | |
    |---|---|
    | paper page + **errata** | [diag.uniroma1.it/gaz/panda2019.html](https://www.diag.uniroma1.it/gaz/panda2019.html) — note φ₁ for joint 2 is **0.87224**, not 8722.4 |
    | open PDF | [hal.science/hal-02265293](https://hal.science/hal-02265293/) |
    | code — `M(q)`, `C(q,q̇)`, `g(q)`, friction | [github.com/marcocognetti/FrankaEmikaPandaDynModel](https://github.com/marcocognetti/FrankaEmikaPandaDynModel) (MATLAB, C++, V-REP) |
    | official kinematics / limits | [Franka Control Interface docs](https://frankarobotics.github.io/docs/) |

    Model form: `M(q)q̈ + C(q,q̇)q̇ + g(q) + τ_f(q̇) = τ`. The DH constants from that
    table are readable straight off this MJCF's body offsets — `d₁ = 0.333`,
    `d₃ = 0.316`, `d₅ = 0.384`, `a₄ = 0.0825`, `a₅ = −0.0825`, `a₇ = 0.088`, flange
    `0.107` — which is a fast way to confirm you're looking at the same robot the paper
    identified.

!!! warning "That model is *underneath* this env, not equal to it"

    Two gaps matter before any of the above gets used to reason about `PandaReach-v0`:

    - **MuJoCo doesn't integrate the equation above.** It solves its own constrained
      forward dynamics ([computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html)),
      and menagerie added `armature="0.1"` and `damping="1"` on every joint for solver
      stability. Those are **not identified values** — they are simulation terms with no
      counterpart in Gaz et al., and they change the `ctrl → y` response measurably.
    - **You never command τ.** Section 6: the actuators are PD position servos,
      `τ = kp·(ctrl − qpos) − kd·qvel`. The identified rigid-body dynamics sit *inside* a
      servo loop, so the plant this repo identifies is `ctrl → y`, not `τ → y`. That is
      the same reason `panda/data_collection.py` records `info["ctrl"]`.

## 1. `MjModel` vs `MjData`

The one distinction that unlocks everything else: **`MjModel` is the robot, `MjData` is the moment.**

- `MjModel` is compiled once from the MJCF XML (`panda_nohand.xml`) and is, for practical purposes, immutable: joint ranges, actuator gains, body masses, the whole kinematic tree. Many simulations could share one `MjModel`.
- `MjData` is the mutable, per-simulation state that changes every step: `qpos`, `qvel`, `ctrl`, and everything derived from them (`site_xpos`, `xpos`, the contact list...). One `MjData` per running episode.

`panda/model.py`'s `load_model()` returns both:

```python
def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(model_path())
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.8
    return model, mujoco.MjData(model)
```

The `headlight` lines are **model**-level edits — they change how *every* `MjData` built from this `model` renders, and can never affect a rollout, because `vis` doesn't participate in dynamics. Compare that to `data.qpos[:] = q0` in `env.reset()`: that's state, thrown away and rebuilt every episode.

Confirm the split yourself:

```bash
uv run python scripts/mujoco_hello.py
```

```
model: /Users/andrewpark/.cache/robot_descriptions/mujoco_menagerie/franka_emika_panda/panda_nohand.xml

nq=7 nv=7 nu=7 na=0 nbody=10 nsite=1 nkey=1
opt.timestep=0.002  frame_skip@50Hz=10
```

Everything on that `nq=...` line is a property of `model` — compiled in, constant for the process's lifetime. None of it says where the arm currently is; for that you need `data`.

## 2. `qpos`, `qvel`, `ctrl`

- `data.qpos` — generalized position, length `model.nq`.
- `data.qvel` — generalized velocity, length `model.nv`.
- `data.ctrl` — actuator command, length `model.nu`.

On this model all three are 7 (`nq=7 nv=7 nu=7`), because every one of the Panda's joints is a plain hinge: one number of position, one number of velocity, one command each. `nq` and `nv` only diverge when a joint needs more numbers to store its position than its velocity — the standard case is a **freejoint** on a floating body, whose orientation is a 4-number quaternion in `qpos` but only a 3-number angular velocity in `qvel`, so `nq = nv + 1`. Nothing in this model has one; that's why they match here, not because they always do.

`ctrl` looks the most familiar of the three and is the one most worth pausing on: what a number written into `ctrl[i]` *means* is entirely decided by actuator `i`'s type (section 6). For this model, `ctrl` is an **absolute joint-angle target**, not a torque or a velocity — `panda/model.py`'s `apply_delta` writes `ctrl = clip(qpos + delta, safe_box)`, which only makes sense if `ctrl` is a position.

## 3. `mj_step` vs `mj_forward` — the stale-kinematics trap

This is the single easiest way to silently break a MuJoCo env, and it's worth internalizing before writing a line of your own code.

`mujoco.mj_step` does two things in order: it computes forces (as if it had called `mj_forward`), then it integrates `qpos`/`qvel` forward by `opt.timestep`. It does **not** re-run the forward pass afterward. So immediately after `mj_step` returns, `data.qpos`/`data.qvel` are the *new*, post-integration state — but `data.site_xpos`, `data.xpos`, and `data.ncon` still describe the *old*, pre-integration pose, because those are only computed during the forward pass that already ran at the top of the step.

Concretely, this is a bug:

```python
mujoco.mj_step(model, data, nstep=frame_skip)
tip = data.site_xpos[tip_site_id]   # stale: one control step behind qpos
```

`panda/env.py`'s `step()` calls `mj_forward` immediately after `mj_step` for exactly this reason:

```python
mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
# REQUIRED. mj_step ends after integration, so site_xpos (and ncon) still
# describe the pre-integration state. Without this, `y` lags one control
# step and every reward is computed against a stale tip position.
mujoco.mj_forward(self.model, self.data)
```

!!! warning "This is not a crash — it's a silent lag"
    Skip the `mj_forward` call and nothing errors. The env keeps running, the reward keeps being a number, `reached` keeps flipping true and false — it's just all computed from where the tip *was* last step, not where it *is*. Every MuJoCo env with a site-based reward inherits this trap. `tests/test_panda_env.py::test_tip_is_fresh_after_step` pins it here, by comparing `env.y` after `step()` against an independent FK computed from the same post-step `qpos`. If that test ever fails, the bug is in `env.step`, not the test.

## 4. `opt.timestep` vs a control period; `frame_skip`

Two different clocks are running:

- `model.opt.timestep` — the physics integrator's step, **0.002 s** (500 Hz). This has to be fine enough for the simulation to stay numerically stable; the actuators here run PD gains up to `kp = 4500` (section 6), which is stiff enough to need a small physics step.
- The control period — how often *you* (or an RL policy) issue a new action, **0.02 s** (50 Hz) in `PandaReachEnv`.

`panda/model.py`'s `frame_skip()` computes how many physics steps fit in one control period, and raises rather than silently rounding to zero if they don't fit at least once:

```python
def frame_skip(model: mujoco.MjModel, dt_ctrl: float = 1.0 / CTRL_HZ) -> int:
    n = int(round(dt_ctrl / model.opt.timestep))
    if n < 1:
        raise ValueError(...)
    return n
```

From the probe output: `opt.timestep=0.002  frame_skip@50Hz=10` — ten physics steps execute inside every `env.step()` call, via `mujoco.mj_step(model, data, nstep=10)`.

## 5. Bodies vs geoms vs sites

Three different things live on the kinematic tree, and only one of them is what you want for "where is the end-effector right now":

- **Bodies** are the tree's rigid links, each carrying mass and inertia. This model has `nbody=10`: `['world', 'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'attachment']`.
- **Geoms** are the collision/visual shapes attached to bodies — what generates `data.ncon` when two of them intersect. Nothing in this project queries a geom by name.
- **Sites** are massless reference frames welded to a body — a labeled point with zero physical effect on the simulation, which is exactly what you want for "read this pose" without inventing a fake body. This model has `nsite=1`: `['attachment_site']`, sitting where a hand would bolt onto `link7`'s flange.

Read it as:

```python
tip = data.site_xpos[model.site("attachment_site").id]
```

`panda/model.py` resolves the id once (`tip_id()`) and says to cache it rather than call it per step — a string lookup on every control tick is wasted work when the id never changes for a given model.

At the `home` keyframe (section 7):

```
tip @ home = [ 0.5545 -0.      0.6245]
```

## 6. Actuators

MJCF gives you `<motor>`, `<position>`, `<velocity>`, and `<general>` as actuator shorthands, but every one of them compiles down to the same two underlying fields MuJoCo actually simulates: `gaintype`/`gainprm` and `biastype`/`biasprm`. The force an actuator applies is, schematically, `gain(...) * ctrl + bias(...)`.

!!! warning "Never hand-roll a lookup table for these enums"
    It's tempting to write `{0: "none", 1: "muscle", ...}` once and move on. Don't — `mjtGain` and `mjtBias` are *different* enumerations that happen to reuse small integers for different things (index `0` is `FIXED` for gain but `NONE` for bias), and both have grown new members across MuJoCo versions (`3` is `DCMOTOR`, not `USER`; `USER` is `5`). A hand-written table silently rots the next time either enum grows. Ask MuJoCo instead:

    ```python
    mujoco.mjtGain(int(model.actuator_gaintype[i])).name   # 'mjGAIN_FIXED'
    mujoco.mjtBias(int(model.actuator_biastype[i])).name   # 'mjBIAS_AFFINE'
    ```

This model's actuators, from the probe:

```
ACTUATORS
  0 actuator1   gain=FIXED  bias=AFFINE kp= 4500.0 kd= 450.0 ctrlrange=[-2.8973  2.8973]
  1 actuator2   gain=FIXED  bias=AFFINE kp= 4500.0 kd= 450.0 ctrlrange=[-1.7628  1.7628]
  2 actuator3   gain=FIXED  bias=AFFINE kp= 3500.0 kd= 350.0 ctrlrange=[-2.8973  2.8973]
  3 actuator4   gain=FIXED  bias=AFFINE kp= 3500.0 kd= 350.0 ctrlrange=[-3.0718 -0.0698]
  4 actuator5   gain=FIXED  bias=AFFINE kp= 2000.0 kd= 200.0 ctrlrange=[-2.8973  2.8973]
  5 actuator6   gain=FIXED  bias=AFFINE kp= 2000.0 kd= 200.0 ctrlrange=[-0.0175  3.7525]
  6 actuator7   gain=FIXED  bias=AFFINE kp= 2000.0 kd= 200.0 ctrlrange=[-2.8973  2.8973]
```

`gaintype=FIXED` means the gain term is just the constant `gainprm[0] = kp`. `biastype=AFFINE` means `bias = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity`, and for a joint-space actuator `length = qpos`, `velocity = qvel`. With `biasprm = [0, -kp, -kd]`, the total force is

```
force = kp * ctrl + (0 - kp*qpos - kd*qvel) = kp*(ctrl - qpos) - kd*qvel
```

— a textbook PD position servo, spelled out by hand instead of through the `<position kp=... kv=...>` convenience tag. That's the whole reason `panda/model.py` never needed a `<position>` override: this model already **is** one.

**Consequence.** `apply_delta` writes `ctrl = clip(qpos_measured + delta, safe_box)` — an absolute target, `delta` away from where the arm *currently* is, not from where it was last commanded. Under a constant delta command the servo settles into a steady-state lag, and that steady-state joint speed is capped near `(kp/kd) * delta_max`. Measured directly (`scripts/mujoco_hello.py`, holding `ctrl = qpos + delta_max` at 50 Hz for 30 steps from the `home` pose):

```
SERVO: constant delta command, 50 Hz
  delta_max=0.05  qdot_ss= 0.42 rad/s   (kp/kd prediction  0.5)
  delta_max=0.10  qdot_ss= 0.84 rad/s   (kp/kd prediction  1.0)
  delta_max=0.20  qdot_ss= 1.63 rad/s   (kp/kd prediction  2.0)
  delta_max=0.40  qdot_ss= 1.98 rad/s   (kp/kd prediction  4.0)
```

`kp/kd` is exactly `10` for every joint here (`4500/450 = 3500/350 = 2000/200 = 10`), and the prediction tracks the measured speed closely at small `delta_max`, then visibly saturates by `0.4` — real dynamics (inertia, gravity, the other six joints) start to dominate once the commanded lag gets large, so the linear servo law is a small-signal approximation, not a hard cap. `PandaReachEnv`'s own `DELTA_MAX = 0.2` sits in the linear region, at `1.63 rad/s`, just under the real Panda's rated `2.175 rad/s`.

!!! note "A servo bounds the *command*, not the *achieved position*"
    `apply_delta` guarantees `ctrl` stays inside the safe box — that's unconditional. It does **not** guarantee `qpos` does. Momentum can carry the physical joint a little past a correctly-clipped target — `panda/validity.py`'s `qpos_safe_box_excursions` diagnostic measures this directly, and is **0** under the random excitation `scripts/record_panda_video.py` drives (nothing pushes one direction long enough to produce an excursion). A DLS-IK oracle, since removed as out-of-scope, once measured a nonzero case under sustained goal-directed commands: a worst-case overshoot of **-0.0029 rad (0.16°)** past the safe box, using at most ~1.0% of the 0.300 rad (17.2°) headroom the safe box leaves before the real hardware limit, with **0 violations** of the real `model.jnt_range` in the same run — that number is no longer reproducible from this tree. Conflating "safe-box excursion" with "joint-limit violation" looks like a safety bug during development; it's benign PD-servo overshoot, nowhere near the real limit.

## 7. Keyframes

An MJCF `<keyframe>` is a named, saved configuration baked into the model at compile time — a convenient default pose you don't want to hardcode as magic numbers in Python. `model.nkey=1`, named `home`:

```
KEYS   ['home']
  key_qpos[0] = [ 0.      0.      0.     -1.5708  0.      1.5708 -0.7853]
```

`scripts/mujoco_hello.py` uses it to compute the "tip at home" FK point from section 5:

```python
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
```

Nothing in `PandaReachEnv` resets to `home` — starts and goals are rejection-sampled instead (`panda/model.py`'s `sample_config`) — but `home` is the natural pose to sanity-check any new snippet against, since it's the one configuration every reader of the XML can look up by eye.

## 8. Rendering

`PandaReachEnv` supports only `render_mode="rgb_array"` (`metadata["render_modes"] = ["rgb_array"]`), drawn offscreen through `mujoco.Renderer`:

```python
self.renderer = mujoco.Renderer(env.model, height=480, width=640)
...
self.renderer.update_scene(self.env.data, camera=self.camera)
frame = self.renderer.render()   # (480, 640, 3) uint8
```

Two details in `panda/rendering.py` will bite you if you don't know them going in:

- **`update_scene` resets `scene.ngeom`.** The goal marker isn't a body in the model — there's nothing to ask MuJoCo to draw — so it's injected as a scratch geom directly into the scene via `mjv_initGeom`, and that has to happen *after* `update_scene` (which zeroes `ngeom`) and *before* `render()` (which reads it). Get the order backwards and the marker either never appears or gets clobbered.
- **This model ships no lights.** `panda_nohand.xml` has none; `mujoco_menagerie` puts them in `scene.xml`, which wraps the *with-hand* model and isn't usable here. Without a fix, `rgb_array` frames come back nearly black. `panda/model.py`'s `load_model()` compensates by bumping the built-in headlight (`model.vis.headlight.ambient`/`diffuse`) — a `vis`-only, model-level edit, tying back to section 1's point that it can never affect a rollout.

!!! note "There is no interactive viewer here, and that's deliberate"
    `mujoco.viewer.launch_passive` raises unless the process was started with MuJoCo's own `mjpython` launcher — it cannot work under a plain `uv run python` script, which is why `render_mode="human"` was never added. To look around the model interactively, use MuJoCo's standalone viewer instead:

    ```bash
    uv run mjpython -m mujoco.viewer --mjcf="$(uv run python -c \
        'import panda.model as m; print(m.model_path())')"
    ```

## 9. `mj_jacSite` and redundancy

`mujoco.mj_jacSite` gives the positional Jacobian at a site: a `(3, nv)` matrix mapping joint velocity to tip velocity, `ẋ_tip = J q̇`.

```python
jacp = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, None, tip_site_id)
```

**The redundancy.** Seven joints drive a 3-D tip position, so the map `q ↦ tip` has a 4-dimensional space of joint motions that, to first order, don't move the tip at all — the null space of `J`. This is a real fact about the arm, not an artifact of any particular controller: any 7-DoF-to-3-D mapping has it.

`panda/env.py` never uses `mj_jacSite` or any Jacobian-based control — its action interface is delta joint targets, not Cartesian ones. A DLS-IK controller built on this Jacobian was used once, during design, to measure the env's solvability ceiling; it has since been removed as out-of-scope (see the design doc's appendix if that measurement needs redoing) and no numbers from it are reproducible from this tree.

The redundancy still matters for what comes next: seven inputs mapping to a 3-D output means many different joint trajectories realise the same tip trajectory. A later DeePC stage's sparse (`ℓ1`-regularized) solution has no built-in reason to prefer the "natural" one among them — it may track the tip well while producing joint motion that looks ugly or wasteful.

## 10. The linear region, and what it means for anchors

A DeePC library asserts the map `u → y` is LTI over the prediction horizon, so the library is valid wherever that holds. `scripts/measure_linear_region.py` measures it directly: fit `Y = U @ G + free` at an anchor over N=12, then apply the *same* input sequences at an offset configuration and check whether the anchor's `G` still predicts, given the test point's own free response (which `y_ini` supplies to DeePC, so handing it over is fair). Threshold is 25 mm — half the 0.05 m `goal_tolerance`.

**Azimuth is a symmetry, not a nonlinearity.** Joint 1's axis is vertical and gravity is along the same axis, so rotating `q1` rotates the whole arm about z and leaves the joint-space dynamics untouched. Measured, driving `q1 = −1.8` and `q1 = 0.6` with identical delta sequences and de-rotating by `Rz(Δq1)`:

```
[0] q1 equivariance: max |Rz(dq)*tip(q1=-1.8) - tip(q1=0.6)| = 6.661e-16 m
```

Machine precision — this is an exact geometric invariant, not an approximation, and the script asserts it. The consequence for `panda/deepc_setup.py` is direct: the four azimuth-keyed libraries differ **only by a known rotation**. Keying on azimuth compensates for `Rz` that could be applied analytically to one library; it buys no model fidelity. This is the "rotating q1 rotates the Jacobian but leaves its singular values alone" remark in that module's docstring, confirmed numerically.

**What actually bounds the linear region is horizon excursion, not anchor distance.** The linearization error at the anchor *itself* — before moving anywhere — as a function of probe amplitude (`n_samples=600`, so this is genuine nonlinearity, not fit variance; the residual *rises* toward ~19 mm as samples grow at `probe=0.1`):

| probe δ | horizon excursion `N·δ` | error at the anchor |
|---|---|---|
| 0.02 | 0.24 rad | 0.4 mm |
| 0.05 | 0.60 rad | 5.4 mm |
| 0.10 | 1.20 rad | 19.0 mm |
| 0.12 | 1.44 rad | **24.9 mm** ← crosses threshold |
| 0.16 | 1.92 rad | 38.0 mm |
| 0.20 | 2.40 rad | **50.7 mm** |

`PandaReachEnv`'s `DELTA_MAX = 0.2` with `N = 12` puts the operating point on the last row: **the local linear model is already 50.7 mm wrong at the anchor, worse than the 50 mm goal tolerance it is supposed to steer inside.** No anchor placement fixes that — the error is there before any offset is applied.

Anchor distance, by contrast, is cheap. At `probe = 0.05` (5.4 mm floor), sweeping each joint away from the anchor:

```
      offset:     0.20     0.40     0.60     0.80     1.00     1.20     1.40
  joint 1:      6.2      8.3     10.9     13.6     16.2     18.8     21.3
  joint 2:      7.2      9.7     11.8     14.1     16.2     18.3        -
  joint 3:      6.0      7.7     10.1     12.8     15.7     18.6     21.5
  joint 4:      6.2      7.7      9.6     11.7     14.1        -        -
  joint 5:      5.5      5.7      6.0      6.3      6.7      7.1      7.5
  joint 6:      5.8      6.3      6.9      7.5      8.0      8.6      9.1
  joint 7:      5.4      5.4      5.4      5.4      5.4      5.4      5.4
```

No joint crosses 25 mm within 1.4 rad — and joint 1's safe box is only ±2.318 total, so a cell radius of 1.4 rad covers most of the reachable range. Joints 5–7 (the wrist) are essentially flat: they reorient the flange without changing the arm's conditioning, so they need no anchoring at all. Joints 2 and 4 (shoulder and elbow — the *extension* axes) grow fastest, which is the ordering `panda/deepc_setup.py`'s docstring predicts, but even they stay under threshold across the sweep.

!!! warning "The actionable knob is `N · delta_max`, not the anchor count"
    Reading these two tables together: anchor placement is a second-order effect on this system, and adding anchors cannot buy back a 50.7 mm modelling error incurred at the anchor. The lever that moves the number is the horizon excursion — the 25 mm crossing sits at `N·δ ≈ 1.44 rad`, against the current `12 × 0.2 = 2.4 rad`. Shortening `N`, lowering `delta_max`, or both, is what shrinks it.

    This is a *separate* limit from the one in `PandaReachEnv.y_ext`'s docstring. That one says tip-only `y` fails to observe the state, so the past window maps one-to-many onto futures — a violation of Willems' lemma's precondition that no amount of data or keying fixes. This one says that even with the state observed, the horizon is long enough for curvature to dominate. Both are live; `output="ext"` addresses only the first.

## Try it yourself

Two commands cover every number this project can still reproduce on `PandaReach-v0`:

```bash
uv run python scripts/mujoco_hello.py                            # model facts + workspace survey
uv run python scripts/record_panda_video.py --episodes 3         # reached 0/3, writes the MP4
```

Random actions reach **0** goals (`panda/validity.py`'s own validity check, measured over 20 seeds; the command above reproduces it over 3 as 0/3). That is the one live bound this tree can check: since random reaches 0, any nonzero reach rate a later controller reports is signal, not noise. There used to be an upper bound too — a DLS-IK oracle, full knowledge of the Jacobian, no learning, no noise, reached 0.90 — but that was a one-off design-time measurement from code since removed as out-of-scope, and is no longer checkable here.

---

That covers everything the rest of this repo's Panda code assumes you already know. For the API surface itself, read `panda/env.py` and `panda/model.py` directly — they're short, and every constant in them cites its own measurement in a comment, including the rationale for `DELTA_MAX=0.2` and `SAFE_MARGIN=0.10` that this page only summarizes. For the *why* the reward mirrors the unicycle's, see the [Journey](../journey/index.md) — the decision log this project already keeps for `TwoWheelGoal-v0`; it predates the Panda env and doesn't cover it directly, but the reward form was carried over unchanged.
