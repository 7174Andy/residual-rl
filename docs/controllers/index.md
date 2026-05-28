# Controllers

The controllers package provides a data-driven predictive controller (DeePC) and the machinery it needs to work on the kinematic unicycle.

## Layout

```text
two_wheel_robot/controllers/
    __init__.py
    data_collection.py    # offline (u, y) trajectory generation
    hankel.py             # block-Hankel matrix construction
    deepc.py              # DeePC + LibrarySwitchingDeePC
```

A controller is anything implementing the `Controller` protocol:

```python
class Controller(Protocol):
    def reset(self, y_initial: np.ndarray) -> None: ...
    def act(self, y_current: np.ndarray, y_ref: np.ndarray) -> np.ndarray: ...
```

(The protocol isn't enforced as a literal `typing.Protocol`; both `DeePC` and `LibrarySwitchingDeePC` follow this shape.)

## The DeePC pipeline

```mermaid
graph LR
    A[Random PE inputs] --> B[Offline rollout in env]
    B --> C[(u, y) trajectory]
    C --> D[Block-Hankel<br>Up, Uf, Yp, Yf]
    D --> E[DeePC QP problem]
    E -->|per step| F[u_t = first action]
    F --> G[env.step]
    G --> C
```

Each box maps to a module:

- [Data collection](data-collection.md) — `controllers/data_collection.py`
- Block Hankel construction — `controllers/hankel.py`
- [DeePC controller](deepc.md) — `controllers/deepc.py`
- [Library switching](library-switching.md) — `controllers/deepc.py::LibrarySwitchingDeePC`

## Why this layered design

- **`dynamics.py`** is pure numpy with no Gym dep → controllers and tests can use it directly.
- **`controllers/`** is RL-library-agnostic → no torch/sb3 here. A controller object talks numpy in and out.
- **`rl/`** is the only place that imports `stable_baselines3`.

This separation lets the same env serve both worlds without polluting either.
