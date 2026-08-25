"""Opt-in Weights & Biases plumbing, shared by every system's scripts.

`wandb` is imported ONLY inside `init_run` when a project name is given, so
runs without `--wandb-project` never touch it — no login, no network, no
change to determinism. Every helper is a no-op on `run=None`, so call sites
need no conditionals.
"""
from __future__ import annotations


def init_run(project: str | None, name: str | None = None,
             config: dict | None = None, tags: list[str] | None = None,
             group: str | None = None, sync_tensorboard: bool = False):
    """Start a W&B run, or return None when no project is given.

    ``sync_tensorboard=True`` mirrors SB3's tensorboard scalars (actor/critic
    losses, entropy coef, ...) into the run — pair it with a
    ``tensorboard_log`` dir on the model.
    """
    if not project:
        return None
    import wandb
    return wandb.init(project=project, name=name, config=config or {},
                      tags=tags, group=group, sync_tensorboard=sync_tensorboard)


def sb3_callback(run, prefix: str = ""):
    """SB3 callback streaming Monitor episode returns to W&B. None-safe.

    ``prefix`` namespaces the keys (e.g. "reacher" -> "reacher/episode_return"),
    which W&B renders as a separate chart section per system.
    """
    if run is None:
        return None
    from stable_baselines3.common.callbacks import BaseCallback
    pre = f"{prefix}/" if prefix else ""

    class _WB(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep:
                    row = {f"{pre}episode_return": float(ep["r"]),
                           f"{pre}episode_length": int(ep["l"])}
                    # end-of-episode task metrics both envs expose in info
                    dist = info.get("dist", info.get("distance"))
                    if dist is not None:
                        row[f"{pre}final_dist"] = float(dist)
                    if "reached" in info:
                        row[f"{pre}reached"] = int(bool(info["reached"]))
                    run.log(row, step=self.num_timesteps)
            return True

    return _WB()


def callbacks(*cbs):
    """Drop Nones; return a list SB3 accepts, or None if nothing remains."""
    out = [c for c in cbs if c is not None]
    return out or None


def log_table(run, key: str, columns: list[str], rows: list) -> None:
    if run is None:
        return
    import wandb
    run.log({key: wandb.Table(columns=columns, data=[list(r) for r in rows])})


def log_image(run, key: str, path: str) -> None:
    if run is None:
        return
    import wandb
    run.log({key: wandb.Image(path)})


def finish(run) -> None:
    if run is not None:
        run.finish()
