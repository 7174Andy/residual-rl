"""Greedy inference videos for the vanilla and residual arms, both systems.

Rolls each 400k policy deterministically on held-out scenarios/seeds, writes
one MP4 per (system, arm) to data/, and — with --wandb-project — logs each
system's pair of videos into a W&B run (namespaced keys, so they land in the
same per-system chart sections as the metrics).

    uv run python scripts/record_inference_videos.py --episodes 3
    uv run python scripts/record_inference_videos.py --wandb-project two-wheel-exp
"""
from __future__ import annotations

import argparse

import numpy as np

from core.video_encoding import encode_video


def record_reacher(arm: str, episodes: int, offset: int = 0) -> str:
    import gymnasium as gym

    import reacher  # noqa: F401  registers the Gym ID
    from reacher.residual_env import ResidualSelectEnv
    from rl.sb3 import load_policy

    with np.load("data/reacher_scenarios_v1.npz") as z:
        eps = [(z["qpos"][i], z["goal"][i])
               for i in range(offset, offset + episodes)]

    frames = []
    if arm == "residual":
        env = ResidualSelectEnv(clone_path="data/dagger_clone_r3.pt",
                                residual_frac=2.0, render_mode="rgb_array")
        model = load_policy("data/reacher_ckpt_seeds/resf2_s0.zip", algo="sac")
        for q0, g in eps:
            obs, _info = env.reset(seed=0, options={"qpos": q0, "goal": g})
            frames.append(env.render())
            trunc = False
            while not trunc:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, _t, trunc, _info = env.step(action)
                frames.append(env.render())
            frames.extend([frames[-1]] * 25)
        env.close()
    else:
        env = gym.make("ReacherGoal-v0", render_mode="rgb_array")
        model = load_policy("data/reacher_vanilla_400k.zip", algo="sac")
        base = env.unwrapped
        for q0, g in eps:
            _obs, _info = env.reset(seed=0, options={"qpos": q0, "goal": g})
            frames.append(env.render())
            trunc = False
            while not trunc:
                action, _ = model.predict(base.build_obs(), deterministic=True)
                _obs, _r, _t, trunc, _info = env.step(action)
                frames.append(env.render())
            frames.extend([frames[-1]] * 25)
        env.close()
    out = f"data/reacher_inference_{arm}_{offset}.mp4"
    encode_video(frames, out, fps=50)
    return out


def record_unicycle(arm: str, episodes: int, seed_base: int,
                    offset: int = 0) -> str:
    from rl.sb3 import load_policy
    from two_wheel_robot.rl.deepc_setup import canonical_action_bounds
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
    from two_wheel_robot.rl.wrappers import vanilla_rl_env

    if arm == "residual":
        env = ResidualDeePCEnv(clone_path="data/clone.pt",
                               libraries_path="data/libraries_v0.npz",
                               residual_frac=2.0, render_mode="rgb_array")
        model = load_policy("data/residual_td3_400k_frac2.zip", algo="td3")
    else:
        env = vanilla_rl_env(canonical_action_bounds("data/libraries_v0.npz"),
                             render_mode="rgb_array")
        model = load_policy("data/vanilla_td3_400k.zip", algo="td3")

    frames = []
    for i in range(offset, offset + episodes):
        obs, _info = env.reset(seed=seed_base + i)
        frames.append(env.render())
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            frames.append(env.render())
        frames.extend([frames[-1]] * 20)
    env.close()
    out = f"data/unicycle_inference_{arm}_{offset}.mp4"
    encode_video(frames, out, fps=40)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=3,
                   help="episodes per video set")
    p.add_argument("--sets", type=int, default=1,
                   help="number of consecutive video sets per arm")
    p.add_argument("--seed-base", type=int, default=4104626029,
                   help="unicycle eval-seed namespace (reacher uses the "
                        "frozen scenarios npz)")
    p.add_argument("--systems", nargs="+", default=["reacher", "unicycle"],
                   choices=["reacher", "unicycle"])
    p.add_argument("--wandb-project", default=None,
                   help="also log the videos to Weights & Biases")
    args = p.parse_args()

    rec = {"reacher": record_reacher,
           "unicycle": lambda arm, n, off: record_unicycle(
               arm, n, args.seed_base, off)}
    outs: dict[str, list[tuple[str, int, str]]] = {}
    for system in args.systems:
        outs[system] = []
        for arm in ("vanilla", "residual"):
            for k in range(args.sets):
                if system == "reacher":
                    path = record_reacher(arm, args.episodes,
                                          k * args.episodes)
                else:
                    path = rec[system](arm, args.episodes, k * args.episodes)
                outs[system].append((arm, k, path))
                print("wrote", path)

    if args.wandb_project:
        import wandb
        for system, vids in outs.items():
            run = wandb.init(project=args.wandb_project,
                             name=f"{system}_inference_videos",
                             tags=[system, "inference", "video"], reinit=True,
                             config={"episodes": args.episodes,
                                     "sets": args.sets})
            for arm, k, path in vids:
                key = (f"{system}/video_{arm}" if args.sets == 1
                       else f"{system}/video_{arm}_set{k}")
                run.log({key: wandb.Video(path)})
            run.finish()
            print(f"logged {system} videos to W&B")


if __name__ == "__main__":
    main()
