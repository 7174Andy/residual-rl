"""Print everything about the Panda model this project depends on.

Run this first if you have never used MuJoCo. Every number in
`docs/superpowers/specs/2026-08-10-panda-reach-env-design.md` and every constant
in `panda/model.py` comes from this output, so it is the way to check they are
still true after a `mujoco` or `robot_descriptions` upgrade.

    uv run python scripts/mujoco_hello.py
"""
from __future__ import annotations

import mujoco
import numpy as np

from panda.model import (
    MIN_TIP_Z,
    frame_skip,
    load_model,
    model_path,
    safe_box,
    tip_id,
)


def main() -> None:
    print(f"model: {model_path()}")
    model, data = load_model()
    tid = tip_id(model)

    print(
        f"\nnq={model.nq} nv={model.nv} nu={model.nu} na={model.na} "
        f"nbody={model.nbody} nsite={model.nsite} nkey={model.nkey}"
    )
    print(f"opt.timestep={model.opt.timestep}  frame_skip@50Hz={frame_skip(model)}")

    print("\nJOINTS")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i} {name:8s} limited={model.jnt_limited[i]} range={model.jnt_range[i]}")

    print("\nSAFE BOX (joint range trimmed 10% each end)")
    lo, hi = safe_box(model)
    print(f"  lo {np.round(lo, 3)}")
    print(f"  hi {np.round(hi, 3)}")

    # This block is what proves no <position> override is needed: gaintype FIXED
    # with biastype AFFINE and biasprm = [0, -kp, -kd] IS a PD position servo,
    # force = kp*(ctrl - q) - kd*qdot.
    print("\nACTUATORS")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        gaintype_name = mujoco.mjtGain(int(model.actuator_gaintype[i])).name.removeprefix("mjGAIN_")
        biastype_name = mujoco.mjtBias(int(model.actuator_biastype[i])).name.removeprefix("mjBIAS_")
        print(
            f"  {i} {name:11s} "
            f"gain={gaintype_name:6s} "
            f"bias={biastype_name:6s} "
            f"kp={model.actuator_gainprm[i][0]:7.1f} "
            f"kd={-model.actuator_biasprm[i][2]:6.1f} "
            f"ctrlrange={np.round(model.actuator_ctrlrange[i], 4)}"
        )

    print("\nSITES ", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                       for i in range(model.nsite)])
    print("BODIES", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                     for i in range(model.nbody)])
    print("KEYS  ", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
                     for i in range(model.nkey)])
    if model.nkey:
        print(f"  key_qpos[0] = {np.round(model.key_qpos[0], 4)}")

    data.qpos[:] = model.key_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"\ntip @ home = {np.round(data.site_xpos[tid], 4)}")

    # Workspace + rejection-rate survey. Reproduces the design-doc figures.
    rng = np.random.default_rng(0)
    n = 4000
    tips = np.empty((n, 3))
    n_contact = 0
    for k in range(n):
        data.qpos[:] = rng.uniform(lo, hi)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        tips[k] = data.site_xpos[tid]
        n_contact += data.ncon > 0
    radius = np.linalg.norm(tips, axis=1)
    print(f"\nWORKSPACE over {n} uniform safe-box samples")
    print(f"  tip radius  {radius.min():.3f} .. {radius.max():.3f}  (mean {radius.mean():.3f})")
    print(f"  tip x       {tips[:, 0].min():.3f} .. {tips[:, 0].max():.3f}")
    print(f"  tip y       {tips[:, 1].min():.3f} .. {tips[:, 1].max():.3f}")
    print(f"  tip z       {tips[:, 2].min():.3f} .. {tips[:, 2].max():.3f}")
    print(f"  self-collision  {n_contact}/{n} = {100 * n_contact / n:.1f}%")
    print(f"  tip z < {MIN_TIP_Z}    {100 * (tips[:, 2] < MIN_TIP_Z).mean():.1f}%")

    half = n // 2
    pair = np.linalg.norm(tips[:half] - tips[half:], axis=1)
    print(f"  start-goal tip distance  mean {pair.mean():.3f}  p10 {np.percentile(pair, 10):.3f}")

    # The servo lag that sets delta_max: q settles at a fixed offset behind ctrl,
    # capping joint speed near (kp/kd)*delta_max.
    fs = frame_skip(model)
    # Derive kp/kd ratio from the model, not hardcoded constants.
    kp_values = model.actuator_gainprm[:, 0]
    kd_values = -model.actuator_biasprm[:, 2]
    kp_kd_ratios = kp_values / kd_values
    kp_kd_mean = kp_kd_ratios.mean()
    kp_kd_range = (kp_kd_ratios.min(), kp_kd_ratios.max())
    kp_kd_str = (
        f"{kp_kd_mean:.1f}"
        if np.isclose(kp_kd_range[0], kp_kd_range[1])
        else f"{kp_kd_range[0]:.1f}–{kp_kd_range[1]:.1f}"
    )
    print(f"\nSERVO: constant delta command, 50 Hz  (kp/kd = {kp_kd_str})")
    for dmax in (0.05, 0.1, 0.2, 0.4):
        d2 = mujoco.MjData(model)
        d2.qpos[:] = model.key_qpos[0]
        mujoco.mj_forward(model, d2)
        for _ in range(30):
            d2.ctrl[:] = np.clip(d2.qpos + dmax, lo, hi)
            mujoco.mj_step(model, d2, nstep=fs)
        print(f"  delta_max={dmax:4.2f}  qdot_ss={np.abs(d2.qvel).mean():5.2f} rad/s"
              f"   (kp/kd prediction {kp_kd_mean * dmax:4.1f})")


if __name__ == "__main__":
    main()
