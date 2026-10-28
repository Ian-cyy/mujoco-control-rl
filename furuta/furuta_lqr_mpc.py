"""Furuta 旋转倒立摆：能量法摆起 + LQR 稳定（平衡阶段可选 MPC 采样）。

- 摆起：能量法（同 furuta_swingup.py），把杆从下垂摇到竖直。
- 平衡：LQR 打底；--balance mpc 时在 LQR 上再叠一层随机采样 MPC。
- viewer 模式：平衡后**不退出**，一直稳到关窗；被打倒(|pole|>FALL_PHI)会退回摆起再试。

用法：
  python3 furuta_lqr_mpc.py
  python3 furuta_lqr_mpc.py --balance lqr
  python3 furuta_lqr_mpc.py --headless 40000
  python3 furuta_lqr_mpc.py --knock 60            # 平衡后自动打一下
"""

import argparse
import time

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np

from mpc_furuta import DT, FURUTA_XML, NTHREAD, UMAX, wrap
from furuta_swingup import (
    FALL_PHI,
    K_ENERGY,
    SW_PHI,
    SW_PHID,
    lqr_gain,
    lqr_torque,
    pole_energy,
)

ARM_DAMP = 1.0
SIGMA = 0.5
H_MPC = 60
NS_MPC = 200
KEEP = 5
W_POLE, W_ARM, W_VEL, W_U = 50.0, 1.0, 0.5, 0.01


def energy_torque(model, data):
    e = pole_energy(model, data)
    phi, phid = data.qpos[1], data.qvel[1]
    u = K_ENERGY * e * np.sign(phid) * np.cos(phi) - ARM_DAMP * data.qvel[0]
    return float(np.clip(u, -UMAX, UMAX))


def mpc_around_lqr(model, data, datas, K):
    base = lqr_torque(K, data)
    n_knots = H_MPC // KEEP
    knots = base + np.random.normal(0.0, SIGMA, (NS_MPC, n_knots))
    knots[0, :] = base                      # 保留"纯 LQR"候选，保证不劣于它
    controls = np.repeat(np.clip(knots, -UMAX, UMAX), KEEP, axis=1)[:, :, None]
    state = np.concatenate([[data.time], data.qpos, data.qvel])
    init = np.tile(state, (NS_MPC, 1))
    states, _ = mujoco.rollout.rollout(
        model, datas, init, controls, persistent_pool=True)
    qpos, qvel = states[:, :, 1:3], states[:, :, 3:5]
    arm, pole = qpos[:, :, 0], wrap(qpos[:, :, 1])
    cost = np.sum(W_POLE * pole ** 2 + W_ARM * arm ** 2
                  + W_VEL * (qvel[:, :, 0] ** 2 + qvel[:, :, 1] ** 2)
                  + W_U * controls[:, :, 0] ** 2, axis=1)
    return float(controls[int(np.argmin(cost)), 0, 0])


def run(controller, steps, viewer, model, data, K, knock=0.0):
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    mode = "swing"
    hold = balanced = 0
    knocked = settled = False
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        phi, phid = wrap(data.qpos[1]), data.qvel[1]
        if mode == "balance" and abs(phi) > FALL_PHI:
            mode, settled = "swing", False
        elif mode == "swing" and abs(phi) < SW_PHI and abs(phid) < SW_PHID:
            mode = "balance"

        if mode == "swing":
            u = energy_torque(model, data)
        elif controller == "mpc":
            u = mpc_around_lqr(model, data, datas, K)
        else:
            u = lqr_torque(K, data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        if mode == "balance" and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= 250:
                hold = 0
                if not settled:
                    balanced += 1
                    settled = True
                    print(f"BALANCED #{balanced} at step {step}  "
                          f"t={step * DT:.2f}s", flush=True)
                    if knock > 0 and not knocked:
                        data.qvel[1] += knock
                        knocked, settled, hold = True, False, 0
                        print(f"KNOCK {knock} applied", flush=True)
                    elif viewer is None:
                        return
        else:
            hold = 0
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", choices=["mpc", "lqr"], default="mpc")
    parser.add_argument("--knock", type=float, default=0.0)
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(FURUTA_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = np.pi
    data.qvel[1] = 0.3
    mujoco.mj_forward(model, data)
    K, maxeig = lqr_gain(model)
    print(f"LQR K={np.round(K, 3)}  |closed-loop|max={maxeig:.3f}", flush=True)

    try:
        if args.headless:
            run(args.balance, args.headless, None, model, data, K, args.knock)
            return
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(args.balance, 10 ** 9, viewer, model, data, K, args.knock)
    finally:
        mujoco.rollout.shutdown_persistent_pool()


if __name__ == "__main__":
    main()
