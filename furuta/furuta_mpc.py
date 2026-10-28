"""Furuta：统一 MPC 摆起 + LQR/MPC 平衡（不依赖能量法和人工初速）。

- 摆起：长视界 MPC（随机采样控制序列，mujoco.rollout 并行预测），
        代价 = 杆竖直 + 速度小 + 省力，末端加重 -> 直接把杆规划到竖直。
- 平衡：LQR 打底 + 可选 MPC 采样（--balance lqr 切纯 LQR）。
- viewer 模式平衡后不退出，可被打倒后自动重摆；headless 可测摆起/恢复时间。

用法：
  python3 furuta_mpc.py
  python3 furuta_mpc.py --balance lqr
  python3 furuta_mpc.py --headless 40000
  python3 furuta_mpc.py --knock 60 --headless 400000
"""

import argparse
import os
import time

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np

from mpc_furuta import DT, FURUTA_XML, NTHREAD, UMAX, wrap
from furuta_swingup import energy_torque, lqr_gain, lqr_torque

NTHREAD = os.cpu_count() or 4

H_SW, KEEP_SW = 600, 25
CEM_NS, CEM_ITERS, CEM_ELITE = 200, 0, 0.2
KICK_AMP, KICK_STEPS = 2.0, 20
W_SW_POLE, W_SW_VEL, W_SW_U, W_SW_TERM = 30.0, 0.5, 0.001, 300.0

H_BAL, NS_BAL, KEEP_BAL, SIGMA = 60, 200, 5, 0.5
W_POLE, W_ARM, W_VEL, W_U = 50.0, 1.0, 0.5, 0.01

CATCH_PHI, CATCH_PHID = 0.25, 1.0
FALL_PHI = 0.7


def _rollout(model, datas, data, controls):
    state = np.concatenate([[data.time], data.qpos, data.qvel])
    init = np.tile(state, (controls.shape[0], 1))
    states, _ = mujoco.rollout.rollout(
        model, datas, init, controls, persistent_pool=True)
    return states


def _swing_cost(states, controls):
    qpos, qvel = states[:, :, 1:3], states[:, :, 3:5]
    pole = wrap(qpos[:, :, 1])
    cost = np.sum(W_SW_POLE * pole ** 2
                  + W_SW_VEL * (qvel[:, :, 0] ** 2 + qvel[:, :, 1] ** 2)
                  + W_SW_U * controls[:, :, 0] ** 2, axis=1)
    return cost + W_SW_TERM * (pole[:, -1] ** 2 + 10.0 * qvel[:, -1, 1] ** 2)


def energy_seed(model, data, scratch, n_knots):
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, scratch)
    seq = []
    for i in range(H_SW):
        u = KICK_AMP if i < KICK_STEPS else energy_torque(model, scratch)
        scratch.ctrl[0] = u
        mujoco.mj_step(model, scratch)
        if i % KEEP_SW == 0:
            seq.append(u)
    seq = np.array(seq[:n_knots]) if seq else np.zeros(1)
    if len(seq) < n_knots:
        seq = np.concatenate([seq, np.full(n_knots - len(seq), seq[-1])])
    return seq


class CEMSwing:
    def __init__(self):
        self.n = H_SW // KEEP_SW

    def control(self, model, data, datas):
        mu = energy_seed(model, data, datas[0], self.n)
        sigma = np.full(self.n, 3.0)
        elite_k = max(1, int(CEM_NS * CEM_ELITE))
        for _ in range(CEM_ITERS):
            samples = np.clip(mu + sigma * np.random.randn(CEM_NS, self.n),
                              -UMAX, UMAX)
            controls = np.repeat(samples, KEEP_SW, axis=1)[:, :, None]
            cost = _swing_cost(_rollout(model, datas, data, controls), controls)
            elite = samples[np.argsort(cost)[:elite_k]]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 0.2
        return float(np.clip(mu[0], -UMAX, UMAX))


def balance_mpc(model, data, datas, K):
    base = lqr_torque(K, data)
    n_knots = H_BAL // KEEP_BAL
    knots = base + np.random.normal(0.0, SIGMA, (NS_BAL, n_knots))
    knots[0, :] = base
    controls = np.repeat(np.clip(knots, -UMAX, UMAX), KEEP_BAL, axis=1)[:, :, None]
    states = _rollout(model, datas, data, controls)
    qpos, qvel = states[:, :, 1:3], states[:, :, 3:5]
    arm, pole = qpos[:, :, 0], wrap(qpos[:, :, 1])
    cost = np.sum(W_POLE * pole ** 2 + W_ARM * arm ** 2
                  + W_VEL * (qvel[:, :, 0] ** 2 + qvel[:, :, 1] ** 2)
                  + W_U * controls[:, :, 0] ** 2, axis=1)
    return float(controls[int(np.argmin(cost)), 0, 0])


def run(controller, steps, viewer, model, data, K, knock=0.0, quiet=False):
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    swing = CEMSwing()
    mode = "swing"
    hold = balanced = 0
    knocked = False
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        phi, phid = wrap(data.qpos[1]), data.qvel[1]
        if mode == "balance" and abs(phi) > FALL_PHI:
            mode = "swing"
            if not quiet:
                print(f"[step {step}] knocked -> swing", flush=True)
        elif mode == "swing" and abs(phi) < CATCH_PHI and abs(phid) < CATCH_PHID:
            mode = "balance"

        if mode == "swing":
            if step % KEEP_SW == 0:
                u = swing.control(model, data, datas)
            else:
                u = data.ctrl[0]
        elif controller == "mpc":
            u = balance_mpc(model, data, datas, K) if step % KEEP_BAL == 0 \
                else data.ctrl[0]
        else:
            u = lqr_torque(K, data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        if mode == "balance" and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= 250:
                balanced += 1
                if not quiet:
                    print(f"BALANCED #{balanced} at step {step}  "
                          f"t={step * DT:.2f}s", flush=True)
                hold = 0
                if knock > 0 and not knocked:
                    data.qvel[1] += knock
                    knocked = True
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
