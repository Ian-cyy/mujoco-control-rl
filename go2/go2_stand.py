"""Go2 四足"站住"：PD 基线 vs 预测采样 MPC（mujoco.rollout 多核并行）。

PD：按 home keyframe 的关节角做位置伺服（kp/kd）。
MPC：在 PD 输出上叠加随机采样的力矩，用 mujoco.rollout 并行预测 H 步，
     按"身高够、机身正、姿态贴 home、别乱用力"打分，取最优第一步。

用法：
  python3 go2_stand.py
  python3 go2_stand.py --controller mpc
  python3 go2_stand.py --headless 1500
"""

import argparse
import os
import time

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(_ROOT, "mujoco_menagerie", "unitree_go2", "scene.xml")
HOME_KEY = 0
TARGET_Z = 0.27
KP = 300.0
KD = 150.0
PD_FF = False          # True: 重力前馈 (τ += qfrc_bias)
N_SAMPLES = 200
H = 15
SIGMA = 4.0
NTHREAD = os.cpu_count() or 4
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
W_Z, W_UP, W_POSE, W_U = 40.0, 30.0, 0.5, 0.001

LQR_W_BASE, LQR_W_LEG, LQR_W_V, LQR_W_U = 100.0, 1.0, 1.0, 0.5
LQR_K = None
LQR_HOME = None
MPC_BASE = "pd"


def compute_lqr(model, q_home):
    """在 home 位姿线性化求 LQR 增益 K (12x36)，u = -K (x - x_home)。"""
    global LQR_K, LQR_HOME
    LQR_HOME = np.array(q_home, dtype=float).copy()
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = q_home
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    n = 2 * model.nv
    A = np.zeros((n, n))
    B = np.zeros((n, model.nu))
    mujoco.mjd_transitionFD(model, scratch, 1e-6, True, A, B, None, None)
    Q = np.diag(np.concatenate([np.full(6, LQR_W_BASE),
                                np.full(12, LQR_W_LEG),
                                np.full(model.nv, LQR_W_V)]))
    R = np.eye(model.nu) * LQR_W_U
    P = solve_discrete_are(A, B, Q, R)
    LQR_K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return LQR_K


def _state_err(model, data, q_home):
    pos = data.qpos[0:3] - q_home[0:3]
    rot = np.zeros(3)
    mujoco.mju_subQuat(rot, data.qpos[3:7], q_home[3:7])
    leg = data.qpos[7:19] - q_home[7:19]
    return np.concatenate([pos, rot, leg, data.qvel])


def lqr_ctrl(model, data):
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    u = -(LQR_K @ _state_err(model, data, LQR_HOME))
    return np.clip(u, lo, hi)


def actuator_joint_addrs(model):
    qadr, dadr = [], []
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
    return np.array(qadr), np.array(dadr)


def pd_ctrl(model, data, qadr, dadr, q_home, offset=None):
    tau = KP * (q_home - data.qpos[qadr]) - KD * data.qvel[dadr]
    if PD_FF:
        tau = tau + data.qfrc_bias[dadr]      # 重力/科氏前馈
    if offset is not None:
        tau = tau + offset
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return np.clip(tau, lo, hi)


def upright_error(data):
    w, x, y, z = data.qpos[3:7]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return (1.0 - up_z) ** 2


def mpc_ctrl(model, data, datas, qadr, dadr, q_home, q_leg_home):
    if MPC_BASE == "lqr":
        base = lqr_ctrl(model, data)
    else:
        base = pd_ctrl(model, data, qadr, dadr, q_home)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    delta = np.random.normal(0.0, SIGMA, size=(N_SAMPLES, model.nu))
    delta[0, :] = 0.0                    # 保留"纯 base(LQR/PD)"候选，保证不劣于它
    u0 = np.clip(base[None, :] + delta, lo, hi)
    controls = np.repeat(u0[:, None, :], H, axis=1)

    state = np.concatenate([[data.time], data.qpos, data.qvel])
    init = np.tile(state, (N_SAMPLES, 1))
    states, _ = mujoco.rollout.rollout(
        model, datas, init, controls, persistent_pool=True)

    if MPC_BASE == "lqr" and LQR_HOME is not None:
        cost = _lqr_aligned_cost(model, states, controls, LQR_HOME)
    else:
        qpos = states[:, :, 1:1 + model.nq]
        z = qpos[:, :, 2]
        qx, qy = qpos[:, :, 4], qpos[:, :, 5]
        upright = (2.0 * (qx * qx + qy * qy)) ** 2
        leg = qpos[:, :, 7:19] - q_leg_home[None, None, :]
        cost = np.sum(W_Z * (z - TARGET_Z) ** 2 + W_UP * upright
                      + W_POSE * np.sum(leg ** 2, axis=2)
                      + W_U * np.sum(controls ** 2, axis=2), axis=1)
    return controls[int(np.argmin(cost)), 0, :]


def _lqr_aligned_cost(model, states, controls, home):
    """与 LQR 一致的代价：x_err^T Q x_err + u^T R u。"""
    qpos = states[:, :, 1:1 + model.nq]
    qvel = states[:, :, 1 + model.nq:]
    pos = qpos[:, :, 0:3] - home[0:3]
    rot = 2.0 * qpos[:, :, 4:7]                 # home 四元数为单位，小角近似
    leg = qpos[:, :, 7:19] - home[7:19]
    base_term = LQR_W_BASE * (np.sum(pos ** 2, axis=2) + np.sum(rot ** 2, axis=2))
    leg_term = LQR_W_LEG * np.sum(leg ** 2, axis=2)
    vel_term = LQR_W_V * np.sum(qvel ** 2, axis=2)
    u_term = LQR_W_U * np.sum(controls ** 2, axis=2)
    return np.sum(base_term + leg_term + vel_term + u_term, axis=1)


def run(controller, steps, viewer, model, data):
    qadr, dadr = actuator_joint_addrs(model)
    q_home = data.qpos[qadr].copy()
    q_leg_home = data.qpos[7:19].copy()
    home_full = data.qpos.copy()
    if controller == "lqr" or MPC_BASE == "lqr":
        compute_lqr(model, home_full)
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        if controller == "mpc":
            data.ctrl[:] = mpc_ctrl(model, data, datas, qadr, dadr,
                                    q_home, q_leg_home)
        elif controller == "lqr":
            data.ctrl[:] = lqr_ctrl(model, data)
        else:
            data.ctrl[:] = pd_ctrl(model, data, qadr, dadr, q_home)
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            print(f"step {step:5d}  base_z={data.qpos[2]:.3f}  "
                  f"upright_err={upright_error(data):.3f}")
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["pd", "mpc", "lqr"], default="pd")
    parser.add_argument("--mpc-base", choices=["pd", "lqr"], default="pd")
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()
    global MPC_BASE
    MPC_BASE = args.mpc_base

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)

    try:
        if args.headless:
            run(args.controller, args.headless, None, model, data)
            return
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(args.controller, 1000000, viewer, model, data)
    finally:
        mujoco.rollout.shutdown_persistent_pool()


if __name__ == "__main__":
    main()
