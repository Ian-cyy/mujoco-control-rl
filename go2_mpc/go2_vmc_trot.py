#!/usr/bin/env python3
# Go2 原地 trot (A+): 足端轨迹 -> 腿IK -> 关节PD, 外加 VMC 平衡层
#   VMC: 用身体高度/姿态/速度误差 去调四足足端目标(高度/落点)
# 用法: python3 go2_vmc_trot.py --gait trot --view [--push 60] [--time 0]
import argparse
import numpy as np
import mujoco

from mpc_go2_srbd import Go2, ik_leg, bezier, LEGS

Z_REF = 0.25
T = 0.5           # 步态周期
DUTY = 0.55
LIFT = 0.08       # 抬脚高度

# VMC 增益 (先给初值, 待调)
KP_H, KD_H = 0.8, 0.08      # 高度
KP_P, KD_P = 0.2, 0.03      # 俯仰
KP_R, KD_R = 0.2, 0.03      # 横滚
KVX, KVY = 0.03, 0.03       # 落脚点(速度)补偿


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", choices=["stand", "trot"], default="trot")
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--push", type=float, default=0.0)
    args = ap.parse_args()

    robot = Go2(); M, D = robot.M, robot.D
    D.qpos[:] = M.key_qpos[0].copy(); D.qvel[:] = 0.0
    mujoco.mj_forward(M, D)

    viewer = None
    if args.view:
        from mujoco import viewer as mjv
        viewer = mjv.launch_passive(M, D)

    nom = {l: robot.foot_pos(l).copy() for l in LEGS}     # 名义落脚点(世界)
    q_des = {l: M.key_qpos[0][list(robot.jadr[l])].copy() for l in LEGS}
    Dik = mujoco.MjData(M)
    trot_pairs = {0: ["FL", "RR"], 1: ["FR", "RL"]}

    dt = M.opt.timestep
    t = 0.0; step = 0
    n = int(args.time/dt) if args.time > 0 else None

    while True:
        if n is not None and step >= n:
            break
        if viewer is not None and not viewer.is_running():
            break
        mujoco.mj_forward(M, D)
        p, rpy, v, w, R = robot.base_state()
        z, vz = p[2], v[2]
        pitch, roll = rpy[0], rpy[1]

        # ---- 步态: 哪些腿摆动, 及基础足端目标 ----
        swing = []
        if args.gait == "trot":
            ph = (t % T)/T
            pair = 0 if ph < 0.5 else 1
            swing = trot_pairs[pair]

        # ---- VMC: 由身体误差算足端修正(高度/姿态) ----
        dz = KP_H*(Z_REF - z) - KD_H*vz
        pterm = KP_P*pitch + KD_P*w[1]     # 前倾(pitch>0) -> 抬前脚
        rterm = KP_R*roll + KD_R*w[0]      # 横滚 -> 抬一侧脚

        for i, l in enumerate(LEGS):
            base = nom[l].copy()
            # 落脚点随速度微调(抗扰/防跑偏)
            base[0] += KVX*v[0]
            base[1] += KVY*v[1]
            # 高度: 四腿一起 +dz
            base[2] += dz
            # 俯仰: 前(x>0) +pterm, 后 -pterm
            base[2] += pterm if nom[l][0] > 0 else -pterm
            # 横滚: 左(y>0) +rterm, 右 -rterm
            base[2] += rterm if nom[l][1] > 0 else -rterm

            if args.gait == "trot" and l in swing:
                s = (ph % 0.5)/0.5
                foot_des = base.copy()
                foot_des[2] = base[2] + 4*LIFT*s*(1-s)
            else:
                foot_des = base
            # 安全限幅: 防止过冲/不可达
            foot_des[0] = np.clip(foot_des[0], nom[l][0]-0.06, nom[l][0]+0.06)
            foot_des[1] = np.clip(foot_des[1], nom[l][1]-0.06, nom[l][1]+0.06)
            foot_des[2] = np.clip(foot_des[2], 0.0, 0.14)

            Dik.qpos[:] = D.qpos
            q_des[l] = ik_leg(M, Dik, robot.jadr[l], robot.dadr[l],
                              robot.calf[l], foot_des, q_des[l])
            tau = np.array([120.0*(q_des[l][k]-D.qpos[robot.jadr[l][k]])
                            - 5.0*D.qvel[robot.dadr[l][k]] for k in range(3)])
            for k in range(3):
                D.ctrl[robot.aadr[l][k]] = np.clip(tau[k], -45, 45)

        mujoco.mj_step(M, D)
        D.xfrc_applied[:] = 0.0
        if args.push and 3.0 <= t < 3.1:
            D.xfrc_applied[robot.base*6 + 0] = args.push
        t += dt; step += 1
        if viewer is not None:
            viewer.sync()
        if step % int(0.5/dt) == 0:
            fz = [round(robot.foot_pos(l)[2], 3) for l in LEGS]
            print(f"t={t:4.1f} base_z={z:.3f} feet_z={fz} "
                  f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f}) v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
