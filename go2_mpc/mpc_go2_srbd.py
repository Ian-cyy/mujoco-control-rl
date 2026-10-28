#!/usr/bin/env python3
# Go2 单刚体(SRBD) MPC + 分层控制 + MuJoCo 仿真
#   gait: stand(四脚站立) / trot(对角原地踏步)
#   MPC(CasADi+IPOPT) 输出四脚接触力 -> 腿 Jacobian 映射为关节力矩; 摆动腿用足端 PD
# 用法:
#   python3 mpc_go2_srbd.py --gait stand --time 5
#   python3 mpc_go2_srbd.py --gait trot  --time 10 --view
import argparse, os, sys, time
import numpy as np
import casadi as ca
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "mujoco_menagerie", "unitree_go2", "scene.xml")

LEGS = ["FL", "FR", "RL", "RR"]
CALF_LEN = 0.213          # 足端相对 calf 原点沿 -z 的距离
MU = 0.6                  # 摩擦系数(用于 MPC 摩擦锥约束)

# -----------------------------------------------------------------------------
def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def quat_to_rpy(q):
    w,x,y,z = q
    r = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    yw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([r, p, yw])

def euler_rate(r, p):
    # ω -> rpẏ 的映射矩阵 E, rpẏ = E ω
    cp = np.cos(p); sp = np.sin(p); tp = np.tan(p)
    return np.array([[1, sp*tp, cp*tp],
                     [0, cp,    -sp],
                     [0, sp/cp, cp/cp]])

# -----------------------------------------------------------------------------
class Go2:
    def __init__(self):
        self.M = mujoco.MjModel.from_xml_path(XML)
        self.D = mujoco.MjData(self.M)
        self.base = self.M.body("base").id
        self.calf = {l: self.M.body(f"{l}_calf").id for l in LEGS}
        # 每条腿 3 个关节的 qpos/dof/act 地址
        self.jadr, self.dadr, self.aadr = {}, {}, {}
        for l in LEGS:
            qs, ds, as_ = [], [], []
            for part in ["hip", "thigh", "calf"]:
                jid = self.M.joint(f"{l}_{part}_joint").id
                qs.append(self.M.jnt_qposadr[jid])
                ds.append(self.M.jnt_dofadr[jid])
                aid = self.M.actuator(f"{l}_{part}").id
                as_.append(aid)
            self.jadr[l] = qs; self.dadr[l] = ds; self.aadr[l] = as_
        self.Jp = np.zeros((3, self.M.nv))
        self.Jr = np.zeros((3, self.M.nv))
        self.foot_off_local = np.array([0, 0, -CALF_LEN])

    def foot_pos(self, l):
        cid = self.calf[l]
        R = self.D.xmat[cid].reshape(3, 3)
        return self.D.xpos[cid] + R @ self.foot_off_local

    def foot_jac(self, l):
        mujoco.mj_jac(self.M, self.D, self.Jp, self.Jr, self.foot_pos(l), self.calf[l])
        return self.Jp[:, self.dadr[l]]        # 3x3

    def foot_vel(self, l):
        return self.foot_jac(l) @ self.D.qvel[self.dadr[l]]

    def base_state(self):
        p = self.D.qpos[0:3].copy()
        R = self.D.xmat[self.base].reshape(3, 3)
        rpy = quat_to_rpy(self.D.qpos[3:7])
        v = self.D.qvel[0:3].copy()
        w = self.D.qvel[3:6].copy()
        return p, rpy, v, w, R

# -----------------------------------------------------------------------------
class SRBD_MPC:
    """单刚体 MPC: x=[p(3),v(3),rpy(3),w(3)], u=四脚世界系接触力(12)"""
    def __init__(self, mass=15.2, I=np.array([0.08,0.24,0.24]), N=12, dt=0.03):
        self.m = mass; self.I = np.diag(I); self.Iinv = np.linalg.inv(self.I)
        self.N = N; self.dt = dt
        self.g = np.array([0, 0, -9.81])
        self._build()

    def _build(self):
        N, dt = self.N, self.dt
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 12)                          # f_FL, f_FR, f_RL, f_RR
        p, v, rpy, w = x[0:3], x[3:6], x[6:9], x[9:12]
        r, pit = rpy[0], rpy[1]
        cp, sp, tp = ca.cos(pit), ca.sin(pit), ca.tan(pit)
        E = ca.vertcat(ca.horzcat(1, sp*tp, cp*tp),
                       ca.horzcat(0, cp,   -sp),
                       ca.horzcat(0, sp/cp, cp/cp))
        F = ca.reshape(u, 3, 4)
        fsum = ca.sum2(F)
        pdot = v
        vdot = ca.vertcat(0, 0, -9.81) + fsum / self.m
        rpydot = E @ w
        # 动力学整体在 _build_opt 里构建
        self.x = x; self.u = u
        self._build_opt()

    def _build_opt(self):
        N, dt = self.N, self.dt
        X = ca.MX.sym('X', 12, N+1)
        U = ca.MX.sym('U', 12, N)
        P = ca.MX.sym('P', 12 + 12 + 12 + 4)   # x0(12) + r(12) + xref(12) + contact(4)
        # 参数布局: x0(12) | r (12, 四足相对基座) | xref(12) | contact(4) | (保留)
        x0 = P[0:12]
        Rrel = ca.reshape(P[12:24], 3, 4)            # 每腿 r_i (基座->足端, 世界)
        xref = P[24:36]
        contact = P[36:40]                            # 1=着地,0=摆动(每腿)
        # 目标权重
        Qx = ca.diag(ca.DM([50]*3 + [30]*3 + [200]*3 + [10]*3))
        Ru = ca.diag(ca.DM([1e-4]*12))
        cost = 0
        g = ca.DM([0, 0, -9.81])
        Iinv = ca.DM(self.Iinv)
        for k in range(N):
            xk = X[:, k]; uk = U[:, k]
            p = xk[0:3]; v = xk[3:6]; rpy = xk[6:9]; w = xk[9:12]
            F = ca.reshape(uk, 3, 4)
            fsum = ca.sum2(F)
            # 力矩 = Σ r_i x (R_world_i f_i) -> r_i 已在世界系(相对基座)
            tau = ca.DM.zeros(3)
            for i in range(4):
                tau += ca.cross(Rrel[:, i], F[:, i])
            vdot = g + fsum/self.m
            cp, sp, tp = ca.cos(rpy[1]), ca.sin(rpy[1]), ca.tan(rpy[1])
            E = ca.vertcat(ca.horzcat(1, sp*tp, cp*tp),
                           ca.horzcat(0, cp,   -sp),
                           ca.horzcat(0, sp/cp, cp/cp))
            wdot = Iinv @ tau
            xdot = ca.vertcat(v, vdot, E @ w, wdot)
            cost += (xk-xref).T @ Qx @ (xk-xref) + uk.T @ Ru @ uk
        # 约束: 起点 + 动力学(Euler)
        xk = X[:, 0]
        cons = [xk - x0]
        for k in range(N):
            p = X[0:3,k]; v = X[3:6,k]; rpy = X[6:9,k]; w = X[9:12,k]
            F = ca.reshape(U[:, k], 3, 4)
            tau = ca.DM.zeros(3)
            for i in range(4):
                tau += ca.cross(Rrel[:, i], F[:, i])
            cp, sp, tp = ca.cos(rpy[1]), ca.sin(rpy[1]), ca.tan(rpy[1])
            E = ca.vertcat(ca.horzcat(1, sp*tp, cp*tp),
                           ca.horzcat(0, cp,   -sp),
                           ca.horzcat(0, sp/cp, cp/cp))
            xdot = ca.vertcat(v, g + ca.sum2(F)/self.m, E @ w, Iinv @ tau)
            cons.append(X[:, k+1] - (X[:, k] + dt*xdot))
        opt = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
               'p': P, 'f': cost, 'g': ca.vertcat(*cons)}
        self.solver = ca.nlpsol('mpc', 'ipopt', opt,
                                {'ipopt': {'print_level': 0, 'max_iter': 200},
                                 'print_time': False})
        self.nX = 12*(N+1); self.nU = 12*N
        self.lbx = np.concatenate([-1e3*np.ones(self.nX), -1e3*np.ones(self.nU)])
        self.ubx = np.concatenate([ 1e3*np.ones(self.nX),  1e3*np.ones(self.nU)])
        self.lbg = np.zeros(12*(N+1)); self.ubg = np.zeros(12*(N+1))

    def solve(self, x0, rrel, xref, contact, u_guess=None):
        N = self.N
        fz_max = 300.0
        lbu = np.zeros((N, 12)); ubu = np.zeros((N, 12))
        for k in range(N):
            for i in range(4):
                b = 3*i
                if contact[i] >= 0.5:
                    lbu[k, b+0] = -MU*fz_max; ubu[k, b+0] = MU*fz_max
                    lbu[k, b+1] = -MU*fz_max; ubu[k, b+1] = MU*fz_max
                    lbu[k, b+2] = 0.0;        ubu[k, b+2] = fz_max
                else:
                    lbu[k, b:b+3] = 0.0; ubu[k, b:b+3] = 0.0
        lbx = np.concatenate([-1e3*np.ones(12*(N+1)), lbu.reshape(-1)])
        ubx = np.concatenate([ 1e3*np.ones(12*(N+1)), ubu.reshape(-1)])
        p = np.concatenate([x0, rrel.reshape(-1), xref, contact])
        guess = u_guess if u_guess is not None else np.zeros(self.nX+self.nU)
        sol = self.solver(x0=guess, p=p, lbx=lbx, ubx=ubx,
                          lbg=self.lbg, ubg=self.ubg)
        X = np.array(sol['x']).reshape(-1)
        U = X[self.nX:].reshape(N, 12)
        return U[0].reshape(3, 4), X   # 首步力(3x4), 完整解(用于热启动)

# -----------------------------------------------------------------------------
def ik_leg(M, Dik, jadr, dadr, calf_id, target, q_init, iters=8, lam=0.03):
    """单腿 3-DOF 数值 IK(阻尼最小二乘): 返回使足端到 target(世界) 的关节角"""
    q = np.array(q_init, dtype=float)
    Jp = np.zeros((3, M.nv)); Jr = np.zeros((3, M.nv))
    for _ in range(iters):
        Dik.qpos[jadr] = q
        mujoco.mj_forward(M, Dik)
        R = Dik.xmat[calf_id].reshape(3, 3)
        fp = Dik.xpos[calf_id] + R @ np.array([0, 0, -CALF_LEN])
        err = target - fp
        mujoco.mj_jac(M, Dik, Jp, Jr, fp, calf_id)
        J = Jp[:, dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(3), err)
        q = q + dq.clip(-0.3, 0.3)
    return q

# -----------------------------------------------------------------------------
def bezier(z0, z1, h, s):
    # 简单摆动足竖直轨迹: 抛物线抬 h
    return (1-s)*z0 + s*z1 + 4*h*s*(1-s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", choices=["stand", "trot"], default="stand")
    ap.add_argument("--time", type=float, default=0.0)   # 0=一直跑(关窗口才停)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--zref", type=float, default=0.27)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--ctrl", choices=["ik", "mpc"], default="ik")
    args = ap.parse_args()

    robot = Go2(); M, D = robot.M, robot.D
    mpc = SRBD_MPC() if args.ctrl == "mpc" else None
    dt_sim = M.opt.timestep
    mpc_dt = 0.03
    steps_per_mpc = max(1, int(round(mpc_dt/dt_sim)))

    # 初始: 用模型自带 home 关键帧(屈膝站姿, 避免奇异的直腿姿态)
    D.qpos[:] = M.key_qpos[0].copy()
    D.qvel[:] = 0.0
    mujoco.mj_forward(M, D)

    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(M, D)

    nom = {l: robot.foot_pos(l) for l in LEGS}    # 名义落脚点
    q_home = M.key_qpos[0].copy()                  # home 关节姿态(PD 基准)
    xref = np.zeros(12); xref[2] = args.zref
    u_prev = np.zeros((3, 4))
    guess = None
    Dik = mujoco.MjData(M)                                       # IK 专用数据
    q_des = {l: q_home[list(robot.jadr[l])].copy() for l in LEGS}  # 各腿目标关节角

    # trot 时序
    T = 0.6; duty = 0.55
    trot_pairs = {0: ["FL", "RR"], 1: ["FR", "RL"]}

    t = 0.0; last_mpc = -1e9
    n = int(args.time/dt_sim) if (args.time and args.time > 0) else None
    step = 0
    while True:
        if n is not None and step >= n:
            break
        if viewer is not None and not viewer.is_running():
            break
        # 步态/接触
        contact = np.ones(4)
        foot_des = {}
        if args.gait == "trot":
            ph = (t % T)/T
            pair = 0 if ph < 0.5 else 1
            swing = trot_pairs[pair]
            contact = np.array([0 if l in swing else 1 for l in LEGS], float)
            for i, l in enumerate(LEGS):
                if l in swing:
                    s = (ph % 0.5)/0.5
                    p0 = nom[l].copy(); 
                    foot_des[l] = np.array([p0[0], p0[1], bezier(0.0, 0.0, 0.12, s)]) + np.array([0,0,0.0])
                else:
                    foot_des[l] = nom[l]
        else:
            for l in LEGS:
                foot_des[l] = nom[l]

        # MPC (仅 mpc 模式; 每 mpc_dt)
        mujoco.mj_forward(M, D)
        if args.ctrl == "mpc" and t - last_mpc >= mpc_dt - 1e-9:
            p, rpy, v, w, R = robot.base_state()
            x0 = np.concatenate([p, v, rpy, w])
            rrel = np.zeros((3, 4))
            for i, l in enumerate(LEGS):
                rrel[:, i] = robot.foot_pos(l) - p
            f, Xsol = mpc.solve(x0, rrel, xref, contact, u_guess=guess)
            guess = Xsol; u_prev = f; last_mpc = t

        # 控制
        for i, l in enumerate(LEGS):
            if args.ctrl == "ik":
                Dik.qpos[:] = D.qpos
                q_des[l] = ik_leg(M, Dik, robot.jadr[l], robot.dadr[l],
                                  robot.calf[l], foot_des[l], q_des[l])
                tau = np.array([200.0*(q_des[l][k]-D.qpos[robot.jadr[l][k]])
                                - 6.0*D.qvel[robot.dadr[l][k]] for k in range(3)])
            else:
                J = robot.foot_jac(l)
                tau_pd = np.array([60.0*(q_home[robot.jadr[l][k]]-D.qpos[robot.jadr[l][k]])
                                   - 3.0*D.qvel[robot.dadr[l][k]] for k in range(3)])
                if args.gait == "trot" and contact[i] < 0.5:
                    fcmd = 2000.0*(foot_des[l]-robot.foot_pos(l)) - 20.0*robot.foot_vel(l)
                    tau = J.T @ fcmd + np.array([D.qfrc_bias[robot.dadr[l][k]] for k in range(3)])
                else:
                    tau = tau_pd - J.T @ u_prev[:, i]
            for k in range(3):
                D.ctrl[robot.aadr[l][k]] = np.clip(tau[k], -45, 45)

        mujoco.mj_step(M, D)
        # 抗扰: t=3.0~3.1s 沿 +x 推一下
        D.xfrc_applied[:] = 0.0
        if args.push and 3.0 <= t < 3.1:
            D.xfrc_applied[robot.base*6 + 0] = args.push
        t += dt_sim
        step += 1
        if viewer is not None:
            viewer.sync()
        if step % int(0.5/dt_sim) == 0:
            p, rpy, v, w, R = robot.base_state()
            fz = [round(robot.foot_pos(l)[2], 3) for l in LEGS]
            print(f"t={t:4.1f} base_z={p[2]:.3f} feet_z={fz} "
                  f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f}) v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})")

    if viewer is not None:
        viewer.close()

if __name__ == "__main__":
    main()
