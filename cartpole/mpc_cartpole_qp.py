"""CartPole 线性 QP-MPC（真 MPC）+ 扰动估计前馈 + 抗扰对比。

关键：外扰(推小车)与电机力作用在**同一关节**，所以模型里 x⁺=A x+B(u+d)，
      d 为等效小车力。用"一步预测残差"在线估计 d（Luenberger 型低通），
      并把 d 前馈进 QP 预测 → 车被持续推也能回 x≈0。

每步解 QP：min Σ x'Qx+u'Ru  s.t. |u|<=Umax（OSQP）。

用法：
  python3 mpc_cartpole_qp.py
  python3 mpc_cartpole_qp.py --controller pid|mpc|qp
  python3 mpc_cartpole_qp.py --compare
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import osqp
import scipy.sparse as sp

import mpc_cartpole as C

N_HORIZON = 30
Q_WEIGHT = [1.0, 100.0, 1.0, 1.0]     # [x, th, vx, dth]
R_WEIGHT = 0.01
UMAX = C.UMAX
OBS_GAIN = 0.0                        # 扰动估计低通增益（0=关闭；未调好）
D_MAX = 60.0

_A = _B = _SX = _SU = _P = None
_XPREV = None
_UPREV = 0.0
_D = 0.0


def reset_filters():
    global _XPREV, _UPREV, _D
    _XPREV = None
    _UPREV = 0.0
    _D = 0.0


def _linearize():
    global _A, _B, _SX, _SU, _P
    model = mujoco.MjModel.from_xml_string(C.CART_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    _A, _B = A, B
    n = N_HORIZON
    Sx = np.zeros((4 * n, 4))
    Su = np.zeros((4 * n, n))
    for k in range(n):
        Sx[4 * k:4 * k + 4] = np.linalg.matrix_power(A, k + 1)
        for j in range(k):
            Su[4 * k:4 * k + 4, j] = np.linalg.matrix_power(A, k - 1 - j) @ B[:, 0]
    _SX, _SU = Sx, Su
    Qbar = np.kron(np.eye(n), np.diag(Q_WEIGHT))
    _P = sp.csc_matrix(2.0 * (Su.T @ Qbar @ Su + np.eye(n) * R_WEIGHT))


def _state(data):
    return np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])


def qp_action(data):
    global _XPREV, _UPREV, _D
    x = _state(data)
    if _XPREV is not None:
        r = x - _A @ _XPREV - (_B[:, 0] * (_UPREV + _D))
        d_meas = float(_B[:, 0] @ r / (_B[:, 0] @ _B[:, 0]))
        _D = float(np.clip(_D + OBS_GAIN * (d_meas - _D), -D_MAX, D_MAX))
    dvec = np.full(N_HORIZON, _D)
    Qbar = np.kron(np.eye(N_HORIZON), np.diag(Q_WEIGHT))
    q = 2.0 * _SU.T @ Qbar @ (_SX @ x + _SU @ dvec)
    Acon = sp.csc_matrix(np.vstack([np.eye(N_HORIZON), -np.eye(N_HORIZON)]))
    lo = np.full(2 * N_HORIZON, -UMAX)
    hi = np.full(2 * N_HORIZON, UMAX)
    prob = osqp.OSQP()
    prob.setup(_P, q, Acon, lo, hi, verbose=False)
    res = prob.solve()
    u = float(np.clip(res.x[0], -UMAX, UMAX)) if res.info.status == "solved" else 0.0
    _XPREV, _UPREV = x, u
    return u


def action(controller, model, data, datas):
    if controller == "qp":
        return qp_action(data)
    if controller == "mpc":
        return C.mpc_action(model, data, datas)
    return C.pid_action(data)


def run(controller, steps, viewer, model, data):
    datas = [mujoco.MjData(model) for _ in range(C.NTHREAD)]
    reset_filters()
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        data.ctrl[0] = action(controller, model, data, datas)
        mujoco.mj_step(model, data)
        if step % 50 == 0:
            print(f"step {step:5d} x={data.qpos[0]:+.3f} angle={C.wrap(data.qpos[1]):+.3f}")
        if viewer is not None:
            viewer.sync()
            time.sleep(C.DT)


def disturb_reject(controller, force, dur=0.1, settle_s=1.0, total=8.0):
    model = mujoco.MjModel.from_xml_string(C.CART_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)
    datas = [mujoco.MjData(model) for _ in range(C.NTHREAD)]
    reset_filters()
    cid = model.body("cart").id
    dt = C.DT
    sN, pN, tN = int(settle_s / dt), int(dur / dt), int(total / dt)
    peak = 0.0
    for st in range(tN):
        if st == sN:
            data.xfrc_applied[cid, 0] = force
        if st == sN + pN:
            data.xfrc_applied[cid, :] = 0.0
        data.ctrl[0] = action(controller, model, data, datas)
        mujoco.mj_step(model, data)
        peak = max(peak, abs(C.wrap(data.qpos[1])))
        if abs(C.wrap(data.qpos[1])) > 1.2:
            return False, peak
    return True, peak


def hold_force(controller, force, secs=10.0):
    model = mujoco.MjModel.from_xml_string(C.CART_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)
    datas = [mujoco.MjData(model) for _ in range(C.NTHREAD)]
    reset_filters()
    cid = model.body("cart").id
    dt = C.DT
    for st in range(int(secs / dt)):
        if st >= 50:
            data.xfrc_applied[cid, 0] = force
        data.ctrl[0] = action(controller, model, data, datas)
        mujoco.mj_step(model, data)
        if abs(C.wrap(data.qpos[1])) > 1.2:
            return f"倒@{st * dt:.1f}s", data.qpos[0]
    return f"x={data.qpos[0]:+.2f} (d_est={_D:.1f})", data.qpos[0]


def compare(forces):
    print(f"=== 抗扰对比：推小车 {forces} N (0.1s) ===")
    for ctrl in ["pid", "mpc", "qp"]:
        limit, peak = 0, 0.0
        for F in forces:
            ok, pk = disturb_reject(ctrl, float(F))
            if ok:
                limit, peak = F, pk
            else:
                break
        print(f"{ctrl:>4}: 稳住到 {limit}N  (最大摆角 {peak:.2f})")
    print("--- 持续恒力(10s) ---")
    for F in [10, 20, 30]:
        for ctrl in ["pid", "mpc", "qp"]:
            print(f"{ctrl:>4} 持续{F}N: {hold_force(ctrl, float(F))[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["qp", "mpc", "pid"], default="qp")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()
    _linearize()
    if args.compare:
        compare([10, 20, 30, 50, 80, 120, 200])
        return
    model = mujoco.MjModel.from_xml_string(C.CART_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.3
    mujoco.mj_forward(model, data)
    try:
        if args.headless:
            run(args.controller, args.headless, None, model, data)
            return
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(args.controller, 100000, viewer, model, data)
    finally:
        mujoco.rollout.shutdown_persistent_pool()


if __name__ == "__main__":
    main()
