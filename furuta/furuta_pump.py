"""Furuta 抬高模型 + "来回摆动泵能" 摆起（去抖振 + 限臂速）。

模型：臂抬高到 z=0.70（杆能真正竖直下垂），无地面接触。
控制器：
  泵能阶段 用几种能量/摆动律（--pump sign|energy|energy_hyst）
  近竖直切 LQR 接住
目标：无接触也能把杆"来回摇上去"并稳住，且能被打倒后自恢复。

用法：python3 furuta_pump.py             # 跑几种律对比
      python3 furuta_pump.py --knock 60  # 摆起后打倒测恢复
"""

import argparse

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

RAISED_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default><joint damping="0.002"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="arm" pos="0 0 0.70">
      <joint name="arm" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="-0.05 0 0 0.25 0 0" size="0.015" rgba="0.4 0.4 0.9 1"/>
      <body name="pole" pos="0.25 0 0">
        <joint name="pole" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.012" rgba="0.9 0.4 0.4 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="arm" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

DT = 0.002
UMAX = 10.0
K_ENERGY = 8.0
ARM_DAMP = 0.6
VMAX_ARM = 8.0          # 臂速上限，超了刹车
BRAKE = 3.0
PHID_EPS = 0.6          # 泵能律的 phid 死区（防抖振）
PHI_SW, PHID_SW = 0.5, 6.0
PHI_FALL = 0.8
HOLD_STEPS = 250
POL = (0, 0, 1.0, 0.0)  # 杆的角度符号（保持正向上）
SINE_F = 0.96           # 正弦泵能频率(Hz)，≈杆自然频率

G = 9.81


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pole_energy(model, data):
    jid = model.body("pole").id
    mp = model.body_mass[jid]
    l = abs(model.body_ipos[jid][2])
    jp = model.body_inertia[jid][1] + mp * l * l
    phi, phid = data.qpos[1], data.qvel[1]
    return 0.5 * jp * phid ** 2 + mp * G * l * (np.cos(phi) - 1.0)


def lqr_gain(model, Qdiag=(10.0, 100.0, 1.0, 1.0), R=1.0):
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    n = model.nv
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, model.nu))
    mujoco.mjd_transitionFD(model, d, 1e-6, True, A, B, None, None)
    P = solve_discrete_are(A, B, np.diag(Qdiag), np.array([[R]]))
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).reshape(-1)


def pump(model, data, prev_sign, kind):
    phi, phid, av = wrap(data.qpos[1]), data.qvel[1], data.qvel[0]
    e = pole_energy(model, data)
    if kind == "sign":
        s = 1.0 if phid >= 0 else -1.0
        u = K_ENERGY * 0.6 * s
    elif kind == "energy":
        s = np.sign(phid * np.cos(phi))
        u = K_ENERGY * e * s
    else:  # energy_hyst：phid 死区内保持上一个方向，消除抖振
        if abs(phid) < PHID_EPS:
            s = prev_sign
        else:
            s = 1.0 if (phid * np.cos(phi)) >= 0 else -1.0
        u = K_ENERGY * e * s
    u = u - ARM_DAMP * av
    if abs(av) > VMAX_ARM:                      # 限臂速
        u -= BRAKE * (abs(av) - VMAX_ARM) * np.sign(av)
    return float(np.clip(u, -UMAX, UMAX)), s


def simulate(kind, knock=0.0, max_steps=60000):
    model = mujoco.MjModel.from_xml_string(RAISED_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = np.pi
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    K = lqr_gain(model)
    pumping = True
    prev_sign = 1.0
    pump_t = 0.0
    hold = 0
    first = knocked = None
    for step in range(max_steps):
        if not (np.isfinite(data.qpos[0]) and np.isfinite(data.qpos[1])):
            return "NaN", first, None
        phi, phid = wrap(data.qpos[1]), data.qvel[1]
        if pumping and abs(phi) < PHI_SW and abs(phid) < PHID_SW:
            pumping = False
        elif not pumping and abs(phi) > PHI_FALL:
            pumping = True
        if pumping:
            if kind == "sine":
                u = UMAX * np.sin(2 * np.pi * SINE_F * pump_t)
                pump_t += DT
            else:
                u, prev_sign = pump(model, data, prev_sign, kind)
        else:
            pump_t = 0.0
            x = np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
            u = float(np.clip(-K @ x, -UMAX, UMAX))
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if not pumping and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= HOLD_STEPS:
                hold = 0
                if first is None:
                    first = step
                    if knock > 0:
                        data.qvel[1] += knock
                        knocked = step
                elif knocked is not None:
                    return "REC", first, step
        else:
            hold = 0
    return "no", first, None


def hybrid(knock=0.0, kb=6.0, cap=8.0, max_steps=120000):
    """打飞 -> 泄能到最低点 -> 正弦重新摆起 -> LQR 接住。"""
    model = mujoco.MjModel.from_xml_string(RAISED_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = np.pi
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    K = lqr_gain(model)
    state = "pump"
    pump_t = 0.0
    hold = 0
    first = knocked = None
    for step in range(max_steps):
        if not (np.isfinite(data.qpos[0]) and np.isfinite(data.qpos[1])):
            return "NaN", first, None
        phi, phid, av = wrap(data.qpos[1]), data.qvel[1], data.qvel[0]
        e = pole_energy(model, data)
        if state == "balance" and abs(phi) > PHI_FALL:
            state, pump_t = "pump", 0.0
        elif state == "pump" and abs(phi) < PHI_SW and abs(phid) < PHID_SW:
            state = "balance"
        elif state == "pump" and abs(phid) > 30.0:      # 被打飞 -> 泄能
            state = "bleed"
        elif state == "bleed":
            if abs(phi) > 3.0 and abs(phid) < 1.0 and abs(av) < 1.0:
                state, pump_t = "pump", 0.0
        if state == "pump":
            u = UMAX * np.sin(2 * np.pi * SINE_F * pump_t)
            pump_t += DT
        elif state == "bleed":
            u = -kb * e * np.sign(phid) - 2.0 * av
        else:
            x = np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
            u = float(np.clip(-K @ x, -UMAX, UMAX))
        if abs(av) > cap:
            u -= 3.0 * (abs(av) - cap) * np.sign(av)
        data.ctrl[0] = float(np.clip(u, -UMAX, UMAX))
        mujoco.mj_step(model, data)
        if state == "balance" and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= HOLD_STEPS:
                hold = 0
                if first is None:
                    first = step
                    if knock > 0:
                        data.qvel[1] += knock
                        knocked = step
                elif knocked is not None:
                    return "REC", first, step
        else:
            hold = 0
    return "no", first, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knock", type=float, default=0.0)
    args = parser.parse_args()
    print("--- 正弦+泄能 混合 ---")
    print("  摆起:", hybrid(0.0), flush=True)
    for K in [60, 100, 150]:
        print(f"  knock={K}:", hybrid(float(K)), flush=True)


if __name__ == "__main__":
    main()
