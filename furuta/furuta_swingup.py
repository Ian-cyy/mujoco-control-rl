"""旋转倒立摆：正弦"来回摇臂"摆起 + LQR 接住（抬高模型，无地面接触）。

- 模型：臂抬到 z=0.70，杆能真正竖直下垂（老模型太矮、杆会戳地）。
- 摆起：**闭环泵能** u=k·φ̇·cosφ·clip((-E)/Esc,0,1)（能量接近目标就停手），
        被扰动/打飞后也能重新泵起；
- 接住：杆接近竖直且慢(|pole|小、|pole_v|小) → 切 LQR 稳住；
- viewer 模式平衡后不退出；被推倒(|pole|>FALL)会重新摇起（实测 knock≤150 均可恢复）。

LQR：竖直点用 mjd_transitionFD 数值线性化 + scipy 解离散 Riccati。

用法：
  python3 furuta_swingup.py
  python3 furuta_swingup.py --headless 20000

（仍导出 energy_torque / pole_energy / FURUTA_XML 等，供 furuta_lqr_mpc 复用。）
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

from mpc_furuta import DT, FURUTA_XML, UMAX, wrap  # noqa: F401

G = 9.81
SW_PHI = 0.5
SW_PHID = 6.0
FALL_PHI = 0.8
HOLD_STEPS = 250
K_ENERGY = 8.0
ARM_DAMP = 1.0
# 闭环泵能：u = PUMP_K·φ̇·cosφ·clip((-E)/PUMP_ESC,0,1)（能量接近目标即停手）
PUMP_K = 80.0
PUMP_ESC = 0.2
PUMP_VCAP = 8.0              # 臂速上限
PUMP_BRAKE = 3.0

LQR_Q = [10.0, 50000.0, 1.0, 1.0]   # [arm, pole, darm, dpole]（调优值）
LQR_R = 0.5

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


def lqr_gain(model):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    n = model.nv
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    Q = np.diag(LQR_Q)
    R = np.array([[LQR_R]])
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig = np.abs(np.linalg.eigvals(A - B @ K))
    return K.reshape(-1), float(eig.max())


def pole_energy(model, data):
    jid = model.body("pole").id
    mp = model.body_mass[jid]
    l = abs(model.body_ipos[jid][2])
    iy = model.body_inertia[jid][1]
    jp = iy + mp * l * l
    phi, phid = data.qpos[1], data.qvel[1]
    return 0.5 * jp * phid ** 2 + mp * G * l * (np.cos(phi) - 1.0)


def energy_torque(model, data):
    e = pole_energy(model, data)
    phi, phid = data.qpos[1], data.qvel[1]
    u = K_ENERGY * e * np.sign(phid) * np.cos(phi) - ARM_DAMP * data.qvel[0]
    return float(np.clip(u, -UMAX, UMAX))


def lqr_torque(K, data):
    x = np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])
    return float(np.clip(-K @ x, -UMAX, UMAX))


def pump_torque(model, data):
    """闭环泵能：强泵 k·φ̇·cosφ，能量接近目标(0)时减小到停。"""
    phi, phid, av = wrap(data.qpos[1]), data.qvel[1], data.qvel[0]
    e = pole_energy(model, data)
    fac = min(1.0, max(0.0, -e / PUMP_ESC))
    u = PUMP_K * phid * np.cos(phi) * fac - ARM_DAMP * av
    if abs(av) > PUMP_VCAP:
        u -= PUMP_BRAKE * (abs(av) - PUMP_VCAP) * np.sign(av)
    return float(np.clip(u, -UMAX, UMAX))


def run(steps, viewer, model, data, K):
    pumping = True
    hold = 0
    settled = False
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        phi, phid = wrap(data.qpos[1]), data.qvel[1]
        if pumping and abs(phi) < SW_PHI and abs(phid) < SW_PHID:
            pumping = False
            print(f"[step {step}] switch to LQR", flush=True)
        elif not pumping and abs(phi) > FALL_PHI:
            pumping = True
            settled = False
            print(f"[step {step}] disturbed -> pump", flush=True)
        if pumping:
            u = pump_torque(model, data)
        else:
            u = lqr_torque(K, data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        if not pumping and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= HOLD_STEPS:
                hold = 0
                if not settled:
                    settled = True
                    print(f"BALANCED at step {step}  t={step * DT:.2f}s",
                          flush=True)
                    if viewer is None:
                        return
        else:
            hold = 0
        if step % 500 == 0:
            print(f"step {step:6d}  arm={data.qpos[0]:+.3f}  "
                  f"pole={phi:+.3f}  {'swing' if pumping else 'lqr'}", flush=True)
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)
    print("NOT BALANCED within steps", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(RAISED_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = np.pi
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    K, maxeig = lqr_gain(model)
    print(f"LQR K={np.round(K,3)}  |closed-loop|max={maxeig:.3f}", flush=True)

    if args.headless:
        run(args.headless, None, model, data, K)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(1000000, viewer, model, data, K)


if __name__ == "__main__":
    main()
