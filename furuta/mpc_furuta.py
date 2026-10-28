"""旋转倒立摆（Furuta）手写 MPC（mujoco.rollout 多核并行）。

结构：底部电机带动一根水平臂绕竖直轴转(θ1)，臂端铰接一根杆(θ2, 绕水平轴)，
      初始杆接近竖直(θ2 小)，靠转动臂把杆稳在竖直。

与我们之前的直线倒立摆(CartPole)对比：
  CartPole：推小车的"力"      控制杆倒；
  Furuta  ：给臂的"力矩"      控制杆倒（非线性更强、更耦合）。

用法：
  python3 mpc_furuta.py
  python3 mpc_furuta.py --headless 2000
"""

import argparse
import os
import time

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np

FURUTA_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default><joint damping="0.002"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="arm" pos="0 0 0.15">
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
H = 120
N_SAMPLES = 600
REPEAT = 10
NTHREAD = os.cpu_count() or 4
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def mpc_action(model, data, datas):
    state = np.concatenate([[data.time], data.qpos, data.qvel])
    init = np.tile(state, (N_SAMPLES, 1))
    n_knots = H // REPEAT
    knots = np.random.uniform(-UMAX, UMAX, size=(N_SAMPLES, n_knots))
    controls = np.repeat(knots, REPEAT, axis=1)[:, :, None]
    states, _ = mujoco.rollout.rollout(
        model, datas, init, controls, persistent_pool=True)
    qpos, qvel = states[:, :, 1:3], states[:, :, 3:5]
    arm, pole = qpos[:, :, 0], wrap(qpos[:, :, 1])
    cost = np.sum(30.0 * pole ** 2 + 1.0 * arm ** 2
                  + 0.05 * qvel[:, :, 0] ** 2 + 0.05 * qvel[:, :, 1] ** 2
                  + 0.001 * controls[:, :, 0] ** 2, axis=1)
    cost += 8.0 * (30.0 * pole[:, -1] ** 2 + arm[:, -1] ** 2)
    return float(controls[int(np.argmin(cost)), 0, 0])


def pd_action(data):
    arm, pole = data.qpos[0], wrap(data.qpos[1])
    darm, dpole = data.qvel[0], data.qvel[1]
    u = 3.0 * arm + 0.3 * darm - 60.0 * pole - 4.0 * dpole
    return float(np.clip(u, -UMAX, UMAX))


def run(controller, steps, viewer, model, data):
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    u = 0.0
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        if step % REPEAT == 0:
            u = mpc_action(model, data, datas) if controller == "mpc" \
                else pd_action(data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if step % 200 == 0:
            print(f"step {step:5d}  arm={data.qpos[0]:+.3f}  "
                  f"pole={wrap(data.qpos[1]):+.3f} rad", flush=True)
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["mpc", "pd"], default="mpc")
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(FURUTA_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.05
    mujoco.mj_forward(model, data)

    if args.headless:
        run(args.controller, args.headless, None, model, data)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(args.controller, 1000000, viewer, model, data)


if __name__ == "__main__":
    main()
