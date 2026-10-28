"""CartPole 手写 MPC（Predictive Sampling，mujoco.rollout 多核并行）与 PID 对照。

MPC 四件套：测量 -> 采样 N 组控制 -> mujoco.rollout 并行预测 H 步 ->
代价打分 -> 取最优第一步（滚动）。rollout 是 C++ 批量接口，会吃多核。

用法：
  python3 mpc_cartpole.py
  python3 mpc_cartpole.py --controller pid
  python3 mpc_cartpole.py --headless 400
"""

import argparse
import os
import time

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np

CART_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 0.1 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="cart" pos="0 0 0.05">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.05"/>
      <geom type="box" size="0.1 0.05 0.05" rgba="0.4 0.4 0.9 1"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.005"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="0.9 0.4 0.4 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="force" joint="slider" gear="1" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""

DT = 0.01
UMAX = 30.0
H = 60
N_SAMPLES = 400
REPEAT = 5
NTHREAD = os.cpu_count() or 4
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS

# PID 增益（可调）
K_ANG, K_DTH, K_X, K_VX = 180.0, 10.0, 4.0, 5.0
# MPC 代价权重（可调）
W_ANG, W_X, W_VX, W_DTH, W_U = 12.0, 1.0, 0.05, 0.02, 0.001
T_ANG, T_X, T_SCALE = 20.0, 1.0, 6.0


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
    ang = wrap(qpos[:, :, 1])
    cost = np.sum(W_ANG * ang ** 2 + W_X * qpos[:, :, 0] ** 2
                  + W_VX * qvel[:, :, 0] ** 2 + W_DTH * qvel[:, :, 1] ** 2
                  + W_U * controls[:, :, 0] ** 2, axis=1)
    cost += T_SCALE * (T_ANG * ang[:, -1] ** 2 + T_X * qpos[:, -1, 0] ** 2)
    return float(controls[int(np.argmin(cost)), 0, 0])


def pid_action(data):
    ang = wrap(data.qpos[1])
    u = (K_ANG * ang + K_DTH * data.qvel[1]
         + K_X * data.qpos[0] + K_VX * data.qvel[0])
    return float(np.clip(u, -UMAX, UMAX))


def run(controller, steps, viewer, model, data):
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        u = mpc_action(model, data, datas) if controller == "mpc" else pid_action(data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if step % 50 == 0:
            print(f"step {step:5d}  x={data.qpos[0]:+.3f}  "
                  f"angle={wrap(data.qpos[1]):+.3f} rad")
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["mpc", "pid"], default="pid")
    parser.add_argument("--headless", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(CART_XML)
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
