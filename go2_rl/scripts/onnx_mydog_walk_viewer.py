#!/usr/bin/env python3
# 轻量看自建狗(mydog)直走策略: MuJoCo + onnxruntime, 无需 JAX
# obs 35 = [gyro(3), gravity(3), command(3), clock sin/cos(2), (q-q_home)(8), qvel(8), last_act(8)]
# 动作: 8 力矩电机 + env 内 PD: target=q_home+act*0.1; tau=Kp*(target-q)-Kd*qd (clip ±12)
import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs",
                   "mydog", "xmls", "mydog_simple.xml")
ONNX = os.path.join(ROOT, "policies", "mydog_walk.onnx")
KP, KD, ACTION_SCALE, TORQUE_LIMIT = 40.0, 0.5, 0.1, 12.0
CTRL_DT, SIM_DT = 0.02, 0.004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--vx", type=float, default=0.6)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    home = np.array(model.keyframe("home").qpos, dtype=np.float64)
    default_pose = home[7:].astype(np.float32)
    act_joints = [model.actuator_trnid[a, 0] for a in range(model.nu)]
    qadr = np.array([model.jnt_qposadr[j] for j in act_joints])
    vadr = np.array([model.jnt_dofadr[j] for j in act_joints])
    default_act = home[qadr].astype(np.float32)

    n_sub = int(round(CTRL_DT / SIM_DT))
    sess = rt.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    imu = model.site("imu").id
    last_action = np.zeros(model.nu, dtype=np.float32)
    cmd = np.array([args.vx, 0.0, 0.0], dtype=np.float32)
    gait_period = 0.5
    phase = 0.0

    viewer = mujoco.viewer.launch_passive(model, data)
    step = 0
    while viewer.is_running():
        gyro = data.sensor("gyro").data
        gravity = data.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        obs = np.hstack([
            gyro,                       # 3
            gravity,                    # 3
            cmd,                        # 3
            np.sin(phase),              # 1  步态时钟
            np.cos(phase),              # 1
            data.qpos[7:] - default_pose,  # 8
            data.qvel[6:],              # 8
            last_action,                # 8
        ]).astype(np.float32)
        act = sess.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        last_action = act.copy()
        target = default_act + act * ACTION_SCALE
        tau = KP * (target - data.qpos[qadr]) - KD * data.qvel[vadr]
        data.ctrl[:] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        phase = (phase + 2.0 * np.pi * CTRL_DT / gait_period) % (2.0 * np.pi)
        viewer.sync()
        step += 1
        if step % 30 == 0:
            p = data.qpos[:3]
            print(f"step={step} x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f} cmd_vx={args.vx}", flush=True)
        time.sleep(CTRL_DT)
    viewer.close()


if __name__ == "__main__":
    main()
