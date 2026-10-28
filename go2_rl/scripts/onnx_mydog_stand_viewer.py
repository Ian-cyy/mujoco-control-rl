#!/usr/bin/env python3
# 轻量看自建狗(mydog)站立策略: MuJoCo + onnxruntime, 无需 JAX
# obs 30 = [gyro(3), gravity(3), (q-q_home)(8), qvel(8), last_act(8)]
# 动作: 8 个力矩电机 + env 内 PD: target=q_home+act*0.1; tau=Kp*(target-q)-Kd*qd (clip ±12)
import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs",
                   "mydog", "xmls", "mydog_simple.xml")
ONNX = os.path.join(ROOT, "policies", "mydog_stand.onnx")
KP, KD, ACTION_SCALE, TORQUE_LIMIT = 40.0, 0.5, 0.1, 12.0
CTRL_DT, SIM_DT = 0.02, 0.004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--xml", default=XML)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    home = np.array(model.keyframe("home").qpos, dtype=np.float64)
    default_pose = home[7:].astype(np.float32)  # 8
    act_joints = [model.actuator_trnid[a, 0] for a in range(model.nu)]
    qadr = np.array([model.jnt_qposadr[j] for j in act_joints])
    vadr = np.array([model.jnt_dofadr[j] for j in act_joints])
    default_act = home[qadr].astype(np.float32)  # 8

    n_sub = int(round(CTRL_DT / SIM_DT))
    sess = rt.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    imu = model.site("imu").id
    last_action = np.zeros(model.nu, dtype=np.float32)

    viewer = mujoco.viewer.launch_passive(model, data)
    step = 0
    while viewer.is_running():
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[imu].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1.0])
        obs = np.hstack([
            gyro,                       # 3
            gravity,                    # 3
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
        viewer.sync()
        step += 1
        if step % 30 == 0:
            print(f"step={step} base_z={data.qpos[2]:+.3f}", flush=True)
        time.sleep(CTRL_DT)
    viewer.close()


if __name__ == "__main__":
    main()
