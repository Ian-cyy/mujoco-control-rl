#!/usr/bin/env python3
# 轻量看 Go2: 普通 MuJoCo + onnxruntime 跑导出的策略(无需 JAX, 秒开)
import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs", "go2", "xmls", "scene_mjx_feetonly_flat_terrain.xml")
ONNX = os.path.join(ROOT, "policies", "go2_policy.onnx")
KP, KD, ACTION_SCALE = 35.0, 0.5, 0.5
CTRL_DT, SIM_DT = 0.02, 0.004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = SIM_DT
    model.dof_damping[6:] = KD
    model.actuator_gainprm[:, 0] = KP
    model.actuator_biasprm[:, 1] = -KP
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    default = np.array(model.keyframe("home").qpos[7:], dtype=np.float32)
    n_sub = int(round(CTRL_DT / SIM_DT))
    sess = rt.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    last_action = np.zeros(model.nu, dtype=np.float32)
    cmd = np.array([args.vx, 0.0, args.wz], dtype=np.float32)

    # 施力用 MuJoCo 内置: 按住 Ctrl + 右键在机器人上拖动即可加力
    # (注意: 不能每步清零 xfrc_applied, 否则会把内置施力抹掉)
    viewer = mujoco.viewer.launch_passive(model, data)
    step = 0
    while viewer.is_running():
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        jang = data.qpos[7:] - default
        jvel = data.qvel[6:]
        obs = np.hstack([linvel, gyro, gravity, jang, jvel, last_action, cmd]).astype(np.float32)
        act = sess.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        last_action = act.copy()
        data.ctrl[:] = act * ACTION_SCALE + default
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        if step % 30 == 0:
            p = data.qpos[:3]
            print(f"step={step} base=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) cmd=({args.vx})", flush=True)
        time.sleep(CTRL_DT)
    viewer.close()


if __name__ == "__main__":
    main()
