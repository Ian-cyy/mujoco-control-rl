#!/usr/bin/env python3
# 轻量看 Go2 跑步/蹦跳策略: MuJoCo + onnxruntime, 无需 JAX
# obs 45维 = [linvel(3), gyro(3), gravity(3), (q-q_home)(12), qvel(12), last_act(12)]
# 动作: ctrl = 默认姿态 + action * action_scale(0.5)
import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs",
                   "go2", "xmls", "scene_mjx_feetonly_flat_terrain.xml")
ONNX = os.path.join(ROOT, "policies", "go2_run.onnx")
KP, KD, ACTION_SCALE = 35.0, 0.5, 0.5
CTRL_DT, SIM_DT = 0.02, 0.004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = SIM_DT
    model.dof_damping[6:] = KD
    model.actuator_gainprm[:, 0] = KP
    model.actuator_biasprm[:, 1] = -KP
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    default = np.array(model.keyframe("home").qpos[7:], dtype=np.float32)
    n_sub = int(round(CTRL_DT / SIM_DT))
    sess = rt.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    imu = model.site("imu").id
    last_action = np.zeros(model.nu, dtype=np.float32)

    viewer = mujoco.viewer.launch_passive(model, data)
    step = 0
    while viewer.is_running():
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        gravity = data.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        obs = np.hstack([
            linvel,                     # 3
            gyro,                       # 3
            gravity,                    # 3
            data.qpos[7:] - default,    # 12
            data.qvel[6:],              # 12
            last_action,                # 12
        ]).astype(np.float32)
        act = sess.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        last_action = act.copy()
        data.ctrl[:] = default + act * ACTION_SCALE
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        if step % 30 == 0:
            p = data.qpos[:3]
            print(f"step={step} base=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})", flush=True)
        time.sleep(CTRL_DT)
    viewer.close()


if __name__ == "__main__":
    main()
