#!/usr/bin/env python3
# Go2 多技能切换 viewer: walk(48维) + getup(42维), MuJoCo + onnxruntime, 无需 JAX
# 状态机: 正常->walk; 检测到翻倒->getup; 站起来->切回 walk
# 施力: 按住 Ctrl + 右键在机器人上拖动 (把它推倒 -> 看它自己爬起再继续走)
import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs",
                   "go2", "xmls", "scene_mjx_fullcollisions_flat_terrain.xml")
WALK_ONNX = os.path.join(ROOT, "policies", "go2_policy.onnx")
GETUP_ONNX = os.path.join(ROOT, "policies", "go2_getup.onnx")
KP, KD, ACTION_SCALE = 35.0, 0.5, 0.5
CTRL_DT, SIM_DT = 0.02, 0.004

# 姿态检测阈值
UPRIGHT_GZ = -0.6      # gravity_z 小于此值算"基本直立"
FALLEN_GZ = -0.4       # 大于此值算"翻了"
FALLEN_HEIGHT = 0.15   # imu 高度低于此算"趴下"
RECOVER_HEIGHT = 0.25  # imu 高度高于此且直立算"站起来了"
RECOVER_HOLD = 20      # 连续这么多拍才切回 walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default=WALK_ONNX)
    ap.add_argument("--getup", default=GETUP_ONNX)
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--wz", type=float, default=0.0)
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
    imu = model.site("imu").id
    cmd = np.array([args.vx, 0.0, args.wz], dtype=np.float32)

    sess_walk = rt.InferenceSession(args.walk, providers=["CPUExecutionProvider"])
    sess_getup = rt.InferenceSession(args.getup, providers=["CPUExecutionProvider"])
    last_walk = np.zeros(model.nu, dtype=np.float32)
    last_getup = np.zeros(model.nu, dtype=np.float32)

    viewer = mujoco.viewer.launch_passive(model, data)
    state = "walk"      # walk | getup
    recover_cnt = 0
    step = 0

    while viewer.is_running():
        gyro = data.sensor("gyro").data
        gravity = data.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        q = data.qpos[7:]
        dq = data.qvel[6:]
        torso_z = data.site_xpos[imu][2]
        upright = gravity[2] < UPRIGHT_GZ
        fallen = (gravity[2] > FALLEN_GZ) or (torso_z < FALLEN_HEIGHT)

        if state == "walk":
            if fallen:
                state = "getup"
                last_getup[:] = 0.0
                recover_cnt = 0
                print(f"[{step}] FALL detected -> getup", flush=True)
            else:
                linvel = data.sensor("local_linvel").data
                obs = np.hstack([linvel, gyro, gravity, q - default, dq,
                                 last_walk, cmd]).astype(np.float32)
                act = sess_walk.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
                last_walk = act.copy()
                data.ctrl[:] = default + act * ACTION_SCALE
        else:  # getup
            if upright and torso_z > RECOVER_HEIGHT:
                recover_cnt += 1
                if recover_cnt >= RECOVER_HOLD:
                    state = "walk"
                    last_walk[:] = 0.0
                    print(f"[{step}] RECOVERED -> walk", flush=True)
            else:
                recover_cnt = 0
            obs = np.hstack([gyro, gravity, q - default, dq,
                             last_getup]).astype(np.float32)
            act = sess_getup.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
            last_getup = act.copy()
            data.ctrl[:] = q + act * ACTION_SCALE

        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        if step % 30 == 0:
            p = data.qpos[:3]
            print(f"step={step} state={state} base=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                  f"torso_z={torso_z:.2f} gz={gravity[2]:+.2f}", flush=True)
        time.sleep(CTRL_DT)
    viewer.close()


if __name__ == "__main__":
    main()
