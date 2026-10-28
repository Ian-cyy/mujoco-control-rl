"""Go2 抗推实验：定时施加外力 + 自动打分（恢复时间）。

控制器/代价直接复用 go2_stand.py，所以你只要改 go2_stand.py 里的
权重 W_Z / W_UP / W_POSE / W_U，再跑本脚本就能看到"恢复时间"变化。

打分：推动后 base_z / 机身姿态回到推动前稳态所需步数（越少越抗推）；
      若 base_z < 0.12 或机身倾斜过大 -> FALLEN。

用法：
  python3 go2_push_test.py --controller pd  --headless --force 100
  python3 go2_push_test.py --controller mpc --headless --force 100
  python3 go2_push_test.py --controller mpc --force 100     # viewer，定时推
  viewer 里也能用 Ctrl+右键手动推
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from go2_stand import (
    HOME_KEY,
    NTHREAD,
    SCENE,
    actuator_joint_addrs,
    mpc_ctrl,
    pd_ctrl,
    upright_error,
)

Z_TOL = 0.01
UPRIGHT_TOL = 0.01
HOLD_STEPS = 30
FALL_Z = 0.12
FALL_UPRIGHT = 0.5


def setup(model, data):
    qadr, dadr = actuator_joint_addrs(model)
    q_home = data.qpos[qadr].copy()
    q_leg_home = data.qpos[7:19].copy()
    datas = [mujoco.MjData(model) for _ in range(NTHREAD)]
    return qadr, dadr, q_home, q_leg_home, datas


def stand_step(model, data, controller, ctx):
    qadr, dadr, q_home, q_leg_home, datas = ctx
    if controller == "mpc":
        data.ctrl[:] = mpc_ctrl(model, data, datas, qadr, dadr,
                                q_home, q_leg_home)
    else:
        data.ctrl[:] = pd_ctrl(model, data, qadr, dadr, q_home)
    mujoco.mj_step(model, data)


def recovered(data, z_ref):
    return abs(data.qpos[2] - z_ref) < Z_TOL and upright_error(data) < UPRIGHT_TOL


def fallen(data):
    return data.qpos[2] < FALL_Z or upright_error(data) > FALL_UPRIGHT


def run_headless(model, data, args, ctx):
    base_id = model.body("base").id
    dt = model.opt.timestep
    settle = int(args.settle / dt)
    push_len = int(args.duration / dt)
    total = int(args.total / dt)

    push_done = False
    hold = 0
    result = "NOT RECOVERED"
    z_ref = data.qpos[2]
    for step in range(total):
        if step == settle:
            z_ref = data.qpos[2]
            data.xfrc_applied[base_id, 0] = args.force
        if step == settle + push_len:
            data.xfrc_applied[base_id, :] = 0.0
            push_done = True
        stand_step(model, data, args.controller, ctx)

        if not push_done and fallen(data):
            result = "FALLEN DURING PUSH"
            break
        if push_done:
            if fallen(data):
                result = "FALLEN AFTER PUSH"
                break
            hold = hold + 1 if recovered(data, z_ref) else 0
            if hold >= HOLD_STEPS:
                result = f"RECOVERED in {step - (settle + push_len)} steps"
                break
    print(f"controller={args.controller} force={args.force}N "
          f"dur={args.duration}s  ->  {result}")


def run_viewer(model, data, args, ctx):
    base_id = model.body("base").id
    dt = model.opt.timestep
    period = int(args.period / dt)
    push_len = int(args.duration / dt)
    tick = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if tick % period == 0:
                data.xfrc_applied[base_id, 0] = args.force
            if tick % period == push_len:
                data.xfrc_applied[base_id, :] = 0.0
            stand_step(model, data, args.controller, ctx)
            viewer.sync()
            time.sleep(dt)
            tick += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["pd", "mpc"], default="mpc")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--force", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--total", type=float, default=6.0)
    parser.add_argument("--period", type=float, default=1.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)
    ctx = setup(model, data)

    if args.headless:
        run_headless(model, data, args, ctx)
    else:
        run_viewer(model, data, args, ctx)


if __name__ == "__main__":
    main()
