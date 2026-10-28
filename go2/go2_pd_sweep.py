"""Go2 PD 增益扫描：KP x KD -> 抗推恢复步数。

用法：
  python3 go2_pd_sweep.py                 # 100N, 4x4, 3次
  python3 go2_pd_sweep.py --force 300
"""

import argparse
import csv
import os
import time

import mujoco
import numpy as np

import go2_stand as G
import go2_push_test as P
from go2_stand import HOME_KEY, SCENE

KP_GRID = [60.0, 120.0, 240.0, 480.0]
KD_GRID = [2.0, 4.0, 8.0, 16.0]


def one_run(kp, kd, force, duration, settle, total):
    G.KP, G.KD = kp, kd
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)
    ctx = P.setup(model, data)
    base_id = model.body("base").id
    dt = model.opt.timestep
    settle_n = int(settle / dt)
    push_n = int(duration / dt)
    total_n = int(total / dt)

    z_ref = data.qpos[2]
    push_done = False
    hold = 0
    for step in range(total_n):
        if step == settle_n:
            z_ref = data.qpos[2]
            data.xfrc_applied[base_id, 0] = force
        if step == settle_n + push_n:
            data.xfrc_applied[base_id, :] = 0.0
            push_done = True
        data.ctrl[:] = G.pd_ctrl(model, data, ctx[0], ctx[1], ctx[2])
        mujoco.mj_step(model, data)
        if not push_done and P.fallen(data):
            return None
        if push_done:
            if P.fallen(data):
                return None
            hold = hold + 1 if P.recovered(data, z_ref) else 0
            if hold >= P.HOLD_STEPS:
                return step - (settle_n + push_n)
    return None


def summarize(vals):
    good = [s for s in vals if s is not None]
    falls = len(vals) - len(good)
    return ("FALL", falls) if not good else (int(np.median(good)), falls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--total", type=float, default=6.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="/tmp/opencode/go2_pd_sweep.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table = {}
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kp", "kd", "rep", "steps", "fall", "seconds"])
        for kp in KP_GRID:
            for kd in KD_GRID:
                vals = []
                for rep in range(args.reps):
                    t0 = time.time()
                    s = one_run(kp, kd, args.force, args.duration,
                                args.settle, args.total)
                    vals.append(s)
                    writer.writerow([kp, kd, rep, "" if s is None else s,
                                     1 if s is None else 0,
                                     round(time.time() - t0, 2)])
                    fh.flush()
                table[(kp, kd)] = summarize(vals)
    print(f"\n=== PD @ {args.force:.0f}N : KP \\ KD (median steps, 括号=倒地) ===")
    print("KP\\KD |" + "".join(f"{int(d):>9}" for d in KD_GRID))
    for kp in KP_GRID:
        cells = [f"{table[(kp, d)][0]}({table[(kp, d)][1]})" for d in KD_GRID]
        print(f"{int(kp):>5} |" + "".join(f"{c:>9}" for c in cells))
    print("(原基线 KP60 KD3 = 238步 @100N; 纯LQR = 29步 @100N)")
    print("done", flush=True)


if __name__ == "__main__":
    main()
