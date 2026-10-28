"""Go2 抗推权重扫描：W_Z(身高) x W_UP(机身正) -> 恢复步数。

复用 go2_stand 的 MPC 与 go2_push_test 的推扰/判据，只改权重扫网格。
输出 CSV + 终端表格。用法：
  python3 go2_weight_sweep.py                 # 默认 4x4 x 3次 x 100N
  python3 go2_weight_sweep.py --reps 1 --force 100
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

Z_GRID = [20.0, 40.0, 80.0, 160.0]
UP_GRID = [15.0, 30.0, 60.0, 120.0]


def one_run(wz, wup, force, duration, settle, total, seed):
    np.random.seed(seed)
    G.W_Z, G.W_UP = wz, wup
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
        P.stand_step(model, data, "mpc", ctx)
        if not push_done and P.fallen(data):
            return None
        if push_done:
            if P.fallen(data):
                return None
            hold = hold + 1 if P.recovered(data, z_ref) else 0
            if hold >= P.HOLD_STEPS:
                return step - (settle_n + push_n)
    return None


def summarize(steps_list):
    good = [s for s in steps_list if s is not None]
    falls = len(steps_list) - len(good)
    if not good:
        return "FALL", falls
    return int(np.median(good)), falls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--total", type=float, default=6.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="/tmp/opencode/go2_sweep.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wz", "wup", "rep", "steps", "fall", "seconds"])
        for wz in Z_GRID:
            for wup in UP_GRID:
                steps_list = []
                for rep in range(args.reps):
                    t0 = time.time()
                    seed = 1000 + rep + int(wz) * 7 + int(wup) * 13
                    s = one_run(wz, wup, args.force, args.duration,
                                args.settle, args.total, seed)
                    secs = time.time() - t0
                    steps_list.append(s)
                    writer.writerow([wz, wup, rep,
                                     "" if s is None else s,
                                     1 if s is None else 0, round(secs, 1)])
                    fh.flush()
                    print(f"wz={wz} wup={wup} rep={rep} -> "
                          f"{'FALL' if s is None else str(s)+' steps'} "
                          f"({secs:.0f}s)", flush=True)
                med, falls = summarize(steps_list)
                print(f"=> wz={wz} wup={wup}: median={med} falls={falls}/{args.reps}",
                      flush=True)

    print("\n=== 汇总表 (median steps, 括号=倒地次数) ===", flush=True)
    header = "W_Z\\W_UP |" + "".join(f"{int(u):>10}" for u in UP_GRID)
    print(header)
    rows = {}
    with open(args.out) as fh:
        for row in csv.DictReader(fh):
            rows.setdefault((float(row["wz"]), float(row["wup"])), []).append(
                None if row["steps"] == "" else int(row["steps"]))
    for wz in Z_GRID:
        cells = []
        for u in UP_GRID:
            med, falls = summarize(rows.get((wz, u), []))
            cells.append(f"{med}({falls})")
        print(f"{int(wz):>8} |" + "".join(f"{c:>10}" for c in cells))
    print("done", flush=True)


if __name__ == "__main__":
    main()
