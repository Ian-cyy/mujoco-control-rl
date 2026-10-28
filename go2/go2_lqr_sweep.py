"""Go2 LQR 的 Q/R 扫描：w_base(机身与姿态) x w_u(用力=R) -> 抗推恢复步数。

Q = diag(w_base*[6], w_leg*[12], w_v*[nu])，R = w_u*I；w_leg/w_v 固定。
控制器可用纯 LQR 或 LQR 为底的 MPC（--controller lqr|mpc）。

用法：
  python3 go2_lqr_sweep.py --controller lqr
  python3 go2_lqr_sweep.py --controller mpc
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

WB_GRID = [20.0, 100.0, 500.0]
WU_GRID = [0.2, 1.0, 5.0]


def one_run(wb, wu, controller, force, duration, settle, total, seed):
    np.random.seed(seed)
    G.LQR_W_BASE, G.LQR_W_U = wb, wu
    G.MPC_BASE = "lqr" if controller == "mpc" else "pd"
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)
    home_full = data.qpos.copy()
    G.compute_lqr(model, home_full)
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
        if controller == "lqr":
            data.ctrl[:] = G.lqr_ctrl(model, data)
        else:
            data.ctrl[:] = G.mpc_ctrl(model, data, ctx[4], ctx[0], ctx[1],
                                      ctx[2], ctx[3])
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


def summarize(steps_list):
    good = [s for s in steps_list if s is not None]
    falls = len(steps_list) - len(good)
    if not good:
        return "FALL", falls
    return int(np.median(good)), falls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["lqr", "mpc"], default="lqr")
    parser.add_argument("--force", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--total", type=float, default=6.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="/tmp/opencode/go2_lqr_sweep.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table = {}
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wb", "wu", "rep", "steps", "fall", "seconds"])
        for wb in WB_GRID:
            for wu in WU_GRID:
                steps_list = []
                for rep in range(args.reps):
                    t0 = time.time()
                    s = one_run(wb, wu, args.controller, args.force,
                                args.duration, args.settle, args.total,
                                1000 + rep + int(wb) * 7 + int(wu * 100) * 13)
                    secs = time.time() - t0
                    steps_list.append(s)
                    writer.writerow([wb, wu, rep,
                                     "" if s is None else s,
                                     1 if s is None else 0, round(secs, 2)])
                    fh.flush()
                    print(f"[{args.controller}] wb={wb} wu={wu} rep={rep} -> "
                          f"{'FALL' if s is None else str(s)+' steps'} "
                          f"({secs:.1f}s)", flush=True)
                table[(wb, wu)] = summarize(steps_list)

    print(f"\n=== {args.controller}: w_base \\ w_u (median steps, 括号=倒地) ===")
    print("w_base\\w_u |" + "".join(f"{u:>10}" for u in WU_GRID))
    for wb in WB_GRID:
        cells = [f"{table[(wb, u)][0]}({table[(wb, u)][1]})" for u in WU_GRID]
        print(f"{int(wb):>10} |" + "".join(f"{c:>10}" for c in cells))
    print("(PD 基线: 238 步 @100N)")
    print("done", flush=True)


if __name__ == "__main__":
    main()
