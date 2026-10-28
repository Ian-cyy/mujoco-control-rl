"""Go2 LQR Q/R 细调：用"多方向/多时长"扰动集的生存分数。

Q = diag(w_base*[6], w_leg*[12], w_v*[nu])，R = w_u*I；w_leg/w_v 固定。
分数 = 在 DIRS x FORCES x DURS 这组扰动里"能恢复不倒"的个数。

用法：python3 go2_lqr_fine.py
"""

import mujoco

import go2_stand as G
import go2_push_test as P
from go2_stand import HOME_KEY, SCENE

WB_GRID = [5.0, 20.0, 50.0, 100.0, 300.0]
WU_GRID = [0.2, 0.5, 1.0, 2.0, 5.0]
DIRS = [(0, 1), (0, -1), (1, 1), (1, -1)]
FORCES = [300.0, 600.0]
DURS = [0.1, 0.3]


def run(wb, wu, force, dur, axis, sign, settle=1.0, total=6.0):
    G.LQR_W_BASE, G.LQR_W_U = wb, wu
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, HOME_KEY)
    home = data.qpos.copy()
    G.compute_lqr(model, home)
    bid = model.body("base").id
    dt = model.opt.timestep
    sN, pN, tN = int(settle / dt), int(dur / dt), int(total / dt)
    zref = data.qpos[2]
    done = False
    hold = 0
    for st in range(tN):
        if st == sN:
            zref = data.qpos[2]
            data.xfrc_applied[bid, axis] = sign * force
        if st == sN + pN:
            data.xfrc_applied[bid, :] = 0.0
            done = True
        data.ctrl[:] = G.lqr_ctrl(model, data)
        mujoco.mj_step(model, data)
        if not done and P.fallen(data):
            return None
        if done:
            if P.fallen(data):
                return None
            hold = hold + 1 if P.recovered(data, zref) else 0
            if hold >= P.HOLD_STEPS:
                return st - (sN + pN)
    return None


def score(wb, wu):
    n = 0
    for axis, sign in DIRS:
        for F in FORCES:
            for dur in DURS:
                if run(wb, wu, F, dur, axis, sign) is not None:
                    n += 1
    return n


def main():
    total = len(DIRS) * len(FORCES) * len(DURS)
    table = {}
    for wb in WB_GRID:
        for wu in WU_GRID:
            table[(wb, wu)] = score(wb, wu)
            print(f"wb={wb} wu={wu} -> {table[(wb, wu)]}/{total}", flush=True)
    print(f"\n=== LQR 生存分 /{total}: w_base \\ w_u ===")
    print("w_base\\w_u |" + "".join(f"{u:>7}" for u in WU_GRID))
    best = (None, -1)
    for wb in WB_GRID:
        cells = []
        for wu in WU_GRID:
            v = table[(wb, wu)]
            cells.append(f"{v:>7}")
            if v > best[1]:
                best = ((wb, wu), v)
        print(f"{int(wb):>10} |" + "".join(cells))
    print(f"\n最优: w_base={best[0][0]} w_u={best[0][1]}  分数={best[1]}/{total}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
