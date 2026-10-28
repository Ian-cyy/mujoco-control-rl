"""CartPole 调参：让"杆保持竖直 + 小车回中"，对比 PID / MPC。

指标：初始杆角 0.3，跑 STEPS 步；记录
  max_ang = 1s 之后 |杆角| 的最大值（越小越稳，>0.3 视为倒）
  final_x = 末了 |小车位置|（越小越回中）
loss = (0 if max_ang<0.3 else 10) + max_ang*10 + final_x

用法：python3 cartpole_tune.py            # 扫 PID 和 MPC
      python3 cartpole_tune.py --pid      # 只 PID
      python3 cartpole_tune.py --mpc      # 只 MPC
"""

import argparse

import mujoco
import numpy as np

import mpc_cartpole as C

STEPS = 600


def simulate(controller, init_ang=0.3):
    model = mujoco.MjModel.from_xml_string(C.CART_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = init_ang
    mujoco.mj_forward(model, data)
    datas = [mujoco.MjData(model) for _ in range(C.NTHREAD)]
    max_ang = 0.0
    for step in range(STEPS):
        u = C.mpc_action(model, data, datas) if controller == "mpc" \
            else C.pid_action(data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if step > 100:
            max_ang = max(max_ang, abs(C.wrap(data.qpos[1])))
    final_x = abs(data.qpos[0])
    return max_ang, final_x


def loss(max_ang, final_x):
    return (0.0 if max_ang < 0.3 else 10.0) + max_ang * 10.0 + final_x


def tune_pid():
    print("=== PID: 固定 K_ANG=120,K_DTH=12，扫 K_X x K_VX ===")
    C.K_ANG, C.K_DTH = 120.0, 12.0
    best = (None, 1e9)
    for kx in [0.0, 1.5, 3.0, 6.0, 12.0, 24.0]:
        row = []
        for kvx in [0.0, 1.0, 2.0, 4.0, 8.0]:
            C.K_X, C.K_VX = kx, kvx
            ma, fx = simulate("pid")
            lo = loss(ma, fx)
            row.append(f"kvx{int(kvx)}:{lo:.2f}")
            if lo < best[1]:
                best = ((kx, kvx), lo)
        print(f"kx={int(kx):>3} | " + "  ".join(row))
    print("best PID:", best)
    C.K_X, C.K_VX = best[0]
    return best


def tune_mpc():
    print("=== MPC: 扫 W_X x T_X (其余固定) ===")
    best = (None, 1e9)
    for wx in [1.0, 5.0, 20.0, 50.0]:
        row = []
        for tx in [1.0, 5.0, 20.0]:
            C.W_X, C.T_X = wx, tx
            ma, fx = simulate("mpc")
            lo = loss(ma, fx)
            row.append(f"tx{int(tx)}:{lo:.2f}")
            if lo < best[1]:
                best = ((wx, tx), lo)
        print(f"wx={int(wx):>3} | " + "  ".join(row))
    print("best MPC:", best)
    C.W_X, C.T_X = best[0]
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", action="store_true")
    parser.add_argument("--mpc", action="store_true")
    args = parser.parse_args()
    print("baseline PID:", loss(*simulate("pid")))
    print("baseline MPC:", loss(*simulate("mpc")))
    if args.pid or not args.mpc:
        tune_pid()
    if args.mpc or not args.pid:
        tune_mpc()


if __name__ == "__main__":
    main()
