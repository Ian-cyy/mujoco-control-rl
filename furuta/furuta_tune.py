"""Furuta 调参：能量律(K_ENERGY/ARM_DAMP) 与 LQR(Q/R) -> 摆起所需步数。

指标：从下垂(qpos1=pi, qvel=0.3)开始，能量法摆起→LQR 稳住，
      返回"到平衡(BALANCED)的步数"（越小越好，None=没稳住）。

用法：python3 furuta_tune.py
"""

import mujoco

import furuta_swingup as F

MAX_STEPS = 40000


def simulate():
    model = mujoco.MjModel.from_xml_string(F.FURUTA_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 3.141592653589793
    data.qvel[1] = 0.3
    mujoco.mj_forward(model, data)
    K, _ = F.lqr_gain(model)
    swinging = True
    hold = 0
    for step in range(MAX_STEPS):
        phi, phid = F.wrap(data.qpos[1]), data.qvel[1]
        if swinging and abs(phi) < F.SW_PHI and abs(phid) < F.SW_PHID:
            swinging = False
        if not swinging and abs(phi) > F.FALL_PHI:
            swinging = True
        u = F.energy_torque(model, data) if swinging else F.lqr_torque(K, data)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if not swinging and abs(phi) < 0.1 and abs(phid) < 1.0:
            hold += 1
            if hold >= F.HOLD_STEPS:
                return step
        else:
            hold = 0
    return None


def sweep_energy():
    print("=== K_ENERGY x ARM_DAMP -> 摆起步数 ===")
    best = (None, 1 << 30)
    for ke in [4.0, 8.0, 12.0, 16.0, 24.0]:
        row = []
        for b in [0.5, 1.0, 2.0, 3.0, 4.0]:
            F.K_ENERGY, F.ARM_DAMP = ke, b
            s = simulate()
            row.append(f"b{b}:{'X' if s is None else s}")
            if s is not None and s < best[1]:
                best = ((ke, b), s)
        print(f"KE={int(ke):>2} | " + " ".join(row))
    print("best energy:", best)
    F.K_ENERGY, F.ARM_DAMP = best[0]
    return best


def sweep_lqr():
    print("=== LQR_Q(pole) x LQR_R -> 摆起步数 (用最优能量律) ===")
    F.LQR_Q = [10.0, 100.0, 1.0, 1.0]
    best = (None, 1 << 30)
    for qp in [20.0, 100.0, 500.0]:
        row = []
        for r in [0.5, 1.0, 2.0, 5.0]:
            F.LQR_Q = [10.0, qp, 1.0, 1.0]
            F.LQR_R = r
            s = simulate()
            row.append(f"R{r}:{'X' if s is None else s}")
            if s is not None and s < best[1]:
                best = ((qp, r), s)
        print(f"Qpole={int(qp):>3} | " + " ".join(row))
    print("best lqr:", best)
    return best


if __name__ == "__main__":
    sweep_energy()
    sweep_lqr()
