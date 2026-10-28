"""Go2 PD 细搜：以"临界推力"(能扛住不倒的最大推力)为指标。

对每组 (KP,KD) 用二分法找临界推力：survive(F)=能恢复且不倒。
PD 确定性、无采样，无需重复。

用法：python3 go2_pd_fine.py [--fmax 1500] [--tol 20]
"""

import argparse

import numpy as np

from go2_pd_sweep import one_run

KP_GRID = [15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0, 140.0, 220.0, 360.0]
KD_GRID = [8.0, 12.0, 16.0, 20.0, 26.0, 34.0, 46.0]


def survive(kp, kd, force):
    return one_run(kp, kd, force, 0.3, 1.0, 6.0) is not None


def critical_force(kp, kd, fmax, tol):
    lo, hi = 0.0, float(fmax)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if survive(kp, kd, mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmax", type=float, default=1500.0)
    parser.add_argument("--tol", type=float, default=20.0)
    args = parser.parse_args()

    table = {}
    for kp in KP_GRID:
        for kd in KD_GRID:
            table[(kp, kd)] = critical_force(kp, kd, args.fmax, args.tol)
            print(f"kp={kp} kd={kd} -> critical={table[(kp, kd)]:.0f}N", flush=True)

    print("\n=== 临界推力(N): KP \\ KD ===")
    print("KP\\KD |" + "".join(f"{int(d):>7}" for d in KD_GRID))
    best = (None, -1)
    for kp in KP_GRID:
        cells = []
        for kd in KD_GRID:
            v = table[(kp, kd)]
            cells.append(f"{int(v):>7}")
            if v > best[1]:
                best = ((kp, kd), v)
        print(f"{int(kp):>5} |" + "".join(cells))
    print(f"\n最优: KP={best[0][0]} KD={best[0][1]}  临界推力≈{best[1]:.0f}N")
    print("done", flush=True)


if __name__ == "__main__":
    main()
