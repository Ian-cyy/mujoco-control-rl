"""Check left/right symmetry of mydog_simple.xml in the world frame.

For each left leg (L1 rear, L2 front) we take a set of landmark points
(hip joint anchor, knee joint anchor, foot geom center, thigh/shank mesh
centers) and compare them with the corresponding right leg (L3, L4) under
the mirror map  (x, y, z) -> (x, -y, z).

Run:  python3 scripts/mydog_symmetry_check.py [path/to/model.xml]
"""

import sys
import numpy as np
import mujoco

DEFAULT_XML = (
    "/home/cyy/mujoco_ws/go2_rl/mjplay/mujoco_playground/_src/"
    "locomotion/mydog/xmls/mydog_simple.xml"
)

PAIRS = [(1, 3, "rear "), (2, 4, "front")]


def landmarks(model, data, i):
    out = {}
    for tag, jname in (("hip", f"hip_L{i}"), ("knee", f"knee_L{i}")):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        out[tag] = data.xanchor[jid].copy()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_col{i}")
    out["foot"] = data.geom_xpos[gid].copy()
    out["thigh"] = data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"L{i}_thigh")
    ].copy()
    out["shank"] = data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"L{i}_shank")
    ].copy()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else DEFAULT_XML
    mode = "home keyframe" if "--home" in sys.argv else "zero joints (qpos0)"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    if "--home" in sys.argv and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[:] = model.qpos0
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[model.jnt_qposadr[j]] = 0.0
    mujoco.mj_forward(model, data)

    print(f"model: {path}")
    print(f"pose : {mode}\n")
    worst = 0.0
    for a, b, label in PAIRS:
        la = landmarks(model, data, a)
        lb = landmarks(model, data, b)
        print(f"[{label}] L{a} vs mirror(L{b})")
        for tag in la:
            ref = la[tag].copy()
            ref[1] *= -1.0
            d = np.abs(ref - lb[tag])
            worst = max(worst, d.max())
            print(
                f"  {tag:5s}  L{a}=({la[tag][0]:+.4f},{la[tag][1]:+.4f},{la[tag][2]:+.4f})"
                f"  L{b}=({lb[tag][0]:+.4f},{lb[tag][1]:+.4f},{lb[tag][2]:+.4f})"
                f"  |d|=({d[0]:.4f},{d[1]:.4f},{d[2]:.4f}) max={d.max():.4f}"
            )
        print()
    print(f"worst deviation = {worst:.5f} m")
    print("OK (symmetric)" if worst < 1e-4 else "NOT symmetric")


if __name__ == "__main__":
    main()
