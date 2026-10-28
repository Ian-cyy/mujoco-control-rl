"""Create mirrored copies (about local y) of the mydog leg mesh assets.

A reflection (M = diag(1,-1,1)) cannot be represented by rotating the
original STL, so the right legs need genuinely mirrored mesh geometry.
For each leg mesh  X.stl  we write  X_R.stl  with vertices y -> -y and the
triangle winding swapped so face normals stay outward.

Run:  python3 scripts/mirror_mydog_meshes.py
"""

import os
import struct
import numpy as np

ASSETS = (
    "/home/cyy/mujoco_ws/go2_rl/mjplay/mujoco_playground/_src/"
    "locomotion/mydog/xmls/assets"
)
LEG_MESHES = ["knee_conn", "motor", "thigh_in", "thigh_mid", "thigh_out", "shank", "foot"]


def mirror_stl(src, dst):
    with open(src, "rb") as f:
        header = f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        rec = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    fl = rec[:, :48].copy().view("<f4").reshape(n, 12).copy()
    attr = rec[:, 48:50].copy()

    normal = fl[:, 0:3]
    v0 = fl[:, 3:6]
    v1 = fl[:, 6:9]
    v2 = fl[:, 9:12]

    def mirror(v):
        return np.stack([v[:, 0], -v[:, 1], v[:, 2]], axis=1)

    normal = mirror(normal)
    v0, v1, v2 = mirror(v0), mirror(v1), mirror(v2)
    v1, v2 = v2, v1  # swap winding -> keep outward normals

    out = np.concatenate([normal, v0, v1, v2], axis=1).astype("<f4")
    out = np.concatenate([out.view(np.uint8).reshape(n, 48), attr], axis=1)

    with open(dst, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n))
        f.write(out.tobytes())
    return n


def main():
    for name in LEG_MESHES:
        src = os.path.join(ASSETS, f"{name}.stl")
        dst = os.path.join(ASSETS, f"{name}_R.stl")
        n = mirror_stl(src, dst)
        print(f"{name}.stl -> {name}_R.stl  ({n} tris)")


if __name__ == "__main__":
    main()
