"""Make mydog_simple.xml left/right symmetric by mirroring the left legs.

The two left legs (L1 rear, L2 front) are treated as the reference.  The
right legs are rebuilt as exact reflections about the body's sagittal
x-z plane (the map  y -> -y ), so L3 <- mirror(L1) and L4 <- mirror(L2).

Transform rules applied recursively to every element of the reference
subtree (all leg body frames are parallel to the base frame, so a plain
reflection is correct):
    pos        (x, y, z)          -> (x, -y, z)
    quat       (w, x, y, z)       -> (w, -x, y, -z)      [ M R M ]
    fromto     x1 y1 z1 x2 y2 z2  -> y1,y2 negated
    axis       (x, y, z)          -> (x, -y, z)
    fullinertia Ixy, Iyz          -> negated
    euler      (x, y, z)          -> (x, -y, z)

Names are rewritten only inside the produced subtree (L1->L3, col1->col3)
because the actuator and sensor sections already reference the L3/L4 names.

Usage:
    python3 scripts/mirror_mydog_legs.py [--write]
"""

import copy
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XML = os.path.join(ROOT, "envs", "mydog", "xmls", "mydog_simple.xml")

# (reference body, target body, name rewrites inside the reflected subtree)
JOBS = [
    ("L1_thigh", "L3_thigh", [("L1", "L3"), ("col1", "col3")]),
    ("L2_thigh", "L4_thigh", [("L2", "L4"), ("col2", "col4")]),
]

# meshes used by the legs; the right legs use mirrored copies  X.stl -> X_R.stl
LEG_MESHES = ["knee_conn", "motor", "thigh_in", "thigh_mid", "thigh_out", "shank", "foot"]


def _floats(s):
    return [float(x) for x in s.split()]


def _fmt(v, nd=8):
    return " ".join(f"{x:.{nd}f}" for x in v)


def reflect(el):
    a = el.attrib
    if "pos" in a:
        v = _floats(a["pos"]); v[1] = -v[1]; a["pos"] = _fmt(v)
    if "quat" in a:
        w, x, y, z = _floats(a["quat"]); a["quat"] = _fmt([w, -x, y, -z])
    if "fromto" in a:
        v = _floats(a["fromto"]); v[1] = -v[1]; v[4] = -v[4]; a["fromto"] = _fmt(v)
    if "axis" in a:
        v = _floats(a["axis"]); v[1] = -v[1]; a["axis"] = _fmt(v)
    if "euler" in a:
        v = _floats(a["euler"]); v[1] = -v[1]; a["euler"] = _fmt(v)
    if "fullinertia" in a:
        v = _floats(a["fullinertia"]); v[3] = -v[3]; v[5] = -v[5]
        a["fullinertia"] = _fmt(v)
    for c in el:
        reflect(c)


def rename(el, pairs):
    for k, val in el.attrib.items():
        nv = val
        for old, new in pairs:
            nv = nv.replace(old, new)
        el.attrib[k] = nv
    for c in el:
        rename(c, pairs)


def find_body(base, name):
    for b in base.findall("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def rewrite_mesh(el):
    """Point every mesh geom in the (reflected) subtree at its mirrored asset."""
    if "mesh" in el.attrib:
        el.set("mesh", el.get("mesh") + "_R")
    for c in el:
        rewrite_mesh(c)


def add_mirrored_assets(root):
    asset = root.find("asset")
    existing = {m.get("name") for m in asset.findall("mesh")}
    for name in LEG_MESHES:
        nm = f"{name}_R"
        if nm in existing:
            continue
        e = ET.SubElement(asset, "mesh")
        e.set("name", nm)
        e.set("file", f"assets/{name}_R.stl")
        print(f"asset + {nm}")


def joint_order(root):
    """Hinge-joint names in qpos order (tree traversal)."""
    names = []

    def walk(el):
        if el.tag in ("body", "worldbody"):
            for c in el:
                if c.tag == "joint":
                    names.append(c.get("name"))
                if c.tag == "body":
                    walk(c)

    walk(root.find("worldbody"))
    return names


def fix_keyframe(root):
    """Rebuild the right-leg joints of the 'home' keyframe as mirrors of the left.

    After mirroring, the right joints use the reflected axis (-y), so the
    mirrored physical pose of a left joint at angle q is the right joint at
    angle -q.  Hence  L3 <- -L1  and  L4 <- -L2 (name-wise).
    """
    order = joint_order(root)
    idx = {n: i for i, n in enumerate(order)}
    kf = root.find("keyframe/key")
    qpos = _floats(kf.get("qpos"))
    mirror_src = {"L3": "L1", "L4": "L2"}
    for right, left in mirror_src.items():
        for joint in ("hip", "knee"):
            ri = idx[f"{joint}_{right}"]
            li = idx[f"{joint}_{left}"]
            qpos[7 + ri] = -qpos[7 + li]
            print(f"keyframe home: {joint}_{right} = -{joint}_{left} = {qpos[7 + ri]:+.5f}")
    kf.set("qpos", _fmt(qpos))


def main():
    write = "--write" in sys.argv
    tree = ET.parse(XML)
    root = tree.getroot()
    base = find_body(root.find("worldbody"), "base")

    for ref_name, tgt_name, pairs in JOBS:
        ref = find_body(base, ref_name)
        mirrored = copy.deepcopy(ref)
        reflect(mirrored)
        rename(mirrored, pairs)
        rewrite_mesh(mirrored)

        tgt = find_body(base, tgt_name)
        idx = list(base).index(tgt)
        base.remove(tgt)
        base.insert(idx, mirrored)
        print(f"{tgt_name} <- mirror({ref_name})")

    add_mirrored_assets(root)
    fix_keyframe(root)

    ET.indent(tree, space="  ")
    out = ET.tostring(root, encoding="unicode")
    out = '<?xml version="1.0" encoding="utf-8"?>\n' + out + "\n"

    if write:
        with open(XML, "w") as f:
            f.write(out)
        print(f"written: {XML}")
    else:
        print("\n--- dry run (pass --write to apply) ---")
        print(out)


if __name__ == "__main__":
    main()
