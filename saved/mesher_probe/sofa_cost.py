#!/usr/bin/env python3
"""How much does SOFA care about the vessel triangle count?

Same anatomy (topcow_mr_001), same centerlines, five collision meshes from
3.7 k to 387 k triangles. Time the simulator step in two regimes: free
advance (few contacts) and pushing/rotating against the wall (many).

    python3 /opt/eve_training/saved/mesher_probe/sofa_cost.py
"""
import os
import shutil
import sys
import time

import numpy as np
import pyvista as pv

sys.path.insert(0, "/opt/eve_training/eve_bench")

P = "/opt/eve_training/saved/mesher_probe"
SRC = "/opt/eve_training/topbrain_data/anatomies/topcow_mr_001"
ROOT = os.path.join(P, "sofa_cost")
VARIANTS = [("baked_3k7", SRC + "/vessel_architecture_collision.obj"),
            ("remesh_b_16k", P + "/topcow_mr_001_remesh_b.vtp"),
            ("remesh_a_20k", P + "/topcow_mr_001_remesh_a.vtp"),
            ("remesh_u_38k", P + "/topcow_mr_001_remesh_u.vtp"),
            ("sdf_full_387k", P + "/topcow_mr_001_sdf_full.vtp")]


def build_variants():
    out = {}
    for tag, f in VARIANTS:
        d = os.path.join(ROOT, tag)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "Centrelines_comb")):
            shutil.copytree(SRC + "/Centrelines_comb", os.path.join(d, "Centrelines_comb"))
        obj = os.path.join(d, "vessel_architecture_collision.obj")
        m = pv.read(f).extract_surface().triangulate()
        if not os.path.exists(obj):
            pv.save_meshio(obj, m)
        out[tag] = m.n_cells
    return out


def time_steps(iv, action, n):
    t0 = time.time()
    for _ in range(n):
        iv.step(action)
    return (time.time() - t0) / n * 1000.0


def main():
    tris = build_variants()
    from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
    adv = np.array([[12.0, 0.0], [12.0, 0.0]], dtype=np.float32)
    grind = np.array([[3.0, 1.5], [3.0, -1.5]], dtype=np.float32)
    print("%-14s %8s %10s %12s %12s" % ("mesh", "tris", "load_s", "advance_ms", "contact_ms"))
    for tag, _ in VARIANTS:
        t0 = time.time()
        iv = DualDeviceNavTopBrain(anatomy_dir=ROOT, seed=0, episodes_between_change=1, only=[tag])
        iv.reset(episode_number=0, seed=3)
        load = time.time() - t0
        rows = []
        for seed in (3, 4):
            iv.reset(episode_number=0, seed=seed)
            time_steps(iv, adv, 5)                      # warm up
            a = time_steps(iv, adv, 40)                 # ~190 mm of free advance
            c = time_steps(iv, grind, 60)               # then push and twist at the wall
            rows.append((a, c))
        a = np.mean([r[0] for r in rows]); c = np.mean([r[1] for r in rows])
        print("%-14s %8d %10.1f %12.1f %12.1f" % (tag, tris[tag], load, a, c), flush=True)
        iv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
