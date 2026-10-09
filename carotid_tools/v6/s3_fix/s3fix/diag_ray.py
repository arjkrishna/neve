#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_ray.py (= /sp/v6/s3fix/diag_ray.py in the build containers)
"""s3_fix diagnosis 3: values along rays from the RCCA centreline through a point (fragment centroid):
f_tube (max over branches of r - d, as sdf_mesher), f_real (-implicit distance of the section surface as the bake
loads it), f_cap (the section capsule as sdf_union), f_sec = min(f_real, f_cap), and the ownership margin terms.
    python3 diag_ray.py <name> <section> <x> <y> <z> [--tree DIR] [--fan]
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402

B = BV.B
from scipy.spatial import cKDTree  # noqa: E402
from sdf_union import _capsule_radii  # noqa: E402
from sdf_union_v4 import Samples, ownership_margin  # noqa: E402


def tubeval(X, P, R, step=0.25):
    s = B.arclen(P)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    Pd = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)
    Rd = np.interp(q, s, R)
    d, nn = cKDTree(Pd).query(X, k=min(8, len(Pd)))
    return (Rd[nn] - d).max(1)


def main():
    name, sname = sys.argv[1], sys.argv[2]
    c = np.array([float(v) for v in sys.argv[3:6]])
    tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else "/repo/carotid_data/anatomies_v6"
    root = os.path.join(tree, name)
    from eve_bench.dualdevicenav import load_branches
    import pyvista as pv
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs, _ = B.sections_for(root, br)
    sec = [s for s in secs if s.name == sname][0]
    rb = B.branch(br, sec.branch)
    P, R = np.asarray(rb.coordinates, float), np.asarray(rb.radii, float)
    S = B.arclen(P)
    d, i = cKDTree(P).query(c)
    print("nearest centreline point s %.2f r %.2f, centroid at %.2f mm" % (S[i], R[i], d))
    cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
    dirs = [c - P[i]]
    if "--fan" in sys.argv:
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        u = dirs[0] - (dirs[0] @ t) * t
        u /= np.linalg.norm(u)
        w = np.cross(t, u)
        dirs = [np.cos(a) * u + np.sin(a) * w for a in np.radians([-40, -20, 0, 20, 40])]
    for D in dirs:
        D = D / np.linalg.norm(D)
        X = P[i] + np.arange(0, 6.01, 0.15)[:, None] * D
        ft = np.max([tubeval(X, np.asarray(b.coordinates, float), np.asarray(b.radii, float)) for b in br], axis=0)
        fr = -np.asarray(pv.PolyData(X).compute_implicit_distance(sec.surface)["implicit_distance"])
        fc = tubeval(X, sec.points, cap_r)
        m = ownership_margin(X, sec, samp)
        print(" r_mm   f_tube  f_real  f_cap   f_sec   margin   union~")
        for k in range(len(X)):
            fs = min(fr[k], fc[k])
            un = max(ft[k], fs) if (fs > ft[k] and m[k] >= 0.5) else ft[k]
            print("%5.2f %7.2f %7.2f %7.2f %7.2f %8.2f %7.2f" % (k * 0.15, ft[k], fr[k], fc[k], fs, m[k], un))


if __name__ == "__main__":
    main()
