#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_neck.py (= /sp/v6/s3fix/diag_neck.py in the build containers)
"""s3_fix diagnosis 4: where does a dropped sliver join the main lumen in the SCAN?
On a fine grid (default 0.15 mm) in a box around a point (the fragment centroid, +- half), label the connected components
of (a) the scanned lumen {f_real > 0} of the section as the bake loads it, (b) the same clipped by the section capsule
{min(f_real, f_cap) > 0} (what the bake can see), (c) (b) united with every branch tube {f_tube > 0}. Reports whether
the component holding the fragment centroid's nearest lumen voxel is the one holding the centreline, and, for (a), the
distance from the centreline / capsule value along the shortest connecting path region (the voxels of the sliver's
component nearest to the main lumen).
    python3 diag_neck.py <name> <section> <x> <y> <z> [half_mm=8] [h=0.15] [--tree DIR]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402

B = BV.B
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sdf_union import _capsule_radii  # noqa: E402


def tubeval(X, P, R, step=0.25, k=8):
    s = B.arclen(P)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    Pd = np.stack([np.interp(q, s, P[:, i]) for i in range(3)], 1)
    Rd = np.interp(q, s, R)
    d, nn = cKDTree(Pd).query(X, k=min(k, len(Pd)))
    return (Rd[nn] - d).max(1)


def main():
    name, sname = sys.argv[1], sys.argv[2]
    c = np.array([float(v) for v in sys.argv[3:6]])
    half = float(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else 8.0
    h = float(sys.argv[7]) if len(sys.argv) > 7 and not sys.argv[7].startswith("--") else 0.15
    tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else "/repo/carotid_data/anatomies_v6"
    root = os.path.join(tree, name)
    from eve_bench.dualdevicenav import load_branches
    import pyvista as pv
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    secs, _ = B.sections_for(root, br)
    sec = [s for s in secs if s.name == sname][0]
    ax = [np.arange(c[i] - half, c[i] + half + 1e-9, h) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    shp = G.shape[:3]
    X = G.reshape(-1, 3)
    fr = -np.asarray(pv.PolyData(X).compute_implicit_distance(sec.surface)["implicit_distance"])
    cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
    fc = tubeval(X, sec.points, cap_r)
    ft = np.max([tubeval(X, np.asarray(b.coordinates, float), np.asarray(b.radii, float)) for b in br], axis=0)
    rb = B.branch(br, sec.branch)
    P, R = np.asarray(rb.coordinates, float), np.asarray(rb.radii, float)
    S = B.arclen(P)
    dcl, icl = cKDTree(P).query(X)
    # the centreline voxel(s) and the sliver seed (inside lumen, nearest the fragment centroid but > tube)
    for tag, mask in (("scan", fr > 0), ("scan_in_capsule", np.minimum(fr, fc) > 0),
                      ("bake_view(scan_in_capsule | tubes)", (np.minimum(fr, fc) > 0) | (ft > 0))):
        lab, n = ndimage.label(mask.reshape(shp))
        lab = lab.ravel()
        main_lab = np.bincount(lab[(dcl < 0.5) & (lab > 0)]).argmax() if ((dcl < 0.5) & (lab > 0)).any() else 0
        sl = (lab > 0) & (ft < -0.3) & (fr > 0)
        # sliver seed: lumen voxels outside the tubes nearest c
        if sl.any():
            j = np.nonzero(sl)[0][np.argmin(np.linalg.norm(X[sl] - c, axis=1))]
            seed_lab = lab[j]
        else:
            seed_lab = -1
        same = bool(seed_lab == main_lab and seed_lab > 0)
        out = {"components": int(n), "main": int(main_lab), "seed": int(seed_lab), "sliver_joined_to_main": same}
        if seed_lab > 0 and not same:
            # the sliver component's voxels closest to the main component
            A = X[lab == seed_lab]
            Bm = X[lab == main_lab] if main_lab > 0 else None
            if Bm is not None and len(Bm):
                d, k = cKDTree(Bm).query(A)
                i = np.argmin(d)
                out["gap_min_mm"] = round(float(d[i]), 3)
                out["gap_at"] = np.round(A[i], 2).tolist()
            out["sliver_voxels"] = int((lab == seed_lab).sum())
            out["sliver_s"] = [round(float(S[icl[lab == seed_lab]].min()), 1), round(float(S[icl[lab == seed_lab]].max()), 1)]
        if seed_lab > 0 and same and tag == "scan":
            # where does the sliver join? voxels of the joined component outside every tube: the narrowest bit is the
            # neck; report the capsule value there (neck outside the capsule if f_cap < 0)
            m = (lab == seed_lab) & (ft < 0)
            out["outside_tube_voxels"] = int(m.sum())
            out["capsule_value_on_them"] = [round(float(np.percentile(fc[m], q)), 2) for q in (0, 5, 50)] if m.any() else None
            out["outside_capsule_voxels"] = int((m & (fc < 0)).sum())
        print(tag, out, flush=True)


if __name__ == "__main__":
    main()
