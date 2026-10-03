#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\cmp_iso.py (= /sp/v5_va/route13/cmp_iso.py in the build containers)
"""v5_route13: compare the RCCA route wall (right_route_detail's own attribution and depth) of two undecimated unions
of one anatomy, per 5 mm of RCCA arclength, and say which branch the cells near the RCCA are attributed to.
    python3 cmp_iso.py <name> <isoA.vtp> <rootA> <isoB.vtp> <rootB>        (neve-build-meshlab)"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
from sdf_union_v4 import Samples  # noqa: E402


def cells(iso, samp, cut):
    rc = samp.index_of("RCCA")
    cen = np.asarray(iso.cell_centers().points)
    A = np.asarray(iso.compute_cell_sizes(length=False, volume=False)["Area"])
    dd, ii = cKDTree(samp.p).query(cen, k=24)
    best = ii[np.arange(len(ii)), np.argmax(samp.r[ii] - dd, axis=1)]
    own = (samp.b[best] == rc) & (samp.s[best] >= cut)
    dep = np.abs(np.linalg.norm(cen - samp.p[best], axis=1) - samp.r[best])
    # cells within 10 mm of the RCCA centreline (cut on) attributed to another branch
    rp = samp.p[(samp.b == rc) & (samp.s >= cut)]
    nearR = cKDTree(rp).query(cen)[0] < 10.0
    return cen, A, best, own, dep, nearR


def main():
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    name, ia, ra, ib, rb = sys.argv[1:6]
    out = {"name": name, "A": ia, "B": ib}
    C = {}
    for tag, iso_p, root in (("A", ia, ra), ("B", ib, rb)):
        iso = pv.read(iso_p)
        samp = Samples(load_branches(os.path.join(root, "Centrelines_comb")))
        cut = float(json.load(open(os.path.join(root, "provenance.json")))["host_cut_mm"])
        C[tag] = cells(iso, samp, cut) + (samp,)
        out[tag + "_detail"] = B.right_route_detail(iso, samp, root)
        out[tag + "_cells"] = int(iso.n_cells)
        cen, A, best, own, dep, nearR, samp = C[tag]
        oth = nearR & ~own
        names = np.array([n.split("/")[-1][-12:] for n in samp.names])
        u, c = np.unique(names[samp.b[best[oth]]], return_counts=True)
        out[tag + "_near_rcca_attributed_elsewhere"] = {str(k): int(v) for k, v in zip(u, c)}
    rcs = C["A"][6].s[C["A"][6].b == C["A"][6].index_of("RCCA")]
    bins = []
    for lo in np.arange(0, rcs.max() + 5, 5.0):
        row = {"s": float(lo)}
        diff = False
        for tag in ("A", "B"):
            cen, A, best, own, dep, nearR, samp = C[tag]
            m = own & (samp.s[best] >= lo) & (samp.s[best] < lo + 5)
            row[tag] = [round(float(A[m].sum()), 2), round(float(A[m & (dep > 0.3)].sum()), 2)]
        if abs(row["A"][0] - row["B"][0]) > 0.05 or abs(row["A"][1] - row["B"][1]) > 0.05:
            bins.append(row)
    out["bins_differing"] = bins
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
