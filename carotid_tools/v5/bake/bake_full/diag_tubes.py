#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\diag_tubes.py (= /sp/v5_va/bake_full/diag_tubes.py in the build containers)
"""v5_bake_full diagnosis (read-only): a tube-only mesh with a handle ("centerline-level fusion"). Which VA makes
it, and where? (1) genus of the tube-only union with all branches, without the RVA, without the LVA; (2) for each VA
branch, the closest tube-to-tube gap to every other branch (|p_i - q_j| - r_i - r_j, min over samples) along the VA,
from its start, outside the first JUNC_MM (its own junction with the parent); (3) the assignment's clearance record.
    python3 diag_tubes.py <name> [anatomies_root]        (neve-build-meshlab)
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
from sdf_mesher import iso_surface, tube_field  # noqa: E402
from sdf_union_v4 import genus  # noqa: E402

JUNC_MM = 12.0


def main():
    from eve_bench.dualdevicenav import load_branches
    name = sys.argv[1]
    A = sys.argv[2] if len(sys.argv) > 2 else "/repo/carotid_data/anatomies_v5"
    root = os.path.join(A, name)
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    names = [str(b.name).split("/")[-1] for b in br]
    out = {"name": name, "root": A}
    for tag, drop in (("all", None), ("without RVA", "RVA"), ("without LVA", "LVA")):
        sub = [b for b in br if not (drop and str(b.name).endswith(drop + ".mrk"))]
        f = tube_field(sub, spacing=B.SPACING_MM)
        iso, st, _ = B.keep_largest(iso_surface(f))
        g = genus(iso)
        out[tag] = {"genus": [c["genus"] for c in g], "comps_before_keep": st["comps"]}
        print(tag, out[tag], flush=True)
        del f, iso
    for va in ("RVA", "LVA"):
        b = B.branch(br, va)
        P, R = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        S = B.arclen(P)
        rows = []
        for ob in br:
            if ob is b:
                continue
            Q, RQ = np.asarray(ob.coordinates, float), np.asarray(ob.radii, float)
            d, j = cKDTree(Q).query(P, k=min(16, len(Q)))
            gap = (d - RQ[j]).min(axis=1) - R
            m = S > JUNC_MM
            if not m.any():
                continue
            i = int(np.argmin(np.where(m, gap, np.inf)))
            rows.append({"other": str(ob.name).split("/")[-1], "min_gap_mm": round(float(gap[i]), 3),
                         "at_s_mm": round(float(S[i]), 1), "va_r_mm": round(float(R[i]), 2),
                         "n_samples_gap_below_0.45": int((m & (gap < 0.45)).sum())})
        rows.sort(key=lambda r: r["min_gap_mm"])
        out[va + "_gaps"] = rows[:5]
        print(va, json.dumps(rows[:5]), flush=True)
    prov = json.load(open(os.path.join(root, "provenance.json")))
    for side in ("right", "left"):
        g = (prov.get("v5_va", {}).get(side) or {}).get("gate")
        if g:
            out["gate_" + side] = {k: v for k, v in g.items() if "clear" in k or "exit" in k}
            print(side, out["gate_" + side], flush=True)
    os.makedirs("/repo/carotid_data/va_sources/v5_reg/bake_full_diag", exist_ok=True)
    json.dump(out, open("/repo/carotid_data/va_sources/v5_reg/bake_full_diag/%s.tubes.json" % name, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
