#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\va_lumen.py (= /sp/v5_va/checks/va_lumen.py in the build containers)
"""v5_checks (extra to item 1): the vertebral lumens in the shipped OBJ, v5 against v4, with v4's route-check
measures (topo_audit.winding / surface_dist: winding-number inside test, exact point-triangle distance), per VA:
body points (3 mm end caps excluded) outside, lumen_min and where, share of points whose wall distance is below
0.65 mm (0.35 mm wire radius + 0.3 mm contact distance: v4's navigable criterion), the same on the donor stretch.
Not a v4 gate (v4 gates the RCCA / LCCA routes only); reported because the SOFA test drives the wire into the RVA.
    python3 va_lumen.py [workers=2]   (neve-build-meshlab, read-only)  -> /sp/v5_va/checks/va_lumen.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/sp/v4_build/audit")
import topo_audit as TA  # noqa: E402

A5, V4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"
NEED = 0.65


def lumen(V, F, p, s, lo=None, hi=None):
    L = s[-1]
    m = (s > 3.0) & (s < L - 3.0)
    if lo is not None:
        m &= (s >= lo) & (s <= hi)
    if not m.any():
        return None
    P, S = p[m], s[m]
    W = TA.winding(P, V, F)
    ins = np.abs(W) > 0.5
    d = TA.surface_dist(P, V, F)
    di = np.where(ins, d, 0.0)
    k = int(np.argmin(np.where(ins, d, np.inf))) if ins.any() else 0
    return {"pts": int(len(P)), "outside": int((~ins).sum()), "lumen_min_mm": round(float(d[ins].min()), 3) if ins.any() else None,
            "lumen_min_at_mm": round(float(S[k]), 1), "lumen_p05_mm": round(float(np.percentile(d[ins], 5)), 3) if ins.any() else None,
            "frac_below_0.65": round(float((di < NEED).mean()), 4), "n_below_0.65": int((di < NEED).sum()),
            "navigable_0.65": bool(ins.all() and d.min() >= NEED)}


def one(name):
    out = {"name": name}
    for tag5, root in (("v5", os.path.join(A5, name)), ("v4", os.path.join(V4, name))):
        br = TA.load_branches(os.path.join(root, "Centrelines_comb"))
        V, F, _ = TA.read_obj(os.path.join(root, "vessel_architecture_collision.obj"))
        prov = json.load(open(os.path.join(root, "provenance.json")))
        for tag in ("RVA", "LVA"):
            p = [x for x in br if x[0] == tag][0][1]
            s = TA.arclen(p)
            r = {"whole": lumen(V, F, p, s)}
            if tag5 == "v5":
                xf = prov["xform"].get("vaC_R" if tag == "RVA" else "vaC_L")
                if xf:
                    r["donor_stretch"] = lumen(V, F, p, s, float(xf["lo_mm"]), float(xf["hi_mm"]))
            out["%s_%s" % (tag5, tag)] = r
    return out


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    names = sorted(n for n in os.listdir(A5) if n.startswith("case_"))
    with Pool(nw) as pool:
        res = pool.map(one, names, chunksize=4)
    json.dump(res, open("/sp/v5_va/checks/va_lumen.json", "w"), indent=1)
    for t in ("v5_RVA", "v5_LVA", "v4_RVA", "v4_LVA"):
        w = [r[t]["whole"] for r in res]
        print(t, "outside", sum(x["outside"] for x in w), "not navigable(0.65)", sum(not x["navigable_0.65"] for x in w),
              "lumen_min p0/p50", np.percentile([x["lumen_min_mm"] for x in w], [0, 50]).round(3).tolist())
