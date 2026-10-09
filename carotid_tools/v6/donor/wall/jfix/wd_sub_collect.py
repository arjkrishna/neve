#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\wd_sub_collect.py (= /sp/v6/wall/jfix/wd_sub_collect.py in the build containers)
"""v6_wall (e) D5 for the donor subclavian stretches: aggregate wd_proxy.py 'subclavian' outputs (/v6w/wd/subclavian/
<unit>_<side>.json: the stretch's own stage-A surface unioned with its VMTK-line tube exactly as the bake does,
share of wall area > 0.3 mm off the tube under exact ownership, undecimated, 1 mm bins of the prepped line).

Per side: whole; and 'core' = without the first / last 8 mm (the capsule tapers to the tube there) and without
+-6 mm about the VA ostium (the VA branch's own stub / lid counts as relief there; it is the junction, not wall).
Ostium = the v6_va request's ostium_s_mm point on its centreline, projected to the nearest point of the prepped line.
Also the polyline lumen minimum to that union over the core bins.
    python3 wd_sub_collect.py  -> /v6w/wd/wd_sub_collect.json   (neve-build-meshlab: topbrain_tools prep_siphon)"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import prep_siphon  # noqa: E402


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


rows = {}
for f in sorted(glob.glob("/v6w/wd/subclavian/*.json")):
    d = json.load(open(f))
    u, sd = d["unit"], d["side"]
    v = "rsub" if sd == "R" else "lsub"
    vj = json.load(open("/v6w/subclavian/%s/%s_stageA_vmtk.json" % (u, v)))
    p, r = prep_siphon(np.asarray(vj["points"], float), np.asarray(vj["radii"], float), 1.0)
    s = arclen(p)
    req = [e for e in json.load(open("/v6w/subclavian/req/%s.json" % u)) if e["side"] == sd][0]
    C = np.asarray(req["centerline_ras_mm"], float)
    sc = arclen(C)
    O = np.array([np.interp(float(req["ostium_s_mm"]), sc, C[:, k]) for k in range(3)])
    jo = int(np.argmin(np.linalg.norm(p - O, axis=1)))
    s_o, d_o = float(s[jo]), float(np.linalg.norm(p[jo] - O))
    ar = np.asarray(d["bins_1mm"]["area"], float)
    a3 = np.asarray(d["bins_1mm"]["a03"], float)
    lm = d["bins_1mm"]["lumen_min_union"]
    nb = len(ar)
    b = np.arange(nb) + 0.5
    core = (b >= 8.0) & (b <= s[-1] - 8.0) & (np.abs(b - s_o) > 6.0)
    lmc = [lm[i] for i in range(nb) if core[i] and lm[i] is not None]
    rows["%s:%s" % (u, sd)] = {
        "length_mm": round(float(s[-1]), 1), "ostium_s_on_line_mm": round(s_o, 1), "ostium_offset_mm": round(d_o, 2),
        "misr_median": round(float(np.median(r)), 2), "whole_0.3": round(float(d["frac_0.3_whole"]), 4),
        "core_0.3": round(float(a3[core].sum() / max(ar[core].sum(), 1e-9)), 4) if core.any() else None,
        "core_mm": int(core.sum()), "lumen_min_core": round(float(min(lmc)), 3) if lmc else None}
    print(u, sd, rows["%s:%s" % (u, sd)], flush=True)


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    return [round(float(v), 4) for v in np.percentile(x, [10, 50, 90])] + [round(float(x.min()), 4), round(float(x.max()), 4)]


summ = {"sides": len(rows), "core_0.3_p10_med_p90_min_max": q([r["core_0.3"] for r in rows.values()]),
        "whole_0.3_p10_med_p90_min_max": q([r["whole_0.3"] for r in rows.values()]),
        "lumen_min_core_p10_med_p90_min_max": q([r["lumen_min_core"] for r in rows.values()]),
        "misr_median_p10_med_p90": q([r["misr_median"] for r in rows.values()])[:3]}
json.dump({"summary": summ, "per_side": rows}, open("/v6w/wd/wd_sub_collect.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
