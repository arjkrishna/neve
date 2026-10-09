#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\pilot_build\route_diag_v6.py (= /sp/v6/pilot_build/route_diag_v6.py in the build containers)
"""v6_pilot: which VA takes which RCCA real wall IN v6 -- route13's diag13.one (imported unchanged) pointed at the v6
tree for both of its inputs: "v4" = /pw/r13v6/<name> (the v6 tree's centrelines + provenance, and the v6 pin report
as its mesh_v4.json = v4's margin and v4's suppressed stretches shifted onto the v6 sections), "v5" = /pw/tree.
From the saved voxel table (npz, sections lower_R / siphon_R of the v6 carotid side): REFUSED_BY_VA = RCCA real
voxels every non-VA term owns (A >= m, not in a pinned stretch) but the v6 VAs refuse; split by VA, RCCA arclength.
Stated diffs (wrapper only): diag13's A4 / A5 / OUT, bake_v5b.REPORT_V5 = mesh_v6.json, is_va = VA branches only
(so the re-posed siphon is a carotid section here).
    python3 route_diag_v6.py <name>     (neve-build-meshlab) -> /pw/r13v6/diag/<name>.{json,rcca_v4.npz}, refused.json
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/pilot_build")
import bake_v6p as BV  # noqa: E402  (branch fallback; REPORT_V5 = mesh_v6.json)

V, B = BV.V, BV.B
V.is_va = lambda sec: sec.branch in ("RVA", "LVA", "(17)", "(18)", "(19)")
sys.path.insert(0, "/sp/v5_va/route13")
import diag13 as D  # noqa: E402

name = sys.argv[1]
R = "/pw/r13v6"
dst = os.path.join(R, name)
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
shutil.copytree(os.path.join("/pw/tree", name, "Centrelines_comb"), os.path.join(dst, "Centrelines_comb"))
shutil.copy2(os.path.join("/pw/tree", name, "provenance.json"), os.path.join(dst, "provenance.json"))
shutil.copy2(os.path.join("/pw/v4pin", name, B.REPORT), os.path.join(dst, B.REPORT))
D.A4, D.A5, D.OUT = R, "/pw/tree", os.path.join(R, "diag")
os.makedirs(D.OUT, exist_ok=True)
D.main() if False else None
res = D.one(name)
z = np.load(os.path.join(D.OUT, name + ".rcca_v4.npz"))
m = float(z["margin"][0])
pin = json.load(open(os.path.join(dst, B.REPORT)))
sup = pin.get("suppressed_stretches_mm") or {}
out = {"name": name, "margin": m}
for sec in ("lower_R", "siphon_R"):
    if sec + "__coords" not in z.files:
        continue
    A = z[sec + "__A"].astype(float)
    u = z[sec + "__use4"].astype(bool)
    s = z[sec + "__s_star"].astype(float)
    dep = z[sec + "__depth"].astype(float)
    sp = np.zeros(len(s), bool)
    for lo, hi in sup.get(sec, []):
        sp |= (s >= lo) & (s <= hi)
    ref = (A >= m) & ~sp & ~u
    rr = {"owned_voxels": int(u.sum()), "refused_by_va": int(ref.sum()), "refused_relief_gt_0.3": int((ref & (dep > 0.3)).sum()),
          "refused_share": round(float(ref.sum()) / max(int(u.sum() + ref.sum()), 1), 5)}
    if ref.any():
        rr["rcca_s_mm"] = [round(float(np.percentile(s[ref], q)), 1) for q in (0, 50, 100)]
    out[sec] = rr
out["diag13_sections"] = {k: {kk: v.get(kk) for kk in ("lost_v4_wall_voxels", "competitors_v5") if kk in v}
                          for k, v in (res or {}).get("sections", {}).items()} if isinstance(res, dict) else None
json.dump(out, open(os.path.join(D.OUT, name + ".refused.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:3000])
