#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\diag_lost.py (= /sp/v5_va/swap13/diag_lost.py in the build containers)
"""v5_swap13 diagnostic: for an anatomy whose RCCA right_route_detail under EXACT ownership is below v4, find the v4
RCCA-owned route cells (exact, past the host cut, fork or siphon) with no tolerant match among the v5 RCCA-owned cells
("lost"), and say whether each still exists GEOMETRICALLY in the v5 union (same centroid / area within rrdx's
tolerance) -- then who owns it in v5 -- or not (the wall itself changed there). Reuses vvw/measure.py and
vvw/rrd_v4.rebuild_v4 unchanged (as rrdx.py). Read-only.
    python3 diag_lost.py <name> [fork|siphon]   -> /sp/v5_va/swap13/diag/<name>.json
"""
import collections
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va/vvw")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import measure as M  # noqa: E402
import rrd_v4 as RV  # noqa: E402

TOL = 1e-3
name = sys.argv[1]
seg = sys.argv[2] if len(sys.argv) > 2 else "fork"
os.makedirs("/sp/v5_va/swap13/diag", exist_ok=True)
prov5 = json.load(open(os.path.join(M.A5, name, "provenance.json"), encoding="utf-8"))
prov4 = json.load(open(os.path.join(M.V4, name, "provenance.json"), encoding="utf-8"))
cut5, cut4 = float(prov5.get("host_cut_mm", 0)), float(prov4.get("host_cut_mm", 0))
s5 = M.Samp(M.read_branches(os.path.join(M.A5, name, "Centrelines_comb")))
s4 = M.Samp(M.read_branches(os.path.join(M.V4, name, "Centrelines_comb")))
iso4 = RV.rebuild_v4(name)
V4_ = np.asarray(iso4.points, float)
F4_ = np.asarray(iso4.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
V5, F5 = M.read_vtp(os.path.join(M.ISO, name + ".vtp"))
C5, A5_ = M.tri_geom(V5, F5)
C4, A4_ = M.tri_geom(V4_, F4_)
O5 = M.Own(s5, C5, A5_, "exact")
O4 = M.Own(s4, C4, A4_, "exact")
O4k = M.Own(s4, C4, A4_, "k24")
O5k = M.Own(s5, C5, A5_, "k24")


def segmask(O, s, cut):
    m = (O.b == s.idx("RCCA")) & (O.s >= cut)
    return m & ((O.s < 130.0) if seg == "fork" else (O.s >= 130.0))


m4, m5 = segmask(O4, s4, cut4), segmask(O5, s5, cut5)
i4, i5 = np.nonzero(m4)[0], np.nonzero(m5)[0]


def unmatched(ia, Ca, Da, Aa, ib, Cb, Db, Ab, use_dep=True):
    d, j = cKDTree(Cb[ib]).query(Ca[ia], distance_upper_bound=TOL)
    ok = np.isfinite(d)
    good = np.zeros(len(ia), bool)
    jj = ib[j[ok]]
    g = np.abs(Aa[ia[ok]] - Ab[jj]) <= 1e-6 + 1e-3 * Ab[jj]
    if use_dep:
        g &= np.abs(Da[ia[ok]] - Db[jj]) <= TOL
    good[ok] = g
    return ia[~good]


lost = unmatched(i4, C4, O4.dep, A4_, i5, C5, O5.dep, A5_)
gained = unmatched(i5, C5, O5.dep, A5_, i4, C4, O4.dep, A4_)
# geometric presence of the lost cells anywhere in the v5 union (centroid + area only)
allv5 = np.arange(len(C5))
d, j = cKDTree(C5).query(C4[lost], distance_upper_bound=TOL)
present = np.isfinite(d)
jj = np.where(present, j, 0)
present &= np.abs(A4_[lost] - A5_[jj]) <= 1e-6 + 1e-3 * A5_[jj]
dn, _ = cKDTree(C5).query(C4[lost])
va5 = s5.p[(s5.b == s5.idx("RVA")) | (s5.b == s5.idx("LVA"))]
va4 = s4.p[(s4.b == s4.idx("RVA")) | (s4.b == s4.idx("LVA"))]
d_va5 = cKDTree(va5).query(C4[lost])[0]
d_va4 = cKDTree(va4).query(C4[lost])[0]
deep4 = O4.dep[lost] > 0.3
tag5 = [s5.tags[b] for b in O5.b[jj[present]]]
tag4k = [s4.tags[b] for b in O4k.b[lost]]


def q(x):
    x = np.asarray(x, float)
    return None if not len(x) else [round(float(x.min()), 3), round(float(np.median(x)), 3), round(float(x.max()), 3)]


tot4 = float(A4_[m4].sum())
out = {
    "name": name, "segment": seg, "cut4": cut4, "cut5": cut5,
    "rcca_cells_v4_exact": int(m4.sum()), "rcca_cells_v5_exact": int(m5.sum()),
    "rcca_frac_0.3_v4": float(A4_[m4][O4.dep[m4] > 0.3].sum()) / tot4,
    "rcca_frac_0.3_v5": float(A5_[m5][O5.dep[m5] > 0.3].sum()) / float(A5_[m5].sum()),
    "lost_cells": int(len(lost)), "lost_area_mm2": round(float(A4_[lost].sum()), 3),
    "lost_cells_v4_depth_gt_0.3": int(deep4.sum()), "lost_area_v4_depth_gt_0.3_mm2": round(float(A4_[lost][deep4].sum()), 3),
    "gained_cells": int(len(gained)), "gained_area_mm2": round(float(A5_[gained].sum()), 3),
    "gained_cells_v5_depth_gt_0.3": int((O5.dep[gained] > 0.3).sum()),
    "lost_present_geometrically_in_v5": int(present.sum()),
    "lost_present_area_mm2": round(float(A4_[lost][present].sum()), 3),
    "lost_present_v5_owner_exact": dict(collections.Counter(tag5)),
    "lost_present_depth_v4_rcca_vs_v5_owner": {"v4": q(O4.dep[lost][present]), "v5": q(O5.dep[jj[present]])},
    "lost_absent_cells": int((~present).sum()), "lost_absent_area_mm2": round(float(A4_[lost][~present].sum()), 3),
    "lost_absent_depth_gt_0.3_cells": int((deep4 & ~present).sum()),
    "lost_absent_nearest_v5_cell_mm": q(dn[~present]),
    "lost_v4_owner_k24": dict(collections.Counter(tag4k)),
    "lost_rcca_s_mm_v4": q(O4.s[lost]),
    "lost_dist_to_v5_va_sample_mm": q(d_va5), "lost_dist_to_v4_va_sample_mm": q(d_va4),
    "lost_v4_depth_mm": q(O4.dep[lost]),
}
json.dump(out, open("/sp/v5_va/swap13/diag/%s.json" % name, "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
