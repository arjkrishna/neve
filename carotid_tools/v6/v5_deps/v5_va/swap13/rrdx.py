#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\rrdx.py (= /sp/v5_va/swap13/rrdx.py in the build containers)
"""v5_swap13: right_route_detail v5 vs v4 under EXACT ownership (k24 beside it) for ONE anatomy, on the undecimated
unions -- the final 223 (after the route13 swap).

Code reused unchanged:
  * vvw/measure.py (verify_v5_wall's own implementation of right_route_detail's definition): own markups reader,
    0.25 mm resampling, Own(samp, X, A, 'k24' | 'exact') = argmax_j (r_j - |x - c_j|) over the 24 nearest samples
    (v4's coded cut-off) or over ALL samples; rrd() = RCCA-owned wall past the host cut, frac of area > 0.3 / 0.5 mm,
    fork (s < 130) / siphon (s >= 130);
  * vvw/rrd_v4.rebuild_v4 (= chk_one.rebuild_v4 = ref_v4_union): v4's accepted union rebuilt from mesh_v4.json with
    the repo baker (checked here: its k24 reading equals mesh_v4.json's record at 4 dp).
New here (only bookkeeping): ratios v5/v4 per mode; CELL-FOR-CELL identity of the RCCA route cells everywhere and
beyond 4 / 6 / 8 mm of every v4 / v5 VA sample (as rrd_v4.py / chk_one's "identical beyond D"), two ways: strict
(centroid rounded 1e-4 mm, depth 1e-5 mm, area 1e-6 mm2) and tolerant (version 2; verify_v5_route13's match:
centroid and depth within 1e-3 mm, area within 1e-3 relative); counts of differing cells.
    python3 rrdx.py <name>     (neve-build-meshlab; /repo read-only) -> /sp/v5_va/swap13/rrdx/<name>.json (marker)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va/vvw")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import measure as M  # noqa: E402
import rrd_v4 as RV  # noqa: E402

OUT = "/sp/v5_va/swap13/rrdx"


def keys(C, dep, A, m):
    k = np.c_[np.round(C[m], 4), np.round(dep[m], 5), np.round(A[m], 6)]
    return set(map(tuple, k.tolist()))


TOL_MM = 1e-3      # tolerant match (verify_v5_route13's): centroid within 1e-3 mm, depth within 1e-3 mm, area rel 1e-3


def tol_only(a, b):
    """cells of a with no tolerant match in b"""
    if not len(a[0]):
        return 0
    if not len(b[0]):
        return int(len(a[0]))
    d, i = cKDTree(b[0]).query(a[0], distance_upper_bound=TOL_MM)
    ok = np.isfinite(d)
    j = i[ok]
    good = np.zeros(len(d), bool)
    good[ok] = (np.abs(a[1][ok] - b[1][j]) <= TOL_MM) & (np.abs(a[2][ok] - b[2][j]) <= 1e-6 + 1e-3 * b[2][j])
    return int((~good).sum())


def segstats(A, dep, m):
    a, x = A[m], dep[m]
    tot = float(a.sum()) or 1.0
    return {"area_mm2": round(float(a.sum()), 4), "cells": int(m.sum()),
            "frac_0.3mm": float(a[x > 0.3].sum()) / tot, "frac_0.5mm": float(a[x > 0.5].sum()) / tot}


def route_sets(O, samp, cut, C, A, vt):
    rc = samp.idx("RCCA")
    own = (O.b == rc) & (O.s >= cut)
    segm = {"fork": own & (O.s < 130.0), "siphon": own & (O.s >= 130.0)}
    dist = np.full(len(C), np.inf)
    dist[own] = vt.query(C[own])[0]
    sets, st, arr = {}, {}, {}
    for seg, m in segm.items():
        sets[("all", seg)] = keys(C, O.dep, A, m)
        arr[("all", seg)] = (C[m], O.dep[m], A[m])
        st[seg] = segstats(A, O.dep, m)
        for D in (4, 6, 8):
            mm = m & (dist > D)
            sets[(str(D), seg)] = keys(C, O.dep, A, mm)
            arr[(str(D), seg)] = (C[mm], O.dep[mm], A[mm])
            st["%s_beyond_%d" % (seg, D)] = segstats(A, O.dep, mm)
    return sets, st, arr


def main():
    name = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst) and "--force" not in sys.argv:
        print("done already", dst)
        return 0
    t0 = time.time()
    prov5 = json.load(open(os.path.join(M.A5, name, "provenance.json"), encoding="utf-8"))
    prov4 = json.load(open(os.path.join(M.V4, name, "provenance.json"), encoding="utf-8"))
    rep5 = json.load(open(os.path.join(M.A5, name, "mesh_v5.json"), encoding="utf-8"))
    rep4 = json.load(open(os.path.join(M.V4, name, "mesh_v4.json"), encoding="utf-8"))
    cut5, cut4 = float(prov5.get("host_cut_mm", 0)), float(prov4.get("host_cut_mm", 0))
    s5 = M.Samp(M.read_branches(os.path.join(M.A5, name, "Centrelines_comb")))
    s4 = M.Samp(M.read_branches(os.path.join(M.V4, name, "Centrelines_comb")))
    t = time.time()
    iso4 = RV.rebuild_v4(name)
    V4_ = np.asarray(iso4.points, float)
    F4_ = np.asarray(iso4.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    del iso4
    t_rebuild = time.time() - t
    V5, F5 = M.read_vtp(os.path.join(M.ISO, name + ".vtp"))
    C5, A5_ = M.tri_geom(V5, F5)
    C4, A4_ = M.tri_geom(V4_, F4_)
    del V5, F5, V4_, F4_
    va = np.vstack([s.p[(s.b == s.idx("RVA")) | (s.b == s.idx("LVA"))] for s in (s4, s5)])
    vt = cKDTree(va)
    rec = {"rrdx_version": 2, "name": name, "cut5": cut5, "cut4": cut4, "tris_v5": int(len(C5)), "tris_v4": int(len(C4)),
           "v4_rebuild_seconds": round(t_rebuild, 1), "mesh_v5_record": rep5["right_route_detail"],
           "mesh_v4_record": rep4["right_route_detail"]}
    for mode in ("k24", "exact"):
        t = time.time()
        O5 = M.Own(s5, C5, A5_, mode)
        O4 = M.Own(s4, C4, A4_, mode)
        set5, st5, arr5 = route_sets(O5, s5, cut5, C5, A5_, vt)
        set4, st4, arr4 = route_sets(O4, s4, cut4, C4, A4_, vt)
        r5, r4 = M.rrd(O5, s5, cut5), M.rrd(O4, s4, cut4)
        diff, dtol = {}, {}
        for k in set5:
            a, b = set5[k], set4[k]
            diff["%s_%s" % k] = {"only_v5": len(a - b), "only_v4": len(b - a), "common": len(a & b)}
            dtol["%s_%s" % k] = {"only_v5": tol_only(arr5[k], arr4[k]), "only_v4": tol_only(arr4[k], arr5[k]),
                                 "cells_v5": int(len(arr5[k][0])), "cells_v4": int(len(arr4[k][0]))}
        ident = {D: all(diff["%s_%s" % (D, seg)]["only_v5"] == 0 and diff["%s_%s" % (D, seg)]["only_v4"] == 0
                        for seg in ("fork", "siphon")) for D in ("all", "4", "6", "8")}
        cls = 0 if ident["all"] else next((int(D) for D in ("4", "6", "8") if ident[D]), None)
        ident_t = {D: all(dtol["%s_%s" % (D, seg)]["only_v5"] == 0 and dtol["%s_%s" % (D, seg)]["only_v4"] == 0
                          for seg in ("fork", "siphon")) for D in ("all", "4", "6", "8")}
        cls_t = 0 if ident_t["all"] else next((int(D) for D in ("4", "6", "8") if ident_t[D]), None)
        ratio = {g: {tt: (r5[g][tt] / r4[g][tt] if r4[g][tt] else None) for tt in ("frac_0.3mm", "frac_0.5mm")} for g in r5}
        rec[mode] = {"rrd_v5": r5, "rrd_v4": r4, "rrd_v5_r4": M.r4(r5), "rrd_v4_r4": M.r4(r4),
                     "ratio": ratio, "equal_4dp": M.r4(r5) == M.r4(r4), "stats_v5": st5, "stats_v4": st4,
                     "cell_diff": diff, "identical_cells": ident, "identical_beyond_mm": cls,
                     "cell_diff_tol": dtol, "identical_cells_tol": ident_t, "identical_beyond_mm_tol": cls_t,
                     "seconds": round(time.time() - t, 1)}
        del O5, O4, set5, set4, arr5, arr4
    rec["k24_v4_rebuild_equals_mesh_v4"] = rec["k24"]["rrd_v4_r4"] == rep4["right_route_detail"]
    rec["k24_v5_equals_mesh_v5"] = rec["k24"]["rrd_v5_r4"] == rep5["right_route_detail"]
    rec["seconds"] = round(time.time() - t0, 1)
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            rec["peak_rss_gb"] = round(int(line.split()[1]) / 1048576.0, 3)
    with open(dst + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, default=float)
    os.replace(dst + ".tmp", dst)
    e, k = rec["exact"], rec["k24"]
    print("DONE %s %.0f s peak %s | k24 rebuild==v4rec %s v5==rec %s | exact fork %.4f siphon %.4f class %s tol %s | k24 fork %.4f "
          "siphon %.4f class %s tol %s" % (name, rec["seconds"], rec.get("peak_rss_gb"), rec["k24_v4_rebuild_equals_mesh_v4"],
                                           rec["k24_v5_equals_mesh_v5"], e["ratio"]["fork"]["frac_0.3mm"] or -1,
                                           e["ratio"]["siphon"]["frac_0.3mm"] or -1, e["identical_beyond_mm"],
                                           e["identical_beyond_mm_tol"], k["ratio"]["fork"]["frac_0.3mm"] or -1,
                                           k["ratio"]["siphon"]["frac_0.3mm"] or -1, k["identical_beyond_mm"],
                                           k["identical_beyond_mm_tol"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
