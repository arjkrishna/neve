#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\diag_cov.py (= /sp/v6/s2checks/diag_cov.py in the build containers)
"""diag_cov.py 1.2 (s2_checks, 2026-10-07; 1.2 = + v6 branch tubes near the lost points, lost points saved (.npz); 1.1 frozen diag_cov_v1_1.py; 1.1 = + the bake capsule field at the lost points in v6 and v4, the v6 surface there; 1.0 frozen diag_cov_v1_0.py): WHERE is v4's wall that does not reappear in v6 (D22 coverage < 0.99, reported,
not gated)? Diagnosis only; the frozen checks_v6.py 1.5 is imported unchanged and its route_d22() is re-compiled IN MEMORY
with one inserted statement that keeps the coverage cells (v4's RCCA-owned cells of the piece's gated stretch, carried by
the piece's own transform into v6). The D13 baseline is the one recorded in the build's checks4 record.
For every coverage cell farther than 0.3 mm from the v6 union: inside / outside the v6 union (vtk enclosed points: inside =
v6's lumen now reaches beyond v4's wall there, i.e. the wall became interior to a bigger lumen; outside = v6's wall lies
inside v4's, i.e. lumen lost), the owner branch of the nearest v6 union cell (exact ownership), its v6 RCCA arclength and
its distance to the handle windows / seam zone; and the v6 RCCA-owned cells there: are they on the piece's own scanned
surface (the bake's real-surface carry) or not.
    python3 diag_cov.py <name> <piece fork|siphon> <tree> <iso> <out.json>      (eve-training-fixed / neve-build-meshlab)
"""
import inspect
import json
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v6/checks")
import checks_v6 as C  # noqa: E402

name, piece, tree, iso, outp = sys.argv[1:6]
src = inspect.getsource(C.route_d22)
anchor = '                                 "area_gt_0.3_mm2": round(float(a[d > D22_DEV_MM].sum()), 2)}\n'
assert src.count(anchor) == 1, "anchor not unique in route_d22"
inject = anchor + ('                _DIAG[piece] = {"X": T(C4b[mc]), "d": d, "a": a, "s4": O4b.s[mc], "off": off, "wins": wins,\n'
                   '                                "g6": g6, "zone6": zone6, "V6": V6, "F6": F6, "C6": C6, "O6": O6, "s6": s6,\n'
                   '                                "srcm": srcm, "rc6": rc6, "sd4": sd4, "sds": sds, "Ti": Ti, "m_all": m_all,\n'
                   '                                "A6": A6}\n')
C.__dict__["_DIAG"] = {}
exec(compile(src.replace(anchor, inject), "route_d22_diag", "exec"), C.__dict__)
rec4 = json.load(open("/full/checks4/%s.json" % name))
rep6 = json.load(open("%s/%s/mesh_v6.json" % (tree, name)))
d22 = C.part_route_d22(name, tree, iso, rep6, rec4["route"])
D = C._DIAG[piece]
bad = D["d"] > C.D22_DEV_MM
X, a = D["X"][bad], D["a"][bad]
mesh6 = pv.PolyData(D["V6"], np.c_[np.full(len(D["F6"]), 3), D["F6"]].ravel())
ins = np.asarray(pv.PolyData(X).select_enclosed_points(mesh6, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
tC = cKDTree(D["C6"])
_, j = tC.query(X)
own_b = D["O6"].b[j]
tags = D["s6"].tags
own_name = [tags[b] for b in own_b]
s6v = D["O6"].s[j]
# the piece's own scanned surface (bake carry) near those cells: distance of X to the own surface
Vs = np.asarray(D["srcm"].points, float)
Fs = np.asarray(D["srcm"].faces).reshape(-1, 4)[:, 1:].astype(np.int64)
dos = C.SurfDist(Vs, Fs)(X)
s_on6 = D["s4"][bad] + D["off"]


def runs(s, gap=1.0):
    s = np.sort(s)
    out, a0 = [], s[0]
    for x, y in zip(s[:-1], s[1:]):
        if y - x > gap:
            out.append([round(float(a0), 2), round(float(x), 2)])
            a0 = y
    out.append([round(float(a0), 2), round(float(s[-1]), 2)])
    return out


res = {"name": name, "piece": piece, "coverage": d22[piece].get("coverage"), "reference": d22[piece].get("d22_reference"),
       "value": d22[piece].get("value"), "handle_windows_v6": d22[piece].get("handle_windows_v6"),
       "handle_windows_v4_mapped": d22[piece].get("handle_windows_v4_mapped"), "gate_v6": d22[piece].get("gate_v6"),
       "zone_v6": [round(float(x), 2) for x in D["zone6"]],
       "off_cells": int(bad.sum()), "off_area_mm2": round(float(a.sum()), 2),
       "inside_v6_union_area_mm2": round(float(a[ins].sum()), 2), "outside_v6_union_area_mm2": round(float(a[~ins].sum()), 2),
       "dist_to_v6_union_mm": {"median": round(float(np.median(D["d"][bad])), 3), "max": round(float(D["d"][bad].max()), 3)},
       "nearest_v6_owner_area_mm2": {t: round(float(a[np.asarray(own_name) == t].sum()), 2) for t in sorted(set(own_name))},
       "v6_rcca_s_runs_mm": runs(s_on6),
       "dist_to_own_scanned_surface_mm": {"median": round(float(np.median(dos)), 3), "p90": round(float(np.percentile(dos, 90)), 3)},
       "area_within_0.3_of_own_surface_mm2": round(float(a[dos <= 0.3].sum()), 2)}
# where the v4 wall is INSIDE the v6 union: how deep (signed) and which owner
if ins.any():
    res["inside_owner_area_mm2"] = {t: round(float(a[ins & (np.asarray(own_name) == t)].sum()), 2) for t in sorted(set(np.asarray(own_name)[ins]))}
if (~ins).any():
    res["outside_owner_area_mm2"] = {t: round(float(a[~ins & (np.asarray(own_name) == t)].sum()), 2) for t in sorted(set(np.asarray(own_name)[~ins]))}
# 1.1: is the bake's own section field the CAPSULE at the lost points (v6 world, this tree's bake sections) and at the same
# points carried into v4's world (v4's bake sections)? And the v6 surface there: the gated v6 RCCA cells within 3 mm of the
# lost points -- their distance to v4's carried union and to the piece's own scanned surface.
sec = C.D27_SEC[piece]
try:
    f6, c6, clip6 = C.bake_capsule_clip("%s/%s" % (tree, name), sec, X)
    res["v6_bake_field_at_lost_points"] = {"capsule_clip_area_mm2": round(float(a[clip6].sum()), 2),
                                           "f_real_mm": C._st(f6), "capsule_value_mm": C._st(c6),
                                           "lumen_beyond_capsule_mm": C._st((f6 - c6)[clip6]) if clip6.any() else None}
except Exception as e:  # noqa: BLE001
    res["v6_bake_field_error"] = "%s: %s" % (type(e).__name__, e)
try:
    _, _, clip4 = C.bake_capsule_clip("%s/%s" % (C.V4, name), sec, D["Ti"](X))
    res["v4_bake_capsule_clip_area_mm2"] = round(float(a[clip4].sum()), 2)
except Exception as e:  # noqa: BLE001
    res["v4_bake_field_error"] = "%s: %s" % (type(e).__name__, e)
near = D["m_all"].copy()
iv = np.nonzero(near)[0]
dn, _ = cKDTree(X).query(D["C6"][iv])
iv = iv[dn <= 3.0]
if len(iv):
    d4n, don = D["sd4"](D["C6"][iv]), D["sds"](D["C6"][iv])
    an = D["A6"][iv]
    res["v6_gated_cells_within_3mm"] = {"area_mm2": round(float(an.sum()), 2),
                                        "dist_to_v4_carried_mm": C._st(d4n), "dist_to_own_surface_mm": C._st(don),
                                        "area_off_own_gt_0.3_mm2": round(float(an[don > 0.3].sum()), 2)}
    try:
        f6n, c6n, clip6n = C.bake_capsule_clip("%s/%s" % (tree, name), sec, D["C6"][iv])
        res["v6_gated_cells_within_3mm"]["on_capsule_area_mm2"] = round(float(an[np.abs(c6n) < 0.15].sum()), 2)
        res["v6_gated_cells_within_3mm"]["capsule_value_mm"] = C._st(c6n)
    except Exception as e:  # noqa: BLE001
        res["v6_gated_cells_within_3mm"]["capsule_error"] = "%s: %s" % (type(e).__name__, e)
rep = rep6.get("suppressed_stretches_mm") or {}
res["v6_bake"] = {"suppressed_stretches_mm": rep.get(sec), "rung": (rep6.get("ladder_v5b") or {}).get("rung"),
                  "final_attempt": (rep6.get("attempts") or [None])[-1]}
# 1.2: which v6 branch tubes (centreline radius) come near the lost points: tube clearance |x - c| - r per branch
s6 = D["s6"]
prox = {}
for bi, tag in enumerate(s6.tags):
    m = s6.b == bi
    if not m.any():
        continue
    P_, R_ = s6.p[m], s6.r[m]
    dd, ii = cKDTree(P_).query(X, k=min(16, len(P_)))
    dd, ii = (dd[:, None], ii[:, None]) if dd.ndim == 1 else (dd, ii)
    tc = (dd - R_[ii]).min(axis=1)
    if tc.min() < 3.0:
        prox[tag] = {"tube_clear_min_mm": round(float(tc.min()), 3), "tube_clear_median_mm": round(float(np.median(tc)), 3),
                     "lost_area_within_1mm_of_tube_mm2": round(float(a[tc < 1.0].sum()), 2)}
res["v6_branch_tubes_near_lost_points"] = prox
np.savez_compressed(outp[:-5] + ".npz", X=X, a=a, d=D["d"][bad], s_on6=s_on6)
json.dump(res, open(outp, "w"), indent=1, default=float)
print("DIAG_COV " + json.dumps(res, default=float))
