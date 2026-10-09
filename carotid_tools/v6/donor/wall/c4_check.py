#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\c4_check.py (= /sp/v6/wall/c4_check.py in the build containers)
"""c4_check.py (v6_wall, V6_ITEMS c4): the topcow_mr_021 left-VA TopBrain label pinch -- where it is on the TopBrain
piece, which v5 anatomies carry it in their used range, what lumen it leaves there, and what "tube it at the
neighbouring calibre" would take.
  1. the TopBrain piece (topbrain_data/surfaces_va/topcow_mr_021_lVA.vtp + centerlines_lva/topcow_mr_021_va.json, native
     TopBrain frame): surface section r_eq every 0.5 mm along its centreline (ct_check.section_xy / loop_req), and the
     centreline radius (the graft's TB radius, floored at 1.0 by graft_v5_reg); the pinch = the deepest section, located
     independently of va_narrow (which found it from the w_025 anatomy); the neighbouring calibre = median r_eq 8-15 mm
     either side; the pinch stretch = sections below 0.75 x that calibre;
  2. every v5 anatomy whose xform va_R / va_L surface is that piece: the pinch mapped into the anatomy (x = (p * mirror -
     origin) @ R.T + anchor, sdf_union.load_surface's transform), its branch arclength, whether it lies past the cut (in
     use), the curve radius there, and the polyline lumen there (va_lumen_poly.py output, OBJ and collision_full);
  3. the tube override a v6 grafter would apply: TopBrain-native arclength range and radius profile (linear between the
     two neighbouring calibres), written to /v6w/c4/c4_tube_override.json.
    python3 c4_check.py <va_lumen_poly v5 json>   (neve-build-stagea)  -> /v6w/c4/c4_check.json
"""
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
sys.path.insert(0, "/sp/v5_va/tbo/va_narrow")
import ct_check as C  # noqa: E402

TB_S = "/repo/topbrain_data/surfaces_va/topcow_mr_021_lVA.vtp"
TB_C = "/repo/topbrain_data/centerlines_lva/topcow_mr_021_va.json"
A5 = "/repo/carotid_data/anatomies_v5"
OUT = "/v6w/c4"


def main():
    lum = {r["name"]: r for r in json.load(open(sys.argv[1]))} if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else {}
    os.makedirs(OUT, exist_ok=True)
    cl = json.load(open(TB_C))
    P = np.asarray(cl["points"], float)
    Rr = np.asarray(cl["radii"], float)
    s = C.TA.arclen(P)
    surf = pv.read(TB_S).triangulate()
    ss = np.arange(2.0, s[-1] - 2.0, 0.5)
    prof = []
    for x in ss:
        c = np.array([np.interp(x, s, P[:, k]) for k in range(3)])
        j = int(np.argmin(np.abs(s - x)))
        a, b = max(j - 3, 0), min(j + 3, len(P) - 1)
        t = C.unit(P[b] - P[a])
        e1, e2 = C.frame(t)
        prof.append({"s": float(x), "r_eq": C.loop_req(C.section_xy(surf, c, t, e1, e2)), "r_cl": float(np.interp(x, s, Rr)),
                     "z": float(c[2])})
    req = np.array([p_["r_eq"] if p_["r_eq"] else np.nan for p_ in prof])
    k = int(np.nanargmin(req))
    sp_ = ss[k]
    before = req[(ss >= sp_ - 15) & (ss <= sp_ - 8)]
    after = req[(ss >= sp_ + 8) & (ss <= sp_ + 15)]
    cal_b = float(np.nanmedian(before)) if np.isfinite(before).any() else np.nan
    cal_a = float(np.nanmedian(after)) if np.isfinite(after).any() else np.nan
    cal = np.nanmean([cal_b, cal_a])
    low = np.where(req < 0.75 * cal)[0]
    # the contiguous run around k
    i0 = i1 = k
    while i0 - 1 >= 0 and req[i0 - 1] < 0.75 * cal:
        i0 -= 1
    while i1 + 1 < len(req) and req[i1 + 1] < 0.75 * cal:
        i1 += 1
    pinch = {"s_tb": float(sp_), "r_eq_min": float(req[k]), "r_cl_at_min": prof[k]["r_cl"], "z": prof[k]["z"],
             "calibre_before_8_15": cal_b, "calibre_after_8_15": cal_a, "stretch_below_0.75cal_s": [float(ss[i0]), float(ss[i1])],
             "n_low_sections_total": int(len(low)), "tb_length_mm": float(s[-1])}
    pt_tb = np.array([np.interp(sp_, s, P[:, i]) for i in range(3)])
    rows = []
    for name in sorted(os.listdir(A5)):
        if not name.startswith("case_"):
            continue
        prov = json.load(open(os.path.join(A5, name, "provenance.json")))
        for key, tag in (("va_R", "RVA"), ("va_L", "LVA")):
            xf = (prov.get("xform") or {}).get(key) or {}
            if not str(xf.get("surface", "")).endswith("topcow_mr_021_lVA.vtp"):
                continue
            Rm, o, an, mi = (np.asarray(xf[k_], float) for k_ in ("R", "origin", "anchor", "mirror"))
            X = (pt_tb * mi - o) @ Rm.T + an
            br = {t: (p, r) for t, p, r in C.TA.load_branches(os.path.join(A5, name, "Centrelines_comb"))}
            p, r = br[tag]
            sa = C.TA.arclen(p)
            j = int(np.argmin(np.linalg.norm(p - X, axis=1)))
            cut = float(xf.get("cut_mm")) if xf.get("cut_mm") is not None else None
            L = lum.get(name, {}).get(tag, {})
            near = {}
            for m in ("obj", "full"):
                x = L.get(m) or {}
                tbm = (x.get("per_section") or {}).get("topbrain")
                near[m] = {"whole_min": x.get("lumen_min_mm"), "whole_at": x.get("at_mm"), "whole_section": x.get("section"),
                           "topbrain_min": (tbm or {}).get("min"), "topbrain_at": (tbm or {}).get("at")}
            rows.append({"name": name, "va": tag, "mirror": mi.tolist(), "cut_mm": cut, "pinch_s_anat": float(sa[j]),
                         "pinch_dist_to_branch_mm": float(np.linalg.norm(p[j] - X)),
                         "in_use": bool(cut is not None and sa[j] >= cut), "curve_r_at_pinch": float(r[j]) if len(r) else None,
                         "lumen": near})
    s_from, s_to = float(ss[i0]) - 3.0, float(ss[i1]) + 3.0
    # the graft carries centreline (MISR) radii, the real surface supplies the rest: the override ramps between the
    # piece's OWN centreline radius at the two ends, so the tube is continuous there (r_eq numbers kept for reference)
    rf, rt = float(np.interp(s_from, s, Rr)), float(np.interp(s_to, s, Rr))
    over = {"piece": TB_S, "centreline": TB_C, "s_from_tb": s_from, "s_to_tb": s_to,
            "radius_from": rf, "radius_to": rt, "convention": "centreline radius (the TopBrain line's MISR, as graft_v5_reg "
            "uses it, floored at VA_MIN_R 1.0)", "surface_req_neighbours_8_15mm": [cal_b, cal_a],
            "rule": "graft input: the TopBrain piece's centreline radius r(s) over [s_from, s_to] (TopBrain-native arclength "
                    "of centerlines_lva/topcow_mr_021_va.json, points mapped by nearest point) = max(own, linear from "
                    "radius_from to radius_to); the union is then that tube there (the pinched label surface lies inside it)",
            "anatomies": [r_["name"] + ":" + r_["va"] for r_ in rows if r_["in_use"]]}
    json.dump({"pinch": pinch, "profile": prof, "anatomies": rows, "tube_override": over},
              open(os.path.join(OUT, "c4_check.json"), "w"), indent=1)
    json.dump(over, open(os.path.join(OUT, "c4_tube_override.json"), "w"), indent=1)
    print(json.dumps(pinch, indent=1))
    for r_ in rows:
        print("%-40s %s mirror %s cut %s pinch s %.1f (%.2f mm off) in_use %s r_cl %.2f | OBJ whole %s at %s (%s) TB %s | full TB %s" % (
            r_["name"], r_["va"], r_["mirror"][0], r_["cut_mm"], r_["pinch_s_anat"], r_["pinch_dist_to_branch_mm"],
            r_["in_use"], r_["curve_r_at_pinch"] or -1, r_["lumen"]["obj"]["whole_min"], r_["lumen"]["obj"]["whole_at"],
            r_["lumen"]["obj"]["whole_section"], r_["lumen"]["obj"]["topbrain_min"], r_["lumen"]["full"]["topbrain_min"]))
    print("override", json.dumps(over))


if __name__ == "__main__":
    main()
