#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\gate_ct.py (= /sp/v6/wall/gate_ct.py in the build containers)
"""v6_wall gate 1: the audit's 140 floored points and 136 reference points (tbo/va_narrow/bone_scan.json, positions =
s on the v5 donor centreline, donors_v5/<unit>/<side>.json, native RAS) re-measured on a variant's stage-A surfaces.
Per point, in the plane normal to the v5 centreline (tangent P[j+2] - P[j-2], exactly as bone_scan.py):
  surf_req   r_eq of the section loop containing the point (ct_check.section_xy / loop_req, the audit's functions);
  surf_rmax  the loop's largest distance from the point (a leak indicator); n_loops;
  ct_req     bone_scan.py's own CT read (copied from bone_scan.json; NOT recomputed);
  ratio      ct_req / surf_req (the audit's quantity; v5: floored median 1.71, reference 1.04).
The CT read ends a bone-side ray at I_c + 150 HU, inside the bone's partial-volume rise, so at bone contact it reads
wider than the lumen (w025 LVA: 2.92 against the audit's 3.5 mm width). A second, calibre-continuity measure is
therefore reported: surf_req at the floored point / surf_req at its own reference point (same side, <= 30 mm).
    python3 gate_ct.py <variant_dir> <label>   -> <variant_dir>/gate_ct.json (+ summary printed)"""
import json, os, sys
import numpy as np
import pyvista as pv
sys.path.insert(0, "/sp/v5_va/tbo/va_narrow")
import ct_check as C  # noqa: E402
import vtk  # noqa: E402
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)

vdir, label = sys.argv[1:3]
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
bs_dir = "/sp/v5_va/tbo/va_narrow/bone_scan"
rows = []
surfs = {}
recs = {}


def surf(unit, side):
    k = (unit, side)
    if k not in surfs:
        if label == "v5":
            p = json.load(open(os.path.join(R5, unit, "%s.json" % side)))["surface"]["path"]
        else:
            p = os.path.join(vdir, unit, "%s_stageA.vtp" % ("rva" if side == "R" else "lva"))
        surfs[k] = pv.read(p).triangulate() if os.path.exists(p) else None
    return surfs[k]


def rec(unit, side):
    k = (unit, side)
    if k not in recs:
        r = json.load(open(os.path.join(R5, unit, "%s.json" % side)))
        P = np.asarray(r["centerline"]["points"], float)
        recs[k] = (P, C.TA.arclen(P))
    return recs[k]


def measure(unit, side, s_pt):
    P, s = rec(unit, side)
    j = int(np.argmin(np.abs(s - s_pt)))
    a, b = max(j - 2, 0), min(j + 2, len(P) - 1)
    t = C.unit(P[b] - P[a])
    e1, e2 = C.frame(t)
    m = surf(unit, side)
    if m is None:
        return None
    loops = C.section_xy(m, P[j], t, e1, e2)
    req = C.loop_req(loops)
    rmax = None
    for xy in loops:   # the loop loop_req picked: containing the origin, else nearest
        x, y = xy[:, 0], xy[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = ((y > 0) != (y2 > 0)) & (0 < (x2 - x) * (0 - y) / np.where(y2 - y == 0, 1e-30, y2 - y) + x)
        if cr.sum() % 2:
            rmax = float(np.linalg.norm(xy, axis=1).max())
    return {"surf_req": req, "surf_rmax": rmax, "n_loops": len(loops)}


for f in sorted(os.listdir(bs_dir)):
    if not f.endswith(".json"):
        continue
    d = json.load(open(os.path.join(bs_dir, f)))
    u = d["unit"]
    for sd, rr in d.get("sides", {}).items():
        for r in rr:
            row = {"unit": u, "side": sd, "s_from": r["s_from"], "s_to": r["s_to"], "len_mm": r["len_mm"]}
            for tag in ("floored", "reference"):
                if tag not in r:
                    continue
                m = measure(u, sd, r[tag]["s"])
                row[tag] = dict(s=r[tag]["s"], ct_req=r[tag]["ct_req"], bright=r[tag]["bright_ray_frac"],
                                v5_stageA_req=r[tag]["stageA_req"], misr_v5=r[tag]["misr"], **(m or {}))
            rows.append(row)


def q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return [round(float(v), 3) for v in np.percentile(a, [10, 50, 90])] if len(a) else None


F = [r["floored"] for r in rows if r.get("floored", {}).get("surf_req")]
Rf = [r["reference"] for r in rows if r.get("reference", {}).get("surf_req")]
cont = [r["floored"]["surf_req"] / r["reference"]["surf_req"] for r in rows
        if r.get("floored", {}).get("surf_req") and r.get("reference", {}).get("surf_req")]
summ = {"label": label, "floored_n": len(F), "reference_n": len(Rf),
        "floored": {"surf_req": q([x["surf_req"] for x in F]), "ct_req": q([x["ct_req"] for x in F]),
                    "ratio_ct_over_surf": q([x["ct_req"] / x["surf_req"] for x in F]),
                    "ratio_ge_1.25": sum(1 for x in F if x["ct_req"] / x["surf_req"] >= 1.25),
                    "ratio_ge_1.5": sum(1 for x in F if x["ct_req"] / x["surf_req"] >= 1.5),
                    "ratio_le_0.8_surface_wider": sum(1 for x in F if x["ct_req"] / x["surf_req"] <= 0.8),
                    "surf_req_lt_1.0": sum(1 for x in F if x["surf_req"] < 1.0),
                    "surf_rmax_gt_ct_plus_1.5": sum(1 for x in F if x["surf_rmax"] and x["surf_rmax"] > x["ct_req"] + 1.5)},
        "reference": {"surf_req": q([x["surf_req"] for x in Rf]), "ratio_ct_over_surf": q([x["ct_req"] / x["surf_req"] for x in Rf]),
                      "ratio_ge_1.5": sum(1 for x in Rf if x["ct_req"] / x["surf_req"] >= 1.5),
                      "ratio_le_0.8_surface_wider": sum(1 for x in Rf if x["ct_req"] / x["surf_req"] <= 0.8)},
        "continuity_floored_over_reference": q(cont),
        "continuity_lt_0.75": sum(1 for c in cont if c < 0.75), "continuity_gt_1.25": sum(1 for c in cont if c > 1.25),
        "n_pairs": len(cont)}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(vdir, "gate_ct.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
