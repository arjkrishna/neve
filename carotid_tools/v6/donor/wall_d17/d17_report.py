#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\d17_report.py (= /sp/v6/wall_d17/d17_report.py in the build containers)
"""d17_report.py (s2_harden_wall_bake): the D16 + D17 report from the result files.
  paths (d17/paths/_d17_summary.json), requests (d17/req/_summary.json), the LOCAL subclavian + junction walls
  (d17/local/<u>/local_qc.json: surface identity outside the zones vs the v5-rule run, sub zones, D16 tube, junction
  bottleneck), the VA-tool records (d17/_junction_records/<d>/<s>/junction.json: covered stretch, label end past the
  ostium, junction_failed, genus), wall detail (wd_proxy bins: d17_v5rule = v5-rule surface + its line; d17_local_v5line
  = local surface along the SAME v5-rule line -> bin-for-bin identity outside the zones; d17_local = local surface +
  its own line, as built).
    python3 d17_report.py -> /v6w/d17/D17_REPORT.json (+ printed summary)    (neve-build-stagea)"""
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree

R = "/v6w/d17"
GUARD = 3


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def q(v):
    v = np.asarray([x for x in v if x is not None], float)
    return {"p10": round(float(np.percentile(v, 10)), 4), "median": round(float(np.median(v)), 4),
            "p90": round(float(np.percentile(v, 90)), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "n": int(len(v))} if len(v) else None


rep = {"what": "D16 (pinched ostia tube-bridged) + D17 (the donor's own subclavian to the end of its segmentation; "
               "D15 local wall)", "paths": jl(R + "/paths/_d17_summary.json"), "requests": jl(R + "/req/_summary.json"),
       "paths_check_vs_va_tool": (jl(R + "/_junction_records/_paths_check.json") or {}).get("identical")}
sides = {}
for f in sorted(glob.glob(R + "/local/*/local_qc.json")):
    qc = jl(f)
    u = qc["unit"]
    for sd, s in qc["sides"].items():
        x = "r" if sd == "R" else "l"
        sn = "rsub" if sd == "R" else "lsub"
        row = {"sub_zone_core_mm": (s.get("sub_zones") or {}).get("core_mm"),
               "sub_length_mm": (s.get("sub_zones") or {}).get("length_mm")}
        for t in (sn, "%sjunction" % x):
            a = s.get(t) or {}
            row[t] = {k: a.get(k) for k in ("v5rule_recompute_equals_file", "vertices_outside_zones", "outside_vertices_equal_v5rule_file",
                                            "triangles_outside", "triangles_outside_equal_v5rule", "spill_undone_max_mm",
                                            "largest_choice_differs_outside_vox", "topology")}
        jb = s.get("junction_bridge_local") or {}
        row["junction_bottleneck_ratio"] = (jb.get("after") or {}).get("bottleneck_ratio")
        row["junction_bottleneck_ratio_v5rule_run"] = ((jb.get("local") or {}).get("v5rule_after") or {}).get("bottleneck_ratio")
        row["d16"] = s.get("d16")
        r = jl("%s/_junction_records/%s/%s/junction.json" % (R, u, sd)) or {}
        row["record"] = {"covered_s_mm": r.get("covered_s_mm"), "label_end_past_ostium_mm": (r.get("d17") or {}).get("label_end_past_ostium_mm"),
                         "junction_failed": r.get("junction_failed"), "ostial_tube_bridge": r.get("ostial_tube_bridge"),
                         "surface_info": r.get("surface_info"), "misr_median_stretch": r.get("misr_median_stretch"),
                         "tube_extension_needed": (r.get("d17") or {}).get("tube_extension_needed")}
        # wall detail
        st = jl("%s/v5rule/%s/stations_rays.json" % (R, u)) or {}
        z = (jl("%s/local/%s/sub_zones.json" % (R, u)) or {}).get(sd) or {}
        wd = {}
        for tag in ("d17_v5rule", "d17_local_v5line", "d17_local"):
            d = jl("/v6w/wd/%s/%s_%s.json" % (tag, u, sd))
            if d:
                ar, a3 = np.asarray(d["bins_1mm"]["area"], float), np.asarray(d["bins_1mm"]["a03"], float)
                wd[tag] = {"whole": float(a3.sum() / max(ar.sum(), 1e-9)), "bins": int(len(ar))}
                wd[tag + "_bins"] = (ar, a3)
        if "d17_v5rule" in wd and "d17_local_v5line" in wd and st.get(sn) and z.get("w_sub_station") is not None:
            a5, b5 = wd["d17_v5rule_bins"]
            al, bl = wd["d17_local_v5line_bins"]
            vj = jl("%s/v5rule/%s/%s_stageA_vmtk.json" % (R, u, sn))
            import sys
            sys.path.insert(0, "/opt/eve_training/topbrain_tools")
            from graft_siphon import prep_siphon
            p, _ = prep_siphon(np.asarray(vj["points"], float), np.asarray(vj["radii"], float), 1.0)
            s_ = arclen(p)
            nb = min(len(a5), len(al))
            cen = np.minimum(np.arange(nb) + 0.5, s_[-1])
            Cb = np.stack([np.interp(cen, s_, p[:, k]) for k in range(3)], 1)
            w = np.asarray(z["w_sub_station"], float)
            Ps = np.asarray(st[sn]["P"], float)[:len(w)]
            wb = w[cKDTree(Ps).query(Cb)[1]]
            if row["d16"]:
                # the D16 tube region is a zone too (within r + 2 mm of the tube axis)
                A = np.asarray([row["d16"]["from_xyz"], row["d16"]["to_xyz"]], float)
                dd = np.min(np.linalg.norm(Cb[:, None, :] - A[None], axis=2), axis=1)
                wb = np.where(dd <= row["d16"]["r_mm"] + 2.0 + 3.0, 1.0, wb)
            near = np.zeros(nb, bool)
            for i in np.nonzero(wb > 0)[0]:
                near[max(0, i - GUARD):i + GUARD + 1] = True
            out = ~near
            f5 = b5[:nb] / np.maximum(a5[:nb], 1e-9)
            fl = bl[:nb] / np.maximum(al[:nb], 1e-9)
            row["wd_identity_outside"] = {
                "bins": int(nb), "bins_outside": int(out.sum()),
                "bins_identical": int(np.sum((np.abs(b5[:nb] - bl[:nb]) < 1e-9) & (np.abs(a5[:nb] - al[:nb]) < 1e-9) & out)),
                "bin_frac_absdiff_max": float(np.abs(fl - f5)[out].max()) if out.any() else 0.0,
                "frac_outside_v5rule": float(b5[:nb][out].sum() / max(a5[:nb][out].sum(), 1e-9)),
                "frac_outside_local": float(bl[:nb][out].sum() / max(al[:nb][out].sum(), 1e-9)),
                "frac_zone_v5rule": float(b5[:nb][~out].sum() / max(a5[:nb][~out].sum(), 1e-9)) if (~out).any() else None,
                "frac_zone_local": float(bl[:nb][~out].sum() / max(al[:nb][~out].sum(), 1e-9)) if (~out).any() else None}
        row["wall_detail"] = {k: v for k, v in wd.items() if not k.endswith("_bins")}
        sides["%s:%s" % (u, sd)] = row
rep["sides"] = sides
S = list(sides.values())
rep["summary"] = {
    "sides": len(S),
    "label_end_past_ostium_mm": q([r["record"]["label_end_past_ostium_mm"] for r in S]),
    "tube_extension_needed": sorted(k for k, r in sides.items() if r["record"]["tube_extension_needed"]),
    "junction_failed": {k: r["record"]["junction_failed"] for k, r in sides.items() if r["record"]["junction_failed"]},
    "ostial_tube_bridge": sorted(k for k, r in sides.items() if r["record"]["ostial_tube_bridge"]),
    "d16": {k: {kk: r["d16"].get(kk) for kk in ("ratio_before", "ratio_after", "tubed_mm", "r_mm", "S_mm")} for k, r in sides.items() if r["d16"]},
    "junction_bottleneck_ratio": q([r["junction_bottleneck_ratio"] for r in S]),
    "junction_bottleneck_lt_0.5": sorted(k for k, r in sides.items() if (r["junction_bottleneck_ratio"] or 0) < 0.5),
    "sub_zone_core_mm": q([r["sub_zone_core_mm"] for r in S]),
    "sub_zone_sides_0mm": sum(1 for r in S if (r["sub_zone_core_mm"] or 0) == 0),
    "surface_identity_outside": {
        t: {"v5rule_recompute_equal_all": all(bool((r[t2] or {}).get("v5rule_recompute_equals_file")) for r in S for t2 in [k for k in r if k.endswith(t)]),
            "vertices_all_equal": all((r[t2] or {}).get("outside_vertices_equal_v5rule_file") == (r[t2] or {}).get("vertices_outside_zones")
                                      for r in S for t2 in [k for k in r if k.endswith(t)]),
            "triangles_not_equal_total": int(sum(((r[t2] or {}).get("triangles_outside") or 0) - ((r[t2] or {}).get("triangles_outside_equal_v5rule") or 0)
                                                 for r in S for t2 in [k for k in r if k.endswith(t)]))}
        for t in ("sub", "junction")},
    "genus_junction": q([((r["record"]["surface_info"] or {}).get("genus_largest")) for r in S]),
    "open_edges_junction_max": max(((r["record"]["surface_info"] or {}).get("open_or_nonmanifold_edges") or 0) for r in S) if S else None,
    "wd_whole": {t: q([(r["wall_detail"].get(t) or {}).get("whole") for r in S]) for t in ("d17_v5rule", "d17_local_v5line", "d17_local")},
    "wd_identity_outside": {"bin_frac_absdiff_max": max((r.get("wd_identity_outside") or {}).get("bin_frac_absdiff_max", 0) for r in S) if S else None,
                            "frac_outside_absdiff_max": max(abs(r["wd_identity_outside"]["frac_outside_local"] - r["wd_identity_outside"]["frac_outside_v5rule"])
                                                            for r in S if r.get("wd_identity_outside")) if any(r.get("wd_identity_outside") for r in S) else None,
                            "bins_outside_total": sum((r.get("wd_identity_outside") or {}).get("bins_outside", 0) for r in S),
                            "bins_identical_total": sum((r.get("wd_identity_outside") or {}).get("bins_identical", 0) for r in S)}}
json.dump(rep, open(R + "/D17_REPORT.json", "w"), indent=1, default=float)
print(json.dumps(rep["summary"], indent=1, default=float))
