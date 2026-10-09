#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\d15_report.py (= /sp/v6/wall_local/d15_report.py in the build containers)
"""d15_report.py (s2_harden_wall_bake): one D15 report from the result files (no new measurement):
  surface identity per side (seg/local/<u>/local_qc.json), line splice (seg/local/_splice_summary.json), wall detail per
  donor side v5 / plateau / local and the outside-zone identity (local/wd_local.json), CT vs surface at the audit's 140
  floored + 136 reference points (seg/{v5ref,plateau,local}/gate_ct.json), floored stretches (audit floor_scan v5,
  scan/{plateau,local}/floor_scan.json), the subclavian / junction (local/wd_sub_local.json, subclavian_local/*/local_qc.json),
  the VA-tool junction records (donors_local/_junction_records/_summary.json).
    python3 d15_report.py -> /v6w/donors_local/D15_REPORT.json + printed summary"""
import glob
import json
import os

import numpy as np


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


rep = {"what": "D15 local donor wall (s2_harden_wall_bake): v5's coarse ct_seg4 wall outside the artefact zones, stage 1's "
               "plateau rule inside (2 mm blend)", "files": {}}
# 1 surface identity
ident = {"sides": 0, "all_outside_vertices_equal_v5": True, "all_outside_triangles_equal_v5": True,
         "v5_recompute_equal": True, "spill_undone_max_mm": 0.0, "core_frac": [], "w_gt0_frac": [], "genus_v5": [],
         "genus_local": [], "multi_component_local": [], "per_side": {}}
for f in sorted(glob.glob("/v6w/seg/local/*/local_qc.json")):
    q = jl(f)
    for sd, s in q["sides"].items():
        su = s.get("surface") or {}
        k = "%s:%s" % (q["unit"], sd)
        ident["sides"] += 1
        ident["all_outside_vertices_equal_v5"] &= su.get("outside_vertices_equal_v5_file") == su.get("vertices_outside_zones")
        ident["all_outside_triangles_equal_v5"] &= su.get("triangles_outside_equal_v5") == su.get("triangles_outside")
        ident["v5_recompute_equal"] &= bool(su.get("v5_recompute_equals_v5_file"))
        ident["spill_undone_max_mm"] = max(ident["spill_undone_max_mm"], su.get("spill_undone_max_mm") or 0)
        z = s["zones"]
        ident["core_frac"].append(z["core_frac"])
        ident["w_gt0_frac"].append(z["w_gt0_frac"])
        t, t5 = su.get("topology") or {}, su.get("topology_v5") or {}
        ident["genus_local"].append(t.get("genus_if_one_component"))
        ident["genus_v5"].append(t5.get("genus_if_one_component"))
        if (t.get("components") or 1) > 1:
            ident["multi_component_local"].append(k)
        ident["per_side"][k] = {"core_mm": z["core_mm"], "w_gt0_mm": z["weighted_mm"], "length_mm": z["length_mm"],
                                "core_runs": len(z["core_runs_s_mm"]), "detectors_mm": z["detectors_mm"],
                                "outside_vertices": su.get("vertices_outside_zones"), "vertices": su.get("vertices"),
                                "genus_local": t.get("genus_if_one_component"), "genus_v5": t5.get("genus_if_one_component"),
                                "grid": q.get("grid")}
for k in ("core_frac", "w_gt0_frac"):
    v = np.asarray(ident[k], float)
    ident[k] = [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
for k in ("genus_local", "genus_v5"):
    v = [x for x in ident[k] if x is not None]
    ident[k] = {"sum": float(sum(v)), "n": len(v)}
rep["surface_identity"] = ident
# 2 splice
sp = jl("/v6w/seg/local/_splice_summary.json") or {}
gaps = [v.get("junction_gap_max_mm") for v in sp.values() if isinstance(v, dict) and v.get("status") == "spliced"]
devs = [v.get("raw_outside_zones_dev_max_mm") for v in sp.values() if isinstance(v, dict) and v.get("status") == "spliced"]
rep["line_splice"] = {"sides": len(sp), "spliced": len(gaps), "not_spliced": {k: v.get("status") for k, v in sp.items() if v.get("status") != "spliced"},
                      "junction_gap_max_mm": max(gaps) if gaps else None, "junction_gap_med_mm": float(np.median(gaps)) if gaps else None,
                      "raw_vmtk_outside_dev_max_mm": max(devs) if devs else None,
                      "raw_vmtk_outside_dev_med_mm": float(np.median(devs)) if devs else None}
# 2b records: the local record line outside the zones IS v5's record line (points and MISR), the surface md5 is the
# local stage-A surface's
recs = {"sides": 0, "outside_points_equal_v5": 0, "outside_points": 0, "sides_all_equal": 0, "surface_md5_ok": 0, "per_side": {}}
D = jl("/sp/v6/wall/donors.json")
from scipy.spatial import cKDTree  # noqa: E402
for f in sorted(glob.glob("/v6w/donors_local/*/[RL].json")):
    unit, sd = f.split("/")[-2], os.path.basename(f)[0]
    r6 = jl(f)
    r5 = jl("/repo/carotid_data/va_sources/v5_reg/donors_v5/%s/%s.json" % (unit, sd))
    zf = jl("/v6w/seg/local/%s/zones.json" % unit) or {}
    if not r5 or sd not in zf:
        continue
    st = jl(os.path.join(D[unit]["v5_seg_dir"], "stations_rays.json"))["rva" if sd == "R" else "lva"]
    w = np.asarray(zf[sd]["w_station"], float)
    Ps = np.asarray(st["P"], float)[:len(w)]
    P5, M5 = np.asarray(r5["centerline"]["points"], float), np.asarray(r5["centerline"]["radii_misr"], float)
    P6, M6 = np.asarray(r6["centerline"]["points"], float), np.asarray(r6["centerline"]["radii_misr"], float)
    w5 = w[cKDTree(Ps).query(P5)[1]]
    k6 = {tuple(np.round(p, 9)): m for p, m in zip(P6, M6)}
    out = np.nonzero(w5 == 0)[0]
    eq = sum(1 for i in out if tuple(np.round(P5[i], 9)) in k6 and abs(k6[tuple(np.round(P5[i], 9))] - M5[i]) < 1e-12)
    recs["sides"] += 1
    recs["outside_points"] += len(out)
    recs["outside_points_equal_v5"] += eq
    recs["sides_all_equal"] += int(eq == len(out))
    sv = (r6.get("surface") or {})
    ok_md5 = sv.get("md5") is not None and os.path.exists(sv.get("path", "").replace("/repo/carotid_data/v6_work/wall", "/v6w"))
    recs["surface_md5_ok"] += int(bool(ok_md5))
    recs["per_side"]["%s:%s" % (unit, sd)] = {"outside_points": len(out), "equal": eq,
                                             "floored_mm_misr_lt_1": (r6.get("v6") or {}).get("floored_mm_misr_lt_1"),
                                             "misr_median": (r6.get("v6") or {}).get("misr_median"),
                                             "length_mm": (r6.get("v6") or {}).get("length_mm")}
fl = [v["floored_mm_misr_lt_1"] for v in recs["per_side"].values() if v.get("floored_mm_misr_lt_1")]
if fl:
    recs["floored_mm_misr_lt_1_total"] = {"v5": round(sum(x["v5"] for x in fl), 1), "local": round(sum(x["v6"] for x in fl), 1)}
rep["records"] = recs
# 3 wall detail
wd = jl("/v6w/local/wd_local.json")
if wd:
    rep["wall_detail"] = {"summary": wd["summary"], "per_donor_side": wd["per_donor_side"]}
    psi = wd.get("per_side_identity") or {}
    rep["wall_detail"]["outside_identity_per_side"] = {
        "sides": len(psi), "bin_frac_absdiff_max": max((v["outside_bin_frac_absdiff_max"] for v in psi.values()), default=None),
        "bins_identical_frac_min": min((v["outside_bins_identical_frac"] for v in psi.values()), default=None),
        "bins_identical_frac_med": float(np.median([v["outside_bins_identical_frac"] for v in psi.values()])) if psi else None}
# 4 CT vs surface
ct = {}
for lab, f in (("v5", "/v6w/seg/v5ref/gate_ct.json"), ("plateau", "/v6w/seg/plateau/gate_ct.json"), ("local", "/v6w/seg/local/gate_ct.json")):
    g = jl(f)
    if g:
        ct[lab] = g["summary"]
rep["ct_vs_surface"] = ct
# 5 floored stretches
fs = {}
for lab, f in (("v5_audit", "/sp/v5_va/tbo/va_narrow/floor_scan.json"), ("plateau", "/v6w/scan/plateau/floor_scan.json"),
               ("local", "/v6w/scan/local/floor_scan.json")):
    g = jl(f)
    if g:
        fs[lab] = g.get("summary")
        tot = sum(float(d.get("floored_mm_total") or 0) for d in (g.get("donors") or {}).values())
        fs[lab + "_floored_mm_on_walls"] = round(tot, 1)
for lab, f in (("plateau", "/v6w/scan/plateau/bone_scan.json"), ("local", "/v6w/scan/local/bone_scan.json")):
    g = jl(f)
    if g:
        fs[lab + "_bone_scan"] = g.get("summary")
rep["floored"] = fs
# 6 subclavian / junction
ws = jl("/v6w/local/wd_sub_local.json")
if ws:
    rep["subclavian"] = ws["summary"]
jr = jl("/v6w/donors_local/_junction_records/_summary.json")
if jr:
    rep["junction_records"] = {"sides": len(jr), "failed": {k: v.get("junction_failed") for k, v in jr.items() if v.get("junction_failed")},
                               "status_not_ok": {k: v.get("status") for k, v in jr.items() if v.get("status") not in (None, "ok")}}
json.dump(rep, open("/v6w/donors_local/D15_REPORT.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rep.items() if k not in ("wall_detail",)}, indent=1, default=str)[:6000])
if wd:
    print(json.dumps(wd["summary"], indent=1)[:3000])
