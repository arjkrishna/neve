#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\collect.py (= /sp/v5_va/checks/collect.py in the build containers)
"""v5_checks: collect items 1-5 into /sp/v5_va/results/v5_checks.json (+ a printed summary for the notes).
    python3 collect.py     (neve-build-meshlab, read-only on /repo)
Inputs: checks/out/<name>.json (chk_one.py), checks/loader_v5.json, results/split_v5_baked.json + checks/logs/split.log,
checks/sofa_selection.json + checks/sofa/run4/{v5,v4}/<name>.json, checks/va_lumen.json, checks/item5_v4_untouched.json,
results/v5_bake_full.json (the bake's claims, for agreement).
"""
import glob
import json
import os

import numpy as np

C = "/sp/v5_va/checks"
A5 = "/repo/carotid_data/anatomies_v5"
REF = {"host_tube_0.3mm": 0.0003, "v4_whole_va_0.3mm": 0.021, "topbrain_0.3mm": 0.12, "topbrain_in_situ_0.3mm_ref_c": 0.125,
       "v3_rcca_fork_0.3mm": 0.196, "v3_rcca_siphon_0.3mm": 0.322, "ref_c_rcca_fork_rel_0.15": 0.130,
       "ref_c_rcca_siphon_rel_0.15": 0.257, "source": "notes/ref.md parts c and d; CAROTID_V4.md wall detail"}


def q(x, nd=4):
    x = [v for v in x if v is not None]
    if not x:
        return None
    a = np.asarray(x, float)
    return {"n": int(len(a)), "min": round(float(a.min()), nd), "p10": round(float(np.percentile(a, 10)), nd),
            "median": round(float(np.median(a)), nd), "p90": round(float(np.percentile(a, 90)), nd),
            "max": round(float(a.max()), nd), "mean": round(float(a.mean()), nd)}


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


names = sorted(n for n in os.listdir(A5) if n.startswith("case_"))
recs, missing, errors = {}, [], []
for n in names:
    p = os.path.join(C, "out", n + ".json")
    if os.path.exists(p):
        recs[n] = json.load(open(p))
    elif os.path.exists(os.path.join(C, "out", n + ".error.json")):
        errors.append(n)
    else:
        missing.append(n)
bake = {a["name"]: a for a in json.load(open("/sp/v5_va/results/v5_bake_full.json"))["anatomies"]}
R = {"what": "v5_checks: post-bake checks on the 223 baked v5 anatomies (items 1-5)", "generated_by": "checks/collect.py",
     "n_anatomies": len(names), "n_checked": len(recs), "missing": missing, "errors": errors, "references": REF}

# ------------------------------------------------------------------ item 1
G = {n: r["item1"]["gates"] for n, r in recs.items()}
keys_bool = ["obj_clean_raw", "obj_clean_welded", "full_clean_raw", "full_clean_welded", "union_clean_welded",
             "obj_pml_loader_clean", "numpy_obj_ok", "numpy_full_ok", "obj_budget_ok", "full_budget_ok", "obj_file_clean",
             "full_file_all_tri", "routes_navigable_obj", "routes_navigable_full", "orientation_as_v4", "strict_pass"]
it1 = {"pass_counts": {k: sum(bool(G[n][k]) for n in G) for k in keys_bool},
       "self_intersecting_faces_total": {m: sum((G[n]["selfint"][m] or 0) for n in G) for m in ("obj", "full", "union")},
       "self_intersecting_faces_welded_total": {m: sum((G[n]["selfint"][m + "_welded"] or 0) for n in G) for m in ("obj", "full", "union")},
       "obj_tris": q([G[n]["obj_tris"] for n in G], 0), "full_tris": q([G[n]["full_tris"] for n in G], 0),
       "union_tris": q([G[n]["union_tris"] for n in G], 0),
       "v4_obj_tris": q([recs[n]["item1"]["v4_obj_tris"] for n in recs], 0),
       "route_lumen_min_obj_RCCA": q([G[n]["route_lumen_min_obj"]["RCCA"] for n in G], 3),
       "route_lumen_min_obj_LCCA": q([G[n]["route_lumen_min_obj"]["LCCA"] for n in G], 3),
       "route_lumen_min_full_RCCA": q([g(recs[n], "item1", "full", "routes", "RCCA", "lumen_min_mm") for n in recs], 3),
       "route_lumen_min_full_LCCA": q([g(recs[n], "item1", "full", "routes", "LCCA", "lumen_min_mm") for n in recs], 3),
       "centreline_points_total": sum(G[n]["centreline_points"] for n in G),
       "centreline_outside_obj_all_total": sum(G[n]["centreline_outside_obj_all"] for n in G),
       "centreline_outside_full_all_total": sum(G[n]["centreline_outside_full_all"] for n in G),
       "centreline_outside_obj_caps_excluded_total": sum(G[n]["centreline_outside_obj_caps_excl"] for n in G),
       "full_to_union_mm_p99": q([recs[n]["item1"]["full_to_union_mm"]["p99"] for n in recs], 4),
       "full_to_union_mm_max": q([recs[n]["item1"]["full_to_union_mm"]["max"] for n in recs], 4),
       "obj_to_union_mm_p99": q([recs[n]["item1"]["obj_to_union_mm"]["p99"] for n in recs], 4),
       "obj_to_union_mm_max": q([recs[n]["item1"]["obj_to_union_mm"]["max"] for n in recs], 4),
       "signed_volume_sign": {"v5_obj_negative": sum(G[n]["signed_volume_mm3"]["obj"] < 0 for n in G),
                              "v4_obj_negative": sum(G[n]["signed_volume_mm3"]["v4_obj"] < 0 for n in G)},
       "min_angle_deg_obj": q([recs[n]["item1"]["obj"]["geom"]["min_angle_deg"] for n in recs], 3),
       "zero_area_faces_obj_total": sum(recs[n]["item1"]["obj"]["geom"]["zero_area_faces"] for n in recs),
       "failing": sorted(n for n in G if not G[n]["strict_pass"]),
       "failing_detail": {n: {k: G[n][k] for k in keys_bool if not G[n][k]} for n in G if not G[n]["strict_pass"]}}
# agreement with the bake's own record
agree = 0
for n in G:
    b = bake.get(n, {})
    ok = (b.get("clean", {}).get("obj") == G[n]["obj_clean_raw"] and b.get("obj_tris") == G[n]["obj_tris"]
          and b.get("full_tris") == G[n]["full_tris"] and b.get("union_tris") == G[n]["union_tris"])
    agree += ok
it1["agrees_with_bake_record"] = agree
R["item1_strict_mesh_audit"] = it1

# ------------------------------------------------------------------ item 2
rr = {n: r["item2"]["right_route_detail"] for n, r in recs.items() if "item2" in r}
it2 = {"recomputed_equals_mesh_v5_record": sum(bool(rr[n]["recomputed_equals_v5_record"]) for n in rr),
       "identical_to_v4": sum(bool(rr[n]["identical_v4"]) for n in rr),
       "identical_to_v4_names_not": sorted(n for n in rr if not rr[n]["identical_v4"]),
       "v4_rebuild_max_abs_diff_vs_mesh_v4": max(recs[n]["item2"]["v4_rebuild"]["max_abs_diff_vs_mesh_v4"] for n in rr),
       "ratio": {f"{seg}_{t}": q([rr[n]["ratio_v5_v4"][seg][t] for n in rr], 4)
                 for seg in ("fork", "siphon") for t in ("frac_0.3mm", "frac_0.5mm")},
       "max_abs_diff_v4": q([rr[n]["max_abs_diff_v4"] for n in rr], 4)}
ib = [recs[n]["item2"]["identical_beyond_mm"] for n in rr]
it2["identical_beyond_mm_counts"] = {str(k): ib.count(k) for k in (0, 4, 6, 8, None)}
low = []
for n in rr:
    for seg in ("fork", "siphon"):
        v = rr[n]["ratio_v5_v4"][seg]["frac_0.3mm"]
        if v is not None and v < 0.99:
            low.append((n, seg, v))
it2["ratio_0.3mm_below_0.99"] = sorted(low, key=lambda x: x[2])
it2["n_anatomies_ratio_0.3mm_below_0.99"] = len({x[0] for x in low})
it2["n_anatomies_ratio_0.3mm_below_0.95"] = len({x[0] for x in low if x[2] < 0.95})
hi_ = {n for n in rr for seg in ("fork", "siphon") if (rr[n]["ratio_v5_v4"][seg]["frac_0.3mm"] or 1) > 1.01}
it2["n_anatomies_ratio_0.3mm_above_1.01"] = len(hi_)
# agreement with the bake's identical / beyond
it2["agrees_with_bake_identical"] = sum(bool(rr[n]["identical_v4"]) == bool(g(bake.get(n, {}), "right_route_detail", "identical")) for n in rr)
it2["agrees_with_bake_beyond"] = sum(recs[n]["item2"]["identical_beyond_mm"] == g(bake.get(n, {}), "right_route_detail", "identical_beyond_mm") for n in rr)
it2["bake_beyond_disagreements"] = {n: [recs[n]["item2"]["identical_beyond_mm"], g(bake.get(n, {}), "right_route_detail", "identical_beyond_mm")]
                                    for n in rr if recs[n]["item2"]["identical_beyond_mm"] != g(bake.get(n, {}), "right_route_detail", "identical_beyond_mm")}
# RCCA in the same unions
it2["RCCA_v5"] = {f"{seg}_{t}": q([g(recs[n], "item2", "RCCA", "v5", seg, t) for n in rr], 4)
                  for seg in ("fork", "siphon") for t in ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15")}
it2["RCCA_v4"] = {f"{seg}_{t}": q([g(recs[n], "item2", "RCCA", "v4", seg, t) for n in rr], 4)
                  for seg in ("fork", "siphon") for t in ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15")}
# VAs
rows = []
for n in rr:
    for side in ("right", "left"):
        d = recs[n]["item2"]["va"][side]
        row = {"name": n, "side": side, "donor": d["donor"], "v4_va_kept": d["v4_va_kept"],
               "v4_host_tube": g(d, "v4", "host_tube", "frac_0.3mm"), "v4_whole_va": g(d, "v4", "whole_va", "frac_0.3mm"),
               "v4_topbrain": g(d, "v4", "topbrain", "frac_0.3mm"), "v4_topbrain_real_surface": g(d, "v4", "topbrain_real_surface")}
        if d["donor"]:
            v = d["v5"]
            tu = d["topbrain_unchanged"]
            row.update({"donor_id": v["donor"], "ramp_side": v["ramp_side"], "donor_stretch_mm": v["donor_stretch_mm"],
                        "tubed_mm": v["tubed_mm"], "tubed_frac": v["tubed_frac"], "n_windows": len(v["windows_mm"]),
                        "real_surface": v["real_surface_flag"] and not v["tube_only"],
                        "whole_tube": bool((v.get("ladder_record") or {}).get("whole_tube")),
                        "donor_real": g(v, "donor_real", "frac_0.3mm"), "donor_real_0.5": g(v, "donor_real", "frac_0.5mm"),
                        "donor_real_rel_0.15": g(v, "donor_real", "frac_rel_0.15"),
                        "donor_as_baked": g(v, "donor_stretch_as_baked", "frac_0.3mm"),
                        "windows_only": g(v, "windows_only", "frac_0.3mm"), "seam_zone": g(v, "seam_zone", "frac_0.3mm"),
                        "seam_zone_rel_0.15": g(v, "seam_zone", "frac_rel_0.15"),
                        "junction_stub": g(v, "junction_stub", "frac_0.3mm"),
                        "topbrain": g(v, "topbrain", "frac_0.3mm"), "topbrain_real": g(v, "topbrain_real", "frac_0.3mm"),
                        "whole_va_from_origin": g(v, "whole_va_from_origin", "frac_0.3mm"),
                        "whole_va_from_origin_0.5": g(v, "whole_va_from_origin", "frac_0.5mm"),
                        "whole_va_from_origin_rel_0.15": g(v, "whole_va_from_origin", "frac_rel_0.15"),
                        "whole_va_from_ostium": g(v, "whole_va_from_ostium", "frac_0.3mm"),
                        "tb_len_v5": tu["len_v5_mm"], "tb_len_v4": tu["len_v4_mm"], "tb_kabsch_rms": tu["kabsch_rms_mm"],
                        "tb_radius_diff": tu["radius_max_abs_diff_mm"], "tb_surface_same": tu["surface_same"],
                        "tb_mirror_same": tu["mirror_same"], "tb_R_det": tu["R_det"],
                        "tb_aligned_v5": g(tu, "detail_aligned", "v5", "frac_0.3mm"), "tb_aligned_v4": g(tu, "detail_aligned", "v4", "frac_0.3mm"),
                        "tb_aligned_ratio": g(tu, "detail_aligned", "ratio_0.3mm"),
                        "tb_aligned_ratio_rel": g(tu, "detail_aligned", "ratio_rel_0.15"),
                        "bake_donor_real": g(bake.get(n, {}), "va", side, "donor_real_frac_0.3mm"),
                        "bake_kept_in_situ_over_alone": g(bake.get(n, {}), "va", side, "A2_detail_kept_in_situ_over_alone")})
        else:
            row.update({"v5_whole_va": g(d, "v5", "whole_va", "frac_0.3mm"), "v5_host_tube": g(d, "v5", "host_tube", "frac_0.3mm"),
                        "v5_topbrain": g(d, "v5", "topbrain", "frac_0.3mm"), "centreline_identical_v4": g(d, "v5", "centreline_identical_v4")})
        rows.append(row)
don = [r for r in rows if r["donor"]]
kept = [r for r in rows if not r["donor"]]
va = {"n_va": len(rows), "n_donor_va": len(don), "n_v4_kept_sides": len(kept),
      "donor_real_surface": sum(r["real_surface"] for r in don), "donor_whole_tube": sum(r["whole_tube"] for r in don),
      "donor_with_window": sum(r["n_windows"] > 0 for r in don),
      "tubed_mm_windowed": q([r["tubed_mm"] for r in don if r["n_windows"]], 2),
      "tubed_frac_windowed": q([r["tubed_frac"] for r in don if r["n_windows"]], 4),
      "segments_frac_0.3mm": {k: q([r[k] for r in don], 4) for k in
                              ("donor_real", "donor_as_baked", "windows_only", "seam_zone", "junction_stub", "topbrain",
                               "topbrain_real", "whole_va_from_origin", "whole_va_from_ostium",
                               "v4_host_tube", "v4_whole_va", "v4_topbrain")},
      "donor_real_frac_0.5mm": q([r["donor_real_0.5"] for r in don], 4),
      "donor_real_frac_rel_0.15": q([r["donor_real_rel_0.15"] for r in don], 4),
      "whole_va_from_origin_frac_0.5mm": q([r["whole_va_from_origin_0.5"] for r in don], 4),
      "whole_va_from_origin_frac_rel_0.15": q([r["whole_va_from_origin_rel_0.15"] for r in don], 4),
      "seam_zone_frac_rel_0.15": q([r["seam_zone_rel_0.15"] for r in don], 4),
      "donor_real_gt_v4_host_tube_same_side": sum((r["donor_real"] or 0) > (r["v4_host_tube"] or 0) for r in don),
      "donor_real_min_over_set": min(r["donor_real"] for r in don if r["donor_real"] is not None),
      "whole_va_v5_gt_v4_same_side": sum((r["whole_va_from_origin"] or 0) > (r["v4_whole_va"] or 0) for r in don),
      "topbrain_unchanged": {"surface_same": sum(r["tb_surface_same"] for r in don), "mirror_same": sum(r["tb_mirror_same"] for r in don),
                             "R_proper": sum(abs(r["tb_R_det"] - 1) < 1e-6 for r in don),
                             "kabsch_rms_mm": q([r["tb_kabsch_rms"] for r in don], 4),
                             "radius_max_abs_diff_mm": q([r["tb_radius_diff"] for r in don], 4),
                             "len_v5_minus_v4_mm": q([r["tb_len_v5"] - r["tb_len_v4"] for r in don], 2),
                             "detail_ratio_v5_over_v4_aligned_0.3mm": q([r["tb_aligned_ratio"] for r in don], 4),
                             "detail_ratio_rel_0.15": q([r["tb_aligned_ratio_rel"] for r in don], 4),
                             "ratio_within_0.95_1.05": sum(0.95 <= (r["tb_aligned_ratio"] or 0) <= 1.05 for r in don),
                             "ratio_outside_0.9_1.1": sorted([(r["name"], r["side"], r["tb_aligned_ratio"], r["tb_aligned_v4"])
                                                              for r in don if not 0.9 <= (r["tb_aligned_ratio"] or 0) <= 1.1],
                                                             key=lambda x: x[2] or 0)},
      "v4_kept_sides": [{k: r[k] for k in ("name", "side", "v5_whole_va", "v4_whole_va", "centreline_identical_v4")} for r in kept],
      "bake_vs_check_donor_real_abs_diff": q([abs(r["donor_real"] - r["bake_donor_real"]) for r in don
                                             if r["donor_real"] is not None and r["bake_donor_real"] is not None and r["n_windows"] == 0], 4),
      "set_medians_vs_references": {
          "donor_real_vs_host_tube_0.0003": [g(q([r["donor_real"] for r in don]), "median"), REF["host_tube_0.3mm"]],
          "donor_real_vs_v4_host_tube_this_set": [g(q([r["donor_real"] for r in don]), "median"), g(q([r["v4_host_tube"] for r in don]), "median")],
          "whole_va_vs_v4_va_0.021": [g(q([r["whole_va_from_origin"] for r in don]), "median"), REF["v4_whole_va_0.3mm"]],
          "whole_va_vs_v4_va_this_set": [g(q([r["whole_va_from_origin"] for r in don]), "median"), g(q([r["v4_whole_va"] for r in don]), "median")],
          "topbrain_real_vs_topbrain_0.12": [g(q([r["topbrain_real"] for r in don]), "median"), REF["topbrain_0.3mm"]],
          "topbrain_v4_this_set": g(q([r["v4_topbrain"] for r in don]), "median"),
          "donor_real_vs_rcca_fork_v3_0.196": [g(q([r["donor_real"] for r in don]), "median"), REF["v3_rcca_fork_0.3mm"]],
          "donor_real_rel_0.15_vs_rcca_fork_rel_0.130": [g(q([r["donor_real_rel_0.15"] for r in don]), "median"), REF["ref_c_rcca_fork_rel_0.15"]],
          "rcca_fork_v5_this_set": g(it2["RCCA_v5"]["fork_frac_0.3mm"], "median"),
          "rcca_siphon_v5_this_set": g(it2["RCCA_v5"]["siphon_frac_0.3mm"], "median")},
      "lowest_donor_real": sorted([(r["name"], r["side"], r["donor_id"], r["donor_real"]) for r in don], key=lambda x: x[3] or 0)[:10],
      "per_va": rows}
it2["va"] = va
R["item2_route_and_va_detail"] = it2

# ------------------------------------------------------------------ VA lumen (extra)
if os.path.exists(os.path.join(C, "va_lumen.json")):
    L = json.load(open(os.path.join(C, "va_lumen.json")))
    lu = {}
    for t in ("v5_RVA", "v5_LVA", "v4_RVA", "v4_LVA"):
        w = [r[t]["whole"] for r in L]
        lu[t] = {"outside_pts_total": sum(x["outside"] for x in w), "not_navigable_0.65": sum(not x["navigable_0.65"] for x in w),
                 "lumen_min_mm": q([x["lumen_min_mm"] for x in w], 3), "frac_below_0.65": q([x["frac_below_0.65"] for x in w], 4)}
    for t in ("v5_RVA", "v5_LVA"):
        w = [r[t].get("donor_stretch") for r in L if r[t].get("donor_stretch")]
        lu[t + "_donor_stretch"] = {"n": len(w), "outside_pts_total": sum(x["outside"] for x in w),
                                    "not_navigable_0.65": sum(not x["navigable_0.65"] for x in w),
                                    "lumen_min_mm": q([x["lumen_min_mm"] for x in w], 3)}
    lu["v5_RVA_lumen_below_0.65"] = sorted([(r["name"], r["v5_RVA"]["whole"]["lumen_min_mm"], r["v5_RVA"]["whole"]["lumen_min_at_mm"])
                                           for r in L if not r["v5_RVA"]["whole"]["navigable_0.65"]], key=lambda x: x[1])[:30]
    R["va_lumen_extra"] = lu

# ------------------------------------------------------------------ item 3
log = open(os.path.join(C, "logs", "split.log")).read()
R["item3_split"] = {"exit": [ln for ln in log.splitlines() if ln.startswith("EXIT")], "not_lowered": "not lowered vs baseline" in log,
                    "nesting": [ln.strip() for ln in log.splitlines() if "one_to_one_nested" in ln],
                    "summary_lines": [ln.strip() for ln in log.splitlines() if "roster_v3:" in ln or " max k=" in ln],
                    "result_file": "/sp/v5_va/results/split_v5_baked.json", "baseline": "/sp/v5_va/results/split_v4.json"}

# ------------------------------------------------------------------ item 4
ld = json.load(open(os.path.join(C, "loader_v5.json")))
v4l = json.load(open("/sp/v4_build/audit/sim_loader.json"))
new_prob = {n: [p for p in r["problems"] if p not in set(v4l["per_anatomy"][n]["problems"])] for n, r in ld["per_anatomy"].items()}
pa = ld["per_anatomy"]
R["item4a_loader"] = {"summary": {k: v for k, v in ld["summary"].items() if k != "bad"},
                      "n_18_branches": sum(r["n_branches"] == 18 for r in pa.values()),
                      "insertion_identical_host": sum(r["insertion_dpos"] < 1e-6 for r in pa.values()),
                      "insertion_inside": sum(r["insertion_inside"] for r in pa.values()),
                      "centreline_outside_anatomies": sum(bool(r["centerline_outside"]) for r in pa.values()),
                      "rcca_byte_identical_v3": sum(r["rcca_byte_identical_v3"] for r in pa.values()),
                      "nonva_byte_identical_v4": sum(r["nonva_identical_v4"] for r in pa.values()),
                      "va_files_replaced": {"RVA": sum(not r["va_file_identical_v4"]["RVA"] for r in pa.values()),
                                            "LVA": sum(not r["va_file_identical_v4"]["LVA"] for r in pa.values())},
                      "targets_min": min(r["n_targets"] for r in pa.values()),
                      "problems_not_in_v4_audit": {n: p for n, p in new_prob.items() if p},
                      "problems_shared_with_v4_audit": "host-numbered branches (0),(2)-(7),(11),(14),(18): point gaps 5.8-19.1 mm and one duplicate point, identical in v4's audit (host data, unchanged)"}
sel = json.load(open(os.path.join(C, "sofa_selection.json")))
S = {}
for n in sel["most_changed_15"] + sel["random_10"]:
    row = {"group": "most_changed" if n in sel["most_changed_15"] else "random", "why": sel["why"].get(n)}
    for v in ("v5", "v4"):
        p = os.path.join(C, "sofa", "run4", v, n + ".json")
        if not os.path.exists(p):
            row[v] = None
            continue
        r = json.load(open(p))
        row[v] = {k: r.get(k) for k in ("passed", "rva_exercised", "exception", "branches", "loader_mesh_is_this_obj", "insertion_inside",
                                        "nonfinite_steps", "steps", "ms_per_step", "reset_s", "max_rva_s_mm", "donor_traversed_mm",
                                        "reached_topbrain", "fork_attempts", "final_tip", "tip_outside_steps", "tip_outside_max_mm",
                                        "tip_outside_gt_0.5mm_steps", "dev_pts_checked", "dev_pts_outside", "dev_out_gt_0.3mm_pts",
                                        "dev_outside_max_mm", "dev_out_worst", "dev_out_gt_0.3mm_by_branch", "target_vessel_frame", "tip_out_worst", "tip_out_sites", "unbuckles",
                                        "steps_in_donor_stretch", "first_step_in_donor")}
        row[v]["rva"] = r.get("rva")
        o_ = (r.get("rva") or {}).get("ostium_mm") if v == "v5" else (r.get("rva") or {}).get("s_17_takeoff_mm")
        row[v]["entered_va_20mm_past_ostium"] = bool(o_ is not None and (r.get("max_rva_s_mm") or -1) >= o_ + 20.0)
        row[v]["episodes"] = r.get("episodes"); row[v]["exercised_in_episode"] = r.get("exercised_in_episode")
    S[n] = row
done5 = [S[n]["v5"] for n in S if S[n]["v5"]]
done4 = [S[n]["v4"] for n in S if S[n]["v4"]]
R["item4b_sofa"] = {
    "selection_rule": sel["rule"], "n": len(S), "done_v5": len(done5), "done_v4_control": len(done4),
    "v5_passed": sum(bool(x["passed"]) for x in done5), "v4_passed": sum(bool(x["passed"]) for x in done4),
    "v5_rva_exercised": sum(bool(x["rva_exercised"]) for x in done5), "v4_rva_exercised": sum(bool(x["rva_exercised"]) for x in done4),
    "v5_reached_topbrain": sum(bool(x["reached_topbrain"]) for x in done5), "v4_reached_topbrain": sum(bool(x["reached_topbrain"]) for x in done4),
    "v5_entered_va_20mm_past_ostium": sum(x["entered_va_20mm_past_ostium"] for x in done5), "v4_entered_va_20mm_past_ostium": sum(x["entered_va_20mm_past_ostium"] for x in done4),
    "v5_episodes": sum(x["episodes"] or 0 for x in done5), "v4_episodes": sum(x["episodes"] or 0 for x in done4),
    "v5_steps_total": sum(x["steps"] or 0 for x in done5), "v4_steps_total": sum(x["steps"] or 0 for x in done4),
    "v5_donor_traversed_mm": q([x["donor_traversed_mm"] for x in done5], 1), "v4_rva_max_s_mm": q([x["max_rva_s_mm"] for x in done4], 1),
    "v5_nonfinite_steps": sum(x["nonfinite_steps"] or 0 for x in done5), "v5_exceptions": sum(x["exception"] is not None for x in done5),
    "v5_tip_outside_steps": sum(x["tip_outside_steps"] or 0 for x in done5), "v4_tip_outside_steps": sum(x["tip_outside_steps"] or 0 for x in done4),
    "v5_tip_outside_max_mm": max([x["tip_outside_max_mm"] or 0 for x in done5] or [0]),
    "v4_tip_outside_max_mm": max([x["tip_outside_max_mm"] or 0 for x in done4] or [0]),
    "v5_dev_out_gt_0.3mm_frac": round(sum(x["dev_out_gt_0.3mm_pts"] or 0 for x in done5) / max(sum(x["dev_pts_checked"] or 0 for x in done5), 1), 5),
    "v4_dev_out_gt_0.3mm_frac": round(sum(x["dev_out_gt_0.3mm_pts"] or 0 for x in done4) / max(sum(x["dev_pts_checked"] or 0 for x in done4), 1), 5),
    "v5_dev_outside_max_mm": q([x["dev_outside_max_mm"] for x in done5], 3), "v4_dev_outside_max_mm": q([x["dev_outside_max_mm"] for x in done4], 3),
    "v5_ms_per_step": q([x["ms_per_step"] for x in done5], 1), "v4_ms_per_step": q([x["ms_per_step"] for x in done4], 1),
    "target_on_rcca_vessel_frame_max_dist_mm": max([g(x, "target_vessel_frame", "dist_to_rcca_mm") or 0 for x in done5] or [0]),
    "target_rcca_s_min_mm": min([g(x, "target_vessel_frame", "rcca_s_mm") or 1e9 for x in done5] or [0]),
    "per_anatomy": S}

# ------------------------------------------------------------------ item 5
R["item5_v4_untouched"] = json.load(open(os.path.join(C, "item5_v4_untouched.json")))
p5 = os.path.join(C, "item5_v4_untouched_end.json")
if os.path.exists(p5):
    R["item5_v4_untouched_end"] = json.load(open(p5))

json.dump(R, open("/sp/v5_va/results/v5_checks.json", "w"), indent=1, default=float)
pr = {k: v for k, v in it1.items() if k not in ("failing_detail",)}
print("ITEM1", json.dumps(pr))
print("ITEM2", json.dumps({k: v for k, v in it2.items() if k not in ("va", "identical_to_v4_names_not", "ratio_0.3mm_below_0.99")}))
print("ITEM2 low", it2["ratio_0.3mm_below_0.99"])
print("VA", json.dumps({k: v for k, v in va.items() if k not in ("per_va",)}))
print("LUMEN", json.dumps(R.get("va_lumen_extra")))
print("ITEM3", json.dumps(R["item3_split"]))
print("ITEM4a", json.dumps({k: v for k, v in R["item4a_loader"].items()}))
print("ITEM4b", json.dumps({k: v for k, v in R["item4b_sofa"].items() if k not in ("per_anatomy", "selection_rule")}))
for n, row in S.items():
    a, b = row["v5"], row["v4"]
    f = lambda x: None if not x else (x["passed"], x["rva_exercised"], x["max_rva_s_mm"], x["donor_traversed_mm"], x["fork_attempts"],
                                      x["tip_outside_steps"], round(x["dev_outside_max_mm"] or 0, 2), x["dev_out_gt_0.3mm_pts"])
    print("SOFA %-36s %-12s v5 %s | v4 %s" % (n[:36], row["group"], f(a), f(b)))

# extra diagnostics written by entry_geom.py / tb_geom.py / bake_tb_a2.json (no recomputation here)
R2 = json.load(open("/sp/v5_va/results/v5_checks.json"))
eg = json.load(open(os.path.join(C, "entry_geom.json")))
R2["item4b_sofa"]["entry_geometry_all_223"] = {
    t: {k: q([r[t][k] for r in eg], 1) for k in ("turn_parent_to_va_deg", "va_vs_17_deg", "stub_mm")} for t in ("v5", "v4")}
tg = json.load(open(os.path.join(C, "tb_geom.json")))
R2["item2_route_and_va_detail"]["va"]["topbrain_geometry"] = {
    lab: {"kabsch_rms_mm": q([r[lab]["kabsch_rms_mm"] for r in tg], 4), "radius_max_abs_diff_mm": q([r[lab]["radius_max_abs_diff_mm"] for r in tg], 4)}
    for lab in ("common", "core_minus_10mm")}
R2["item2_route_and_va_detail"]["va"]["topbrain_geometry"]["radius_median_abs_diff_mm_max"] = max(r["common"]["radius_median_abs_diff_mm"] for r in tg)
a2 = json.load(open(os.path.join(C, "bake_tb_a2.json")))
R2["item2_route_and_va_detail"]["va"]["topbrain_in_situ_over_alone_bake_A2"] = q([v[0] for v in a2.values()], 3)
json.dump(R2, open("/sp/v5_va/results/v5_checks.json", "w"), indent=1, default=float)
print("EXTRA", json.dumps({"entry": R2["item4b_sofa"]["entry_geometry_all_223"], "tbgeom": R2["item2_route_and_va_detail"]["va"]["topbrain_geometry"],
                           "tbA2": R2["item2_route_and_va_detail"]["va"]["topbrain_in_situ_over_alone_bake_A2"]}))
