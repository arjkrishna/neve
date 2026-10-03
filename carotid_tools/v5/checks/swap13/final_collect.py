#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\final_collect.py (= /sp/v5_va/swap13/final_collect.py in the build containers)
"""v5_swap13 step 4: the final record of the 223 v5 anatomies after the route13 swap.

    python3 final_collect.py        (neve-build-meshlab; /repo READ-ONLY; writes only under /sp)
Writes
  /sp/v5_va/swap13/build/BUILD_v5.json        -> copied by the host into carotid_data/anatomies_v5/BUILD_v5.json
  /sp/v5_va/results/v5_bake_full_final.json   (bake_full/collect.py's collection over the final 223)
  /sp/v5_va/results/v5_final_summary.json     (every final number the record quotes)

BUILD_v5.json = bake_full/collect.py's build_v5() UNCHANGED (run on the final markers: the 13 swapped anatomies carry
their route13 bake / measure markers), plus stated additions: route13 (the 13 moves), v4_va_kept_sides (15 at
assignment + 2 at bake), donor_flags, licences (NC / VMR counts, attributions with anatomy counts), checks_final
(key numbers + pointer to v5_final_summary.json); grafter / timings / records lines extended for route13.
Check inputs: the 13 swapped anatomies from swap13/ (chk_out, vvw_out, sofa, lumen13, va_lumen13), the other 210 from
the check stage (checks/out, vvw/out, checks/va_lumen.json, checks/sofa/run4) -- their folders are unchanged since
those checks ran (no file newer than 2026-10-02 20:30); route vs v4 (exact + k24) for all 223 from swap13/rrdx;
split results/split_v5_final.json; loader swap13/loader_v5.json (all 223, after the swap).
"""
import collections
import glob
import json
import os
import sys
import time

import numpy as np

SP = "/sp/v5_va"
W = SP + "/swap13"
A5 = "/repo/carotid_data/anatomies_v5"
A4 = "/repo/carotid_data/anatomies_v4"
BOUT = W + "/build"
SUMMARY = SP + "/results/v5_final_summary.json"
BFOUT = SP + "/results/v5_bake_full_final.json"
NOW = time.strftime("%Y-%m-%d %H:%M:%S")
# FC_TEST=1: code-path test while rrdx is still running -- missing rrdx records are stood in by w_050's (numbers
# meaningless), every output goes to swap13/build_test/ (nothing in results/ or build/ is touched)
TEST = bool(os.environ.get("FC_TEST"))
if TEST:
    BOUT = W + "/build_test"
    SUMMARY = BOUT + "/v5_final_summary_TEST.json"
    BFOUT = BOUT + "/v5_bake_full_final_TEST.json"
    os.makedirs(BOUT, exist_ok=True)


def q(x, nd=4):
    x = [v for v in x if v is not None]
    if not x:
        return None
    a = np.asarray(x, float)
    return {"n": int(len(a)), "min": round(float(a.min()), nd), "p10": round(float(np.percentile(a, 10)), nd),
            "median": round(float(np.median(a)), nd), "p90": round(float(np.percentile(a, 90)), nd),
            "max": round(float(a.max()), nd)}


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load(p):
    return json.load(open(p, encoding="utf-8"))


names = sorted(n for n in os.listdir(A5) if n.startswith("case_"))
assert len(names) == 223, len(names)
picks = dict(ln.split("\t") for ln in open(W + "/picks.tsv").read().splitlines() if ln.strip())
swap_marks = {n: open(W + "/done/%s.swap" % n).read().strip() for n in picks if os.path.exists(W + "/done/%s.swap" % n)}
assert len(swap_marks) == 13, "swap markers missing"
prov = {n: load(os.path.join(A5, n, "provenance.json")) for n in names}
R13 = load(SP + "/results/v5_route13.json")["anatomies"]

# ------------------------------------------------------------------ 1. bake collection + BUILD_v5.json (collect.py unchanged)
sys.path.insert(0, SP + "/bake_full")
import collect as BC  # noqa: E402
BC.OUT = BFOUT
cap = {}
_real_build = BC.build_v5
BC.build_v5 = lambda rows, S, kept: cap.update(rows=rows, S=S, kept=kept)
sys.argv = ["collect.py", "--build"]
rc = BC.main()
assert rc == 0 and cap, "collect.py stopped before the build (not every anatomy baked / measured / passing)"
os.makedirs(BOUT, exist_ok=True)
BC.A5 = BOUT                     # build_v5 uses A5 only for its output path
_real_build(cap["rows"], cap["S"], cap["kept"])
BC.A5 = A5
build = load(BOUT + "/BUILD_v5.json")
rows, S, kept = cap["rows"], cap["S"], cap["kept"]
brow = {r["name"]: r for r in rows}

# ------------------------------------------------------------------ 2. strict mesh audit (chk_one item 1)
def chk(n):
    return load((W + "/chk_out/%s.json" if n in picks else SP + "/checks/out/%s.json") % n)


CH = {n: chk(n) for n in names}
G = {n: CH[n]["item1"]["gates"] for n in names}
keys_bool = ["obj_clean_raw", "obj_clean_welded", "full_clean_raw", "full_clean_welded", "union_clean_welded",
             "obj_pml_loader_clean", "numpy_obj_ok", "numpy_full_ok", "obj_budget_ok", "full_budget_ok", "obj_file_clean",
             "full_file_all_tri", "routes_navigable_obj", "routes_navigable_full", "orientation_as_v4", "strict_pass"]
audit = {
    "tool": "checks/chk_one.py item 1 (v4's topo_audit.py unchanged): 210 from the check stage (checks/out), the 13 "
            "swapped re-run after the swap (swap13/chk_out)",
    "pass_counts": {k: sum(bool(G[n][k]) for n in names) for k in keys_bool},
    "swapped_13_strict_pass": sum(bool(G[n]["strict_pass"]) for n in picks),
    "self_intersecting_faces_total": {m: sum((G[n]["selfint"][m] or 0) for n in names) for m in ("obj", "full", "union")},
    "self_intersecting_faces_welded_total": {m: sum((G[n]["selfint"][m + "_welded"] or 0) for n in names) for m in ("obj", "full", "union")},
    "obj_tris": q([G[n]["obj_tris"] for n in names], 0), "full_tris": q([G[n]["full_tris"] for n in names], 0),
    "union_tris": q([G[n]["union_tris"] for n in names], 0),
    "route_lumen_min_obj_mm": {k: q([G[n]["route_lumen_min_obj"][k] for n in names], 3) for k in ("RCCA", "LCCA")},
    "centreline_points_total": sum(G[n]["centreline_points"] for n in names),
    "centreline_outside_obj_total": sum(G[n]["centreline_outside_obj_all"] for n in names),
    "centreline_outside_full_total": sum(G[n]["centreline_outside_full_all"] for n in names),
    "obj_negative_signed_volume_as_v4": sum(G[n]["signed_volume_mm3"]["obj"] < 0 for n in names),
    "full_to_union_p99_mm": q([CH[n]["item1"]["full_to_union_mm"]["p99"] for n in names], 4),
    "obj_to_union_p99_mm": q([CH[n]["item1"]["obj_to_union_mm"]["p99"] for n in names], 4),
    "failing": sorted(n for n in names if not G[n]["strict_pass"]),
    "swapped_13": {n: {"strict_pass": G[n]["strict_pass"], "obj_tris": G[n]["obj_tris"], "full_tris": G[n]["full_tris"],
                       "selfint": G[n]["selfint"], "route_lumen_min_obj_mm": G[n]["route_lumen_min_obj"],
                       "centreline_outside": [G[n]["centreline_outside_obj_all"], G[n]["centreline_outside_full_all"]]}
                   for n in picks}}

# ------------------------------------------------------------------ 3. right route vs v4 (rrdx: exact + k24, all 223)
def rx(n):
    p = W + "/rrdx/%s.json" % n
    if TEST and (not os.path.exists(p) or "rrdx_version" not in open(p).read()):
        p = W + "/rrdx/case_w_050_left__topcow_mr_004_L.json"
    r = load(p)
    assert r.get("rrdx_version") == 2, "rrdx output without the tolerant match (v1): %s" % n
    return r


RX = {n: rx(n) for n in names}


def route_block(mode):
    rat = {n: RX[n][mode]["ratio"] for n in names}
    cls = [RX[n][mode]["identical_beyond_mm_tol"] for n in names]
    cls_s = [RX[n][mode]["identical_beyond_mm"] for n in names]
    low = sorted([(n, seg, round(rat[n][seg]["frac_0.3mm"], 4)) for n in names for seg in ("fork", "siphon")
                  if rat[n][seg]["frac_0.3mm"] is not None and rat[n][seg]["frac_0.3mm"] < 0.99], key=lambda x: x[2])
    mn = min(((rat[n][s]["frac_0.3mm"], n, s) for n in names for s in ("fork", "siphon") if rat[n][s]["frac_0.3mm"]))
    return {
        "identical_cells_everywhere": cls.count(0),
        "identical_cells_beyond_mm": {"4": cls.count(4), "6": cls.count(6), "8": cls.count(8), "not_even_8": cls.count(None)},
        "not_even_8mm": sorted(n for n in names if RX[n][mode]["identical_beyond_mm_tol"] is None),
        "strict_rounded_cells": {"everywhere": cls_s.count(0), "4": cls_s.count(4), "6": cls_s.count(6), "8": cls_s.count(8),
                                 "not_even_8": cls_s.count(None)},
        "differing_cells_tol_everywhere": q([RX[n][mode]["cell_diff_tol"]["all_fork"]["only_v5"] + RX[n][mode]["cell_diff_tol"]["all_siphon"]["only_v5"]
                                             for n in names if RX[n][mode]["identical_beyond_mm_tol"] != 0], 0),
        "equal_at_4dp": sum(bool(RX[n][mode]["equal_4dp"]) for n in names),
        "ratio_fork_0.3mm": q([rat[n]["fork"]["frac_0.3mm"] for n in names], 4),
        "ratio_siphon_0.3mm": q([rat[n]["siphon"]["frac_0.3mm"] for n in names], 4),
        "ratio_fork_0.5mm": q([rat[n]["fork"]["frac_0.5mm"] for n in names], 4),
        "ratio_siphon_0.5mm": q([rat[n]["siphon"]["frac_0.5mm"] for n in names], 4),
        "per_segment_ge_0.99_ge_0.95_mean": {
            "%s_%s" % (s, t): [sum((rat[n][s][t] or 0) >= 0.99 for n in names), sum((rat[n][s][t] or 0) >= 0.95 for n in names),
                               round(float(np.mean([rat[n][s][t] for n in names if rat[n][s][t] is not None])), 4)]
            for s in ("fork", "siphon") for t in ("frac_0.3mm", "frac_0.5mm")},
        "anatomies_below_0.99": len({x[0] for x in low}), "below_0.99": low,
        "anatomies_below_0.99_at_0.5mm": sorted({n for n in names for s in ("fork", "siphon")
                                                 if rat[n][s]["frac_0.5mm"] is not None and rat[n][s]["frac_0.5mm"] < 0.99}),
        "anatomies_below_0.995": len({n for n in names for s in ("fork", "siphon") if (rat[n][s]["frac_0.3mm"] or 1) < 0.995}),
        "anatomies_above_1.01": len({n for n in names for s in ("fork", "siphon") if (rat[n][s]["frac_0.3mm"] or 1) > 1.01}),
        "lowest": [round(mn[0], 4), mn[1], mn[2]],
        "rcca_v5_median_fork_siphon_0.3mm": [round(float(np.median([RX[n][mode]["rrd_v5"]["fork"]["frac_0.3mm"] for n in names])), 4),
                                             round(float(np.median([RX[n][mode]["rrd_v5"]["siphon"]["frac_0.3mm"] for n in names])), 4)],
        "rcca_v4_median_fork_siphon_0.3mm": [round(float(np.median([RX[n][mode]["rrd_v4"]["fork"]["frac_0.3mm"] for n in names])), 4),
                                             round(float(np.median([RX[n][mode]["rrd_v4"]["siphon"]["frac_0.3mm"] for n in names])), 4)],
        "swapped_13": {n: {"fork": round(rat[n]["fork"]["frac_0.3mm"], 4), "siphon": round(rat[n]["siphon"]["frac_0.3mm"], 4),
                           "fork_0.5": round(rat[n]["fork"]["frac_0.5mm"], 4), "siphon_0.5": round(rat[n]["siphon"]["frac_0.5mm"], 4),
                           "class": RX[n][mode]["identical_beyond_mm_tol"], "class_strict": RX[n][mode]["identical_beyond_mm"],
                           "differing_cells_tol": {k: v["only_v5"] + v["only_v4"] for k, v in RX[n][mode]["cell_diff_tol"].items()
                                                   if k.startswith("all_")}} for n in picks}}


def below_diag(mode):
    """swap13/diag_lost.py records (where run) for the anatomies below 0.99 in this mode"""
    out = {}
    for n, seg, v in route_block(mode)["below_0.99"]:
        p = W + "/diag/%s.json" % n
        if os.path.exists(p):
            d = load(p)
            out[n] = {"segment": seg, "ratio_0.3mm": v, "tool": "swap13/diag_lost.py", "lost_cells": d["lost_cells"],
                      "lost_cells_v4_depth_gt_0.3": d["lost_cells_v4_depth_gt_0.3"],
                      "lost_area_v4_depth_gt_0.3_mm2": d["lost_area_v4_depth_gt_0.3_mm2"],
                      "lost_v4_owner_k24": d["lost_v4_owner_k24"], "lost_v4_depth_mm [min, median, max]": d["lost_v4_depth_mm"],
                      "lost_present_geometrically_in_v5": d["lost_present_geometrically_in_v5"],
                      "lost_absent_nearest_v5_cell_mm": d["lost_absent_nearest_v5_cell_mm"],
                      "lost_rcca_s_mm_v4": d["lost_rcca_s_mm_v4"],
                      "lost_dist_to_v5_va_sample_mm": d["lost_dist_to_v5_va_sample_mm"],
                      "lost_dist_to_v4_va_sample_mm": d["lost_dist_to_v4_va_sample_mm"],
                      "gained_cells_v5_depth_gt_0.3": d["gained_cells_v5_depth_gt_0.3"]}
    return out


bake_cls = {n: brow[n]["right_route_detail"]["identical_beyond_mm"] for n in names}
bake_cls = {n: (0 if brow[n]["right_route_detail"]["identical"] else bake_cls[n]) for n in names}
route = {
    "definition": "right_route_detail (bake_meshes_v4): area fraction of the RCCA-owned wall past the host cut standing > 0.3 "
                  "(0.5) mm off the plain tube, fork (RCCA s < 130 mm) / siphon (s >= 130), on the UNDECIMATED unions (v5: "
                  "iso_v5 = the accepted bake; v4: rebuilt from mesh_v4.json with the repo baker). Ownership EXACT = argmax "
                  "of r - |x - c| over ALL centreline samples (the study's convention); k24 = v4's coded 24-nearest cut-off. "
                  "Classes: every RCCA route cell matches a v4 cell (centroid and depth within 1e-3 mm, area within 1e-3 "
                  "relative; verify_v5_route13's tolerant match) everywhere, else beyond 4 / 6 / 8 mm of every v4 / v5 VA "
                  "sample; 'strict_rounded_cells' = the same with exact equality after rounding (centroid 1e-4 mm, depth "
                  "1e-5 mm, area 1e-6 mm2), which also counts um-level float differences.",
    "tool": "swap13/rrdx.py (vvw/measure.py's Own / rrd + vvw/rrd_v4.rebuild_v4, unchanged), all 223 after the swap",
    "validation": {"v4_rebuild_k24_equals_mesh_v4_record": sum(bool(RX[n]["k24_v4_rebuild_equals_mesh_v4"]) for n in names),
                   "v5_k24_equals_mesh_v5_record": sum(bool(RX[n]["k24_v5_equals_mesh_v5"]) for n in names),
                   "k24_tol_class_equals_bake_record_class": sum(RX[n]["k24"]["identical_beyond_mm_tol"] == bake_cls[n] for n in names),
                   "k24_tol_class_disagreements_with_bake_record [tol, bake]": {
                       n: [RX[n]["k24"]["identical_beyond_mm_tol"], bake_cls[n]] for n in names
                       if RX[n]["k24"]["identical_beyond_mm_tol"] != bake_cls[n]},
                   "bake_record_class_definition": "bake_full/measure_full_one.py: k24 right_route_detail rounded to 4 dp "
                                                   "equal to mesh_v4.json (everywhere), else the RCCA-owned wall beyond D mm "
                                                   "of every VA equal (diag_route)"},
    "exact": dict(route_block("exact"), below_0_99_diagnosis=below_diag("exact")),
    "k24": dict(route_block("k24"), below_0_99_diagnosis=below_diag("k24")),
    "bake_record_k24_4dp": {"identical_to_v4": S["right_route_detail"]["identical_to_v4"],
                            "identical_beyond_mm": S["right_route_detail"]["identical_beyond_mm"],
                            "ratio_fork_0.3mm": S["right_route_detail"]["ratio_fork_0.3mm"],
                            "ratio_siphon_0.3mm": S["right_route_detail"]["ratio_siphon_0.3mm"]},
    "chk_one_k24_swapped_13": {n: {"identical_v4": CH[n]["item2"]["right_route_detail"]["identical_v4"],
                                   "ratio_v5_v4": CH[n]["item2"]["right_route_detail"]["ratio_v5_v4"],
                                   "recomputed_equals_mesh_v5": CH[n]["item2"]["right_route_detail"]["recomputed_equals_v5_record"],
                                   "identical_beyond_mm": CH[n]["item2"]["identical_beyond_mm"],
                                   "v4_rebuild_max_abs_diff": CH[n]["item2"]["v4_rebuild"]["max_abs_diff_vs_mesh_v4"]} for n in picks}}

# ------------------------------------------------------------------ 4. VA wall detail (vvw/measure.py: exact, seam = 0)
def vvw(n):
    return load((W + "/vvw_out/%s.json" if n in picks else SP + "/vvw/out/%s.json") % n)


VV = {n: vvw(n) for n in names}
vrows = []
for n in names:
    mv = load("/repo/carotid_data/va_sources/v5_reg/bake_full/_done/%s.measure.json" % n)["va_detail"]
    for side in ("right", "left"):
        e, k = VV[n]["exact"]["va"][side], VV[n]["k24"]["va"][side]
        if not e.get("donor"):
            continue
        c4 = g(CH[n], "item2", "va", side, "v4")
        f = lambda d, key="frac_0.3mm": None if d is None else d.get(key)  # noqa: E731
        vrows.append({"name": n, "side": side, "donor": e["donor"], "swapped": n in picks,
                      "n_windows": len(e["windows"]), "tubed_mm": round(sum(b - a for a, b in e["windows"]), 2),
                      "donor_real": f(e["donor_real"]), "donor_real_0.5": f(e["donor_real"], "frac_0.5mm"),
                      "donor_real_rel": f(e["donor_real"], "frac_rel_0.15"), "donor_real_fit": f(e["donor_real"], "fit_frac_0.3mm"),
                      "donor_real_k24": f(k["donor_real"]), "seam_zone": f(e["seam_zone"]), "windows_only": f(e["windows_only"]),
                      "topbrain": f(e["topbrain"]), "whole_cons": f(e["whole_ostium_cons"]), "whole_ostium_raw": f(e["whole_ostium_raw"]),
                      "whole_origin_raw_k24": f(k["whole_origin_raw"]), "junction_stub": f(e["junction_stub"]),
                      "obj_donor_real": f(g(VV[n], "obj", "cent", side, "donor_real")),
                      "v4_host_core_record": g(mv, side, "v4_host_core_frac_0.3mm"),
                      "v4_whole_va_k24": f(g(c4, "whole_va")) if c4 else None,
                      "A2": g(mv, side, "A2_detail_kept_in_situ_over_alone")})
col = lambda key, rr=None: [r[key] for r in (rr or vrows)]  # noqa: E731
obj_keep = [r["obj_donor_real"] / r["donor_real"] for r in vrows if r["donor_real"] and r["obj_donor_real"] is not None]
wall = {
    "tool": "vvw/measure.py (verify_v5_wall's own implementation; EXACT ownership; seam zone = the 25 mm calibre ramp on the "
            "donor side, reported apart and counted 0 in the whole VA; core trims 2 mm each end) on the undecimated unions: "
            "210 from vvw/out, the 13 swapped re-run (swap13/vvw_out)",
    "donor_vas": len(vrows), "donor_sides": len({r["donor"] for r in vrows}), "donor_units": len({r["donor"].split()[0] for r in vrows}),
    "donor_real_frac_0.3mm": q(col("donor_real")), "donor_real_frac_0.5mm": q(col("donor_real_0.5")),
    "donor_real_frac_rel_0.15r": q(col("donor_real_rel")), "donor_real_vs_fitted_tube_0.3mm": q(col("donor_real_fit")),
    "donor_real_k24_frac_0.3mm": q(col("donor_real_k24")),
    "seam_zone_frac_0.3mm": q(col("seam_zone")), "windows_only_frac_0.3mm": q(col("windows_only")),
    "topbrain_frac_0.3mm": q(col("topbrain")), "junction_stub_frac_0.3mm": q(col("junction_stub")),
    "whole_va_ostium_to_end_seam0_frac_0.3mm": q(col("whole_cons")),
    "whole_va_ostium_raw_frac_0.3mm": q(col("whole_ostium_raw")),
    "whole_va_origin_raw_k24_frac_0.3mm (check-stage convention)": q(col("whole_origin_raw_k24")),
    "v4_host_core_frac_0.3mm_record": q(col("v4_host_core_record")),
    "v4_whole_va_frac_0.3mm_k24": q(col("v4_whole_va_k24")),
    "donor_real_gt_v4_host_core_same_side": sum((r["donor_real"] or 0) > (r["v4_host_core_record"] or 0) for r in vrows),
    "whole_va_seam0_gt_v4_whole_va_same_side": sum((r["whole_cons"] or 0) > (r["v4_whole_va_k24"] or 0) for r in vrows
                                                   if r["v4_whole_va_k24"] is not None),
    "whole_va_compared": sum(r["v4_whole_va_k24"] is not None for r in vrows),
    "obj_relief_kept_over_union": q(obj_keep, 3),
    "detail_kept_in_situ_over_alone_A2_record": q(col("A2"), 3),
    "windows": {"vas_with_window": sum(r["n_windows"] > 0 for r in vrows),
                "windows_per_va": dict(collections.Counter(r["n_windows"] for r in vrows)),
                "tubed_mm": q([r["tubed_mm"] for r in vrows if r["n_windows"]], 2)},
    "lowest_donor_real": sorted([(r["name"], r["side"], r["donor"], round(r["donor_real"], 4)) for r in vrows],
                                key=lambda x: x[3])[:5],
    "swapped_13_moved_vas": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in vrows
                             if r["swapped"] and r["side"] in (g(prov[r["name"]], "v5_va", "route13", "targets") or [])]}

# ------------------------------------------------------------------ 5. lumens
L13 = load(W + "/lumen13.json")
VL_old = {r["name"]: r for r in load(SP + "/checks/va_lumen.json")}
VL13 = load(W + "/va_lumen13.json")
VL = {n: (VL13[n]["now"] if n in picks else VL_old[n]) for n in names}
narrow = sorted([(n, t[3:], VL[n][t]["whole"]["lumen_min_mm"]) for n in names for t in ("v5_RVA", "v5_LVA")
                 if not VL[n][t]["whole"]["navigable_0.65"]], key=lambda x: x[2])
lumen = {
    "route_lumen_undecimated_swapped_13": {
        "tool": "swap13/lumen13.py (bake_meshes_v4.route_report on the undecimated unions)",
        "now_equals_before_swap": sum(bool(L13[n]["now_equals_before"]) for n in picks),
        "now_equals_v4": sum(bool(L13[n]["now_equals_v4"]) for n in picks),
        "rcca_min_mm": q([L13[n]["now"]["RCCA"]["lumen_min_mm"] for n in picks], 3),
        "lcca_min_mm": q([L13[n]["now"]["LCCA"]["lumen_min_mm"] for n in picks], 3),
        "navigable": sum(L13[n]["now"]["RCCA"]["navigable"] and L13[n]["now"]["LCCA"]["navigable"] for n in picks)},
    "route_lumen_min_obj_mm_223": audit["route_lumen_min_obj_mm"],
    "va_lumen_obj": {
        "tool": "checks/va_lumen.py one() (210 check stage + 13 swapped re-run), not a v4 gate",
        "outside_pts_total": sum(VL[n][t]["whole"]["outside"] for n in names for t in ("v5_RVA", "v5_LVA")),
        "vas_below_0.65mm": narrow,
        "lumen_min_mm_v5": q([VL[n][t]["whole"]["lumen_min_mm"] for n in names for t in ("v5_RVA", "v5_LVA")], 3),
        "lumen_min_mm_v4": q([VL[n][t]["whole"]["lumen_min_mm"] for n in names for t in ("v4_RVA", "v4_LVA")], 3),
        "swapped_13_before_vs_now": {n: {t: [VL13[n]["before_swap"][t]["whole"]["lumen_min_mm"], VL13[n]["now"][t]["whole"]["lumen_min_mm"]]
                                         for t in ("v5_RVA", "v5_LVA")} for n in picks}}}

# ------------------------------------------------------------------ 6. split
SPL = load(SP + "/results/split_v5_final.json")
slog = open(W + "/logs/split.log").read()
split = {"command": "python3 v5_split.py /repo/carotid_data/anatomies_v5 --baseline results/split_v4.json --out results/split_v5_final.json",
         "exit": [ln for ln in slog.splitlines() if ln.startswith("EXIT")],
         "not_lowered": "not lowered vs baseline" in slog, "baseline_compared": g(SPL, "baseline", "compared"),
         "baseline_lowered": g(SPL, "baseline", "lowered"),
         "summary_lines": [ln.strip() for ln in slog.splitlines() if "roster_v3:" in ln or " max k=1" in ln],
         "nesting_lines": [ln.strip() for ln in slog.splitlines() if "one_to_one_nested" in ln or "V:" in ln][:12],
         "result": "v5_va/results/split_v5_final.json", "log": "v5_va/swap13/logs/split.log"}

# ------------------------------------------------------------------ 7. loader (all 223, after the swap)
LD = load(W + "/loader_v5.json")
pa = LD["per_anatomy"]
v4l = load("/sp/v4_build/audit/sim_loader.json")
new_prob = {n: [p for p in r["problems"] if p not in set(v4l["per_anatomy"][n]["problems"])] for n, r in pa.items()}
loader = {"tool": "checks/loader_v5.py (v4 audit sim_loader.py on anatomies_v5; output path only changed), all 223 after the swap",
          "anatomies": len(pa), "n_18_branches": sum(r["n_branches"] == 18 for r in pa.values()),
          "insertion_identical_host": sum(r["insertion_dpos"] < 1e-6 for r in pa.values()),
          "insertion_inside": sum(r["insertion_inside"] for r in pa.values()),
          "insertion_wall_mm": LD["summary"]["insertion_wall_mm_minmax"],
          "centreline_outside_anatomies": sum(bool(r["centerline_outside"]) for r in pa.values()),
          "centreline_pts_total": LD["summary"]["centerline_pts_total"],
          "rcca_byte_identical_v3": sum(r["rcca_byte_identical_v3"] for r in pa.values()),
          "nonva_byte_identical_v4": sum(r["nonva_identical_v4"] for r in pa.values()),
          "va_files_replaced": {"RVA": sum(not r["va_file_identical_v4"]["RVA"] for r in pa.values()),
                                "LVA": sum(not r["va_file_identical_v4"]["LVA"] for r in pa.values())},
          "targets_min": min(r["n_targets"] for r in pa.values()),
          "problems_not_in_v4_audit": {n: p for n, p in new_prob.items() if p}}

# ------------------------------------------------------------------ 8. SOFA
def sofa_row(p, v):
    if not os.path.exists(p):
        return None
    r = load(p)
    o_ = g(r, "rva", "ostium_mm") if v == "v5" else g(r, "rva", "s_17_takeoff_mm")
    return {"passed": r.get("passed"), "rva_exercised": r.get("rva_exercised"), "exception": r.get("exception"),
            "branches": r.get("branches"), "nonfinite_steps": r.get("nonfinite_steps"), "episodes": r.get("episodes"),
            "steps": r.get("steps"), "tip_outside_steps": r.get("tip_outside_steps"), "tip_outside_max_mm": r.get("tip_outside_max_mm"),
            "max_rva_s_mm": r.get("max_rva_s_mm"), "ostium_mm": o_, "donor": g(r, "rva", "donor"),
            "entered_va_20mm_past_ostium": bool(o_ is not None and (r.get("max_rva_s_mm") or -1) >= o_ + 20.0),
            "reached_topbrain": r.get("reached_topbrain"), "donor_traversed_mm": r.get("donor_traversed_mm"),
            "dev_outside_max_mm": r.get("dev_outside_max_mm"), "dev_out_gt_0.3mm_pts": r.get("dev_out_gt_0.3mm_pts"),
            "dev_pts_checked": r.get("dev_pts_checked"), "ms_per_step": r.get("ms_per_step")}


def sofa_tally(rr, label):
    rr = [x for x in rr if x]
    don = [x for x in rr if x["donor"]]
    return {"n": len(rr), "passed": sum(bool(x["passed"]) for x in rr), "exceptions": sum(x["exception"] is not None for x in rr),
            "nonfinite_steps": sum(x["nonfinite_steps"] or 0 for x in rr), "episodes": sum(x["episodes"] or 0 for x in rr),
            "steps": sum(x["steps"] or 0 for x in rr),
            "entered_va_20mm_past_ostium": sum(x["entered_va_20mm_past_ostium"] for x in rr),
            "donor_rvas": len(don), "entered_donor_rva": sum(x["entered_va_20mm_past_ostium"] for x in don),
            "reached_topbrain": sum(bool(x["reached_topbrain"]) for x in rr),
            "tip_outside_steps": sum(x["tip_outside_steps"] or 0 for x in rr),
            "tip_outside_max_mm": max([x["tip_outside_max_mm"] or 0 for x in rr] or [0]),
            "dev_out_gt_0.3mm_frac": round(sum(x["dev_out_gt_0.3mm_pts"] or 0 for x in rr) / max(sum(x["dev_pts_checked"] or 0 for x in rr), 1), 5)}


sel = load(SP + "/checks/sofa_selection.json")
set25 = sel["most_changed_15"] + sel["random_10"]
S25 = {}
for n in set25:
    p5 = (W + "/sofa/v5/%s.json" if n in picks else SP + "/checks/sofa/run4/v5/%s.json") % n
    S25[n] = {"v5": sofa_row(p5, "v5"), "v4": sofa_row(SP + "/checks/sofa/run4/v4/%s.json" % n, "v4"), "rerun_after_swap": n in picks}
S13 = {}
for n in picks:
    p4 = W + "/sofa/v4/%s.json" % n
    if not os.path.exists(p4):
        p4 = SP + "/checks/sofa/run4/v4/%s.json" % n
    tg = g(prov[n], "v5_va", "route13", "targets") or []
    v5r = sofa_row(W + "/sofa/v5/%s.json" % n, "v5")
    S13[n] = {"v5": v5r, "v4": sofa_row(p4, "v4"), "moved_sides": tg,
              "moved_va_entered": (v5r["entered_va_20mm_past_ostium"] if (v5r and "right" in tg) else None),
              "note": None if "right" in tg else "only the LVA moved; the schedule drives into the RVA"}
sofa = {"tool": "checks/sofa_rva.py UNCHANGED (training loader + insertion, vessel-frame checks, closed-loop schedule into "
                "the RVA, up to 3 episodes x 300 steps; PASSED line = 18 branches, loader mesh = this OBJ, insertion inside, "
                "0 exceptions, 0 non-finite steps, tip never > 0.5 mm outside); entered = tip >= 20 mm past the VA ostium",
        "set25_final": {"rule": sel["rule"], "v5": sofa_tally([S25[n]["v5"] for n in set25], "v5"),
                        "v4_control": sofa_tally([S25[n]["v4"] for n in set25], "v4"),
                        "rerun_after_swap": [n for n in set25 if n in picks], "per_anatomy": S25},
        "swapped_13": {"v5": sofa_tally([S13[n]["v5"] for n in picks], "v5"),
                       "v4_control": sofa_tally([S13[n]["v4"] for n in picks], "v4"),
                       "rva_moved": sum("right" in S13[n]["moved_sides"] for n in picks),
                       "moved_rva_entered": sum(bool(S13[n]["moved_va_entered"]) for n in picks),
                       "per_anatomy": S13}}
pre = {}
for n in picks:
    b = sofa_row(SP + "/checks/sofa/run4/v5/%s.json" % n, "v5")
    if b:
        a = S13[n]["v5"]
        pre[n] = {"before_swap": {k: b[k] for k in ("passed", "max_rva_s_mm", "ostium_mm", "donor_traversed_mm",
                                                   "entered_va_20mm_past_ostium", "episodes")},
                  "after_swap": {k: a[k] for k in ("passed", "max_rva_s_mm", "ostium_mm", "donor_traversed_mm",
                                                  "entered_va_20mm_past_ostium", "episodes")}}
sofa["swapped_13"]["before_vs_after_swap (check-stage run4 v5 vs swap13 re-run)"] = pre
sofa["swapped_13"]["lost_va_entry_with_move"] = sorted(n for n, r in pre.items() if r["before_swap"]["entered_va_20mm_past_ostium"]
                                                       and not r["after_swap"]["entered_va_20mm_past_ostium"])
allset = sorted(set(set25) | set(picks))
sofa["all_distinct"] = {"n": len(allset),
                        "v5": sofa_tally([(S13[n]["v5"] if n in picks else S25[n]["v5"]) for n in allset], "v5"),
                        "v4_control": sofa_tally([(S13[n]["v4"] if n in picks else S25[n]["v4"]) for n in allset], "v4")}

# ------------------------------------------------------------------ 9. donors, flags, licences
ASG = load(SP + "/results/v5_assignment.json")
units = {t["unit"]: t for t in ASG["table"]}
va_src = collections.Counter()
tnp = cal2 = 0
review = collections.Counter()
for n in names:
    v = prov[n]["v5_va"]
    for side in ("right", "left"):
        x = v.get(side) or {}
        if not x.get("replaced"):
            continue
        d = v["summary"]["donor"][side]
        va_src[d["source"]] += 1
        tnp += bool(d.get("takeoff_not_preserved"))
        cal2 += x.get("splice_calibre_pass") == 2
        if d.get("flags"):
            review["vas_with_donor_review_flags"] += 1
kept_sides = [{"anatomy": n, "side": s, "by": ("bake" if (prov[n]["v5_va"].get(s) or {}).get("kept_by") else "assignment"),
               "why": (prov[n]["v5_va"].get(s) or {}).get("why")} for n in names for s in ("right", "left")
              if not (prov[n]["v5_va"].get(s) or {}).get("replaced")]
nc = [n for n in names if prov[n]["v5_va"]["summary"].get("non_commercial")]
att = collections.Counter(a for n in names for a in (prov[n]["v5_va"]["summary"].get("attributions") or []))
vmr = [n for n in names if any(u.startswith("vmr") for u in (prov[n]["v5_va"]["assignment"].get("assigned_units") or []))]
vmr_lic_file = [n for n in vmr if glob.glob(os.path.join(A5, n, "LICENSE_VMR*"))]
lic_file = sum(os.path.exists(os.path.join(A5, n, "LICENSE_v5_donors.txt")) for n in names)
donors = {"units": len(units), "by_source_units": dict(collections.Counter(t["source"] for t in units.values())),
          "by_source_donor_vas": dict(va_src), "topbrain_patients": len({t["patient"] for t in units.values()}),
          "one_to_one": len(units) == len({t["patient"] for t in units.values()}),
          "reserves_used": ASG["counts"].get("reserves_used"),
          "anatomies_per_unit": {u: t["anatomies"] for u, t in units.items()}}
flags = {"bake_anatomy_flags": S["flags"], "flagged_anatomies": S["flagged_anatomies"],
         "obj_fix": S["obj_fix"], "takeoff_not_preserved_vas": tnp, "calibre_second_pass_vas": cal2,
         "vas_with_donor_review_flags": review["vas_with_donor_review_flags"],
         "va_windows": {"vas": S["donor_vas_with_windows"], "tubed_mm": S["tubed_mm"]}}
licences = {"non_commercial_anatomies": len(nc), "non_commercial_rule": "Arterial-derived donor VA (CC BY-NC 4.0 weights, PolyForm-NC code)",
            "vmr0248_anatomies": len(vmr), "vmr0248_licence": units.get("vmr0248", {}).get("licence"),
            "vmr0248_non_commercial_in_record": units.get("vmr0248", {}).get("non_commercial"),
            "vmr0248_anatomies_with_LICENSE_VMR_file": len(vmr_lic_file),
            "anatomies_with_LICENSE_v5_donors_txt": lic_file,
            "per_unit": {u: {"source": t["source"], "licence": t["licence"], "non_commercial": t["non_commercial"]} for u, t in units.items()},
            "attributions": [{"text": a, "anatomies": c} for a, c in att.most_common()]}

# ------------------------------------------------------------------ 10. route13 record
r13 = {}
for n, var in picks.items():
    e = R13[n]
    ch = e["variants"][e["chosen"]["variant"]]
    bk_bake = load(SP + "/backup_pre_route13/_bake_full_done/%s.bake.json" % n)
    pv = prov[n]["v5_va"].get("route13") or {}
    r13[n] = {"variant": var, "targets": pv.get("targets"), "tool": pv.get("tool"),
              "pose_family": {s: g(ch, "course", s, "pose_family") for s in ("right", "left")},
              "before_k24": e["v5_as_baked"]["route_ratio"], "after_k24_record": ch["route_ratio"],
              "after_exact": {s: {t: round(RX[n]["exact"]["ratio"][s][t], 4) for t in ("frac_0.3mm", "frac_0.5mm")} for s in ("fork", "siphon")},
              "after_exact_class": RX[n]["exact"]["identical_beyond_mm_tol"],
              "rung_before_after": [bk_bake.get("rung"), brow[n]["rung"]],
              "flags_before_after": [bk_bake.get("flags"), brow[n]["flags"]],
              "flags_after": brow[n]["flags"],
              "course": {s: {k: g(ch, "course", s, k) for k in ("course_shift_vs_v5_rms_mm", "course_rms_to_v4_corridor_mm",
                                                                  "takeoff_10mm_deg", "takeoff_10mm_deg_v5", "tip_moved_vs_v4_mm",
                                                                  "join_kink_deg", "final_clearance_mm", "lost_v4_rcca_voxels_v5",
                                                                  "lost_v4_rcca_voxels_chosen")}
                         for s in ("right", "left") if g(ch, "course", s, "target")},
              "va_tubed_mm_after": {s: g(brow[n], "va", s, "tubed_mm") for s in ("right", "left")},
              "swap": swap_marks[n]}
route13 = {"what": "13 anatomies whose right_route_detail (k24, > 0.3 mm) fell below 0.99 of v4 after the full bake: the "
                   "competing VA was MOVED (same donor, sides and assignment) with graft_v5_reg_sep (separation objective: "
                   "fewest v4-kept RCCA real voxels refused under sdf_union_v4's ownership rule, among candidates passing "
                   "every graft_v5_reg gate), re-baked with bake_v5b (standard ownership), measured with "
                   "bake_full/measure_full_one.py, verified (notes/verify_v5_route13.md), and swapped into anatomies_v5 on "
                   "2026-10-03 (v5_swap13). Route priority (bake_v5c) was built, never chosen.",
           "picks": dict(collections.Counter(picks.values())), "route_priority_used": 0,
           "swap": {"replaced": "anatomies_v5/<name>/ (all files), va_sources/v5_reg/iso_v5/<name>.vtp, "
                                "va_sources/v5_reg/bake_full/_done/<name>.{bake,measure}.json",
                    "source": "va_sources/v5_reg/route13/<variant>/<name>/ (+ <variant>_iso, <variant>_bf/_done)",
                    "hash_check": "sha256 of every file equals the pick's (swap13/manifests/<name>.{pick,v5_after}.sha256)",
                    "backup": "v5_va/backup_pre_route13/<name>/ + _iso_v5/ + _bake_full_done/ + BUILD_v5.json (sha256-verified "
                              "byte copies of the pre-swap state; scratchpad, temporary)",
                    "provenance": "unchanged from the pick: v5_va.route13 records the move (tool, targets, separation)"},
           "anatomies": r13}

# ------------------------------------------------------------------ 11. BUILD_v5.json additions
build["grafter"] += ("; route13 (13 anatomies): graft_v5_reg_sep 1.0 (separation objective S1-S7 on graft_v5_reg 1.1) "
                     "moved the competing VA, re-baked with bake_v5b")
build["route13"] = {k: route13[k] for k in ("what", "picks", "route_priority_used", "swap")}
build["route13"]["anatomies"] = {n: {k: r13[n][k] for k in ("variant", "targets", "before_k24", "after_k24_record", "after_exact",
                                                           "after_exact_class", "flags_after")} for n in r13}
build["v4_va_kept_sides"] = {"total": len(kept_sides), "by_assignment": sum(k["by"] == "assignment" for k in kept_sides),
                             "by_bake": sum(k["by"] == "bake" for k in kept_sides), "sides": kept_sides}
build["donor_flags"] = flags
build["licences"].update({k: licences[k] for k in ("non_commercial_anatomies", "non_commercial_rule", "vmr0248_anatomies",
                                                    "vmr0248_licence", "vmr0248_non_commercial_in_record",
                                                    "vmr0248_anatomies_with_LICENSE_VMR_file",
                                                    "anatomies_with_LICENSE_v5_donors_txt", "attributions")})
build["checks_final"] = {
    "file": "v5_va/results/v5_final_summary.json",
    "strict_mesh_audit_pass": audit["pass_counts"]["strict_pass"],
    "self_intersecting_faces_total": audit["self_intersecting_faces_total"],
    "right_route_vs_v4_exact": dict({k: route["exact"][k] for k in ("identical_cells_everywhere", "identical_cells_beyond_mm",
                                                                   "anatomies_below_0.99", "below_0.99", "ratio_fork_0.3mm",
                                                                   "ratio_siphon_0.3mm", "lowest")},
                                    below_0_99_note=(
        "w_018b_left mr_002 (not among the 13: its k24 route equals v4 at 4 dp): the exact-ownership loss is 121 v4 wall "
        "cells (7.85 mm2) near the bifurcation (RCCA s 92-103 mm) that k24 gives to the RECA and exact ownership to the "
        "RCCA, deeper than 0.3 mm (up to 3.9 mm) off its tube; in v5 that wall is reshaped (nearest v5 cell 0-1.6 mm "
        "away) where the donor VA passes 1.6-3.9 mm from it (v4's host VA: 13.5-16 mm). The RCCA's own route wall under "
        "k24 is identical to v4. Not moved; see v5_final_summary.json right_route_vs_v4.exact.below_0_99_diagnosis") if "case_w_018b_left__topcow_mr_002" in
        {x[0] for x in route["exact"]["below_0.99"]} else None),
    "right_route_vs_v4_k24": {k: route["k24"][k] for k in ("identical_cells_everywhere", "identical_cells_beyond_mm",
                                                          "anatomies_below_0.99", "lowest")},
    "va_wall_exact_seam0": {"donor_real_frac_0.3mm": wall["donor_real_frac_0.3mm"],
                            "whole_va_ostium_to_end_seam0_frac_0.3mm": wall["whole_va_ostium_to_end_seam0_frac_0.3mm"],
                            "seam_zone_frac_0.3mm": wall["seam_zone_frac_0.3mm"],
                            "donor_real_gt_v4_host_core_same_side": wall["donor_real_gt_v4_host_core_same_side"],
                            "whole_va_gt_v4_whole_va_same_side": wall["whole_va_seam0_gt_v4_whole_va_same_side"]},
    "split": {"exit": split["exit"], "not_lowered": split["not_lowered"]},
    "loader": {k: loader[k] for k in ("anatomies", "n_18_branches", "insertion_inside", "centreline_outside_anatomies")},
    "sofa": {"set25_v5_passed": sofa["set25_final"]["v5"]["passed"], "set25_v5_entered_va": sofa["set25_final"]["v5"]["entered_va_20mm_past_ostium"],
             "set25_v4_entered_va": sofa["set25_final"]["v4_control"]["entered_va_20mm_past_ostium"],
             "swapped13_v5_passed": sofa["swapped_13"]["v5"]["passed"], "swapped13_moved_rva_entered": sofa["swapped_13"]["moved_rva_entered"],
             "swapped13_rva_moved": sofa["swapped_13"]["rva_moved"],
             "set25_v5_entered_donor_rva": [sofa["set25_final"]["v5"]["entered_donor_rva"], sofa["set25_final"]["v5"]["donor_rvas"]],
             "swapped13_lost_va_entry_with_move": sofa["swapped_13"]["lost_va_entry_with_move"],
             "note": "entered = tip >= 20 mm past the VA ostium; the donor ostium sits 3-10 mm past the subclavian branch "
                     "point through a narrow stub, so the closed-loop schedule enters fewer donor VAs than v4's (user "
                     "question pending: do VA targets matter for training?)"}}
build.setdefault("timings", {})["route13"] = ("v5_route13 (EDT): diagnosis 2026-10-02 22:55-23:22, re-graft (moved_ext) to 23:39, "
                                              "scratch bake + measure to 2026-10-03 00:05 (moved_ext) / 00:23 (moved_pin); "
                                              "swapped into anatomies_v5 2026-10-03 02:12 (v5_swap13)")
build["records"]["collect"] = "v5_va/results/v5_bake_full_final.json (the pre-swap collection: v5_bake_full.json)"
build["records"]["final_summary"] = "v5_va/results/v5_final_summary.json"
build["records"]["route13"] = "v5_va/results/v5_route13.json; notes/v5_route13.md, notes/verify_v5_route13.md, notes/v5_swap13.md"
build["generated"] = NOW
json.dump(build, open(BOUT + "/BUILD_v5.json.tmp", "w"), indent=1)
os.replace(BOUT + "/BUILD_v5.json.tmp", BOUT + "/BUILD_v5.json")

# ------------------------------------------------------------------ 12. the final summary
counts = {"anatomies": len(names), "v4_anatomies": build["v4_anatomies"], "dropped": build["dropped"],
          "baked_and_passing": S["all_pass"], "donor_vas": S["donor_vas"], "replaced": S["replaced"],
          "v4_va_kept_sides": build["v4_va_kept_sides"]["total"],
          "v4_va_kept_by_assignment": build["v4_va_kept_sides"]["by_assignment"],
          "v4_va_kept_by_bake": build["v4_va_kept_sides"]["by_bake"],
          "donor_units": donors["units"], "topbrain_patients": donors["topbrain_patients"],
          "route13_moved": len(picks), "rungs": S["rungs"]}
gates = {"bake_record": {"all_pass": S["all_pass"], "clean_tree": S["clean"], "self_intersecting_faces_total": S["self_intersecting_faces_total"],
                         "routes_navigable": S["routes_navigable"], "centerline_pts_outside_total": S["centerline_pts_outside_total"],
                         "obj_tris": S["obj_tris"], "obj_budget": S["obj_budget"], "full_tris": S["full_tris"],
                         "donor_vas_real_wall_owned": S["donor_vas_real_wall"], "donor_vas_whole_tube": S["donor_vas_whole_tube"],
                         "route_lumen_min_mm": S["route_lumen_min_mm"]},
         "strict_audit": audit}
summary = {"what": "v5 final: the 223 v4 carotid anatomies with real donor lower VAs (strategy B), after the route13 swap",
           "generated": NOW, "generated_by": "v5_va/swap13/final_collect.py",
           "counts": counts, "gates": gates, "right_route_vs_v4": route, "va_wall_detail": wall, "lumen": lumen,
           "split": split, "loader": loader, "sofa": sofa, "donors": donors, "flags": flags, "licences": licences,
           "route13": route13,
           "inputs": {"swapped_13": "v5_va/swap13/{chk_out,vvw_out,rrdx,sofa,lumen13.json,va_lumen13.json}",
                      "other_210": "v5_va/checks/out, v5_va/vvw/out, v5_va/checks/va_lumen.json, v5_va/checks/sofa/run4 "
                                   "(folders unchanged since: no file newer than 2026-10-02 20:30)",
                      "all_223": "v5_va/swap13/rrdx (route exact + k24), v5_va/swap13/loader_v5.json, results/split_v5_final.json, "
                                 "bake markers va_sources/v5_reg/bake_full/_done"}}
json.dump(summary, open(SUMMARY + ".tmp", "w"), indent=1, default=float)
os.replace(SUMMARY + ".tmp", SUMMARY)
print("WROTE", BOUT + "/BUILD_v5.json", SUMMARY)
print("COUNTS", json.dumps(counts))
print("AUDIT", json.dumps({k: audit[k] for k in ("pass_counts", "swapped_13_strict_pass", "self_intersecting_faces_total", "failing")}))
for mode in ("exact", "k24"):
    print("ROUTE", mode, json.dumps({k: v for k, v in route[mode].items() if k not in ("below_0.99", "swapped_13")}))
print("ROUTE validation", json.dumps(route["validation"]))
print("WALL", json.dumps({k: v for k, v in wall.items() if k not in ("swapped_13_moved_vas",)}))
print("LUMEN", json.dumps(lumen["route_lumen_undecimated_swapped_13"]), json.dumps({k: v for k, v in lumen["va_lumen_obj"].items() if k != "swapped_13_before_vs_now"}))
print("SPLIT", json.dumps(split["exit"]), split["not_lowered"])
print("LOADER", json.dumps(loader))
print("SOFA25", json.dumps(sofa["set25_final"]["v5"]), json.dumps(sofa["set25_final"]["v4_control"]))
print("SOFA13", json.dumps(sofa["swapped_13"]["v5"]), json.dumps(sofa["swapped_13"]["v4_control"]),
      sofa["swapped_13"]["rva_moved"], sofa["swapped_13"]["moved_rva_entered"])
print("DONORS", json.dumps(donors["by_source_units"]), json.dumps(donors["by_source_donor_vas"]))
print("FLAGS", json.dumps({k: v for k, v in flags.items() if k != "flagged_anatomies"}))
print("LIC", licences["non_commercial_anatomies"], licences["vmr0248_anatomies"], licences["vmr0248_anatomies_with_LICENSE_VMR_file"],
      licences["anatomies_with_LICENSE_v5_donors_txt"])
