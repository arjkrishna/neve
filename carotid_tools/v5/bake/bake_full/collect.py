#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\collect.py (= /sp/v5_va/bake_full/collect.py in the build containers)
"""v5_bake_full: collect the 223 bake + measure records into /sp/v5_va/results/v5_bake_full.json, and (--build, only
when every anatomy is baked, measured and passes) write carotid_data/anatomies_v5/BUILD_v5.json mirroring
anatomies_v4/BUILD_v4.json. Read-only on the anatomies otherwise.
    python3 collect.py [--build]        (neve-build-meshlab)
"""
import argparse
import glob
import json
import os
import statistics
import time
from collections import Counter

A5 = "/repo/carotid_data/anatomies_v5"
A4 = "/repo/carotid_data/anatomies_v4"
BF = "/repo/carotid_data/va_sources/v5_reg/bake_full"
OUT = "/sp/v5_va/results/v5_bake_full.json"
ASSIGN = "/sp/v5_va/results/v5_assignment.json"
PROBE = "/sp/v5_va/results/v5_bake_probe.json"


def rng(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v:
        return None
    return {"min": round(min(v), 4), "median": round(statistics.median(v), 4), "max": round(max(v), 4), "n": len(v)}


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                                    # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(A5 + "/*/provenance.json"))
    order = [x.strip() for x in open("/sp/v5_va/bake_full/order.txt") if x.strip()]
    rows, missing, failed = [], [], []
    for n in names:
        bk = load(os.path.join(BF, "_done", n + ".bake.json"))
        ms = load(os.path.join(BF, "_done", n + ".measure.json"))
        bf = load(os.path.join(BF, "_done", n + ".bake_failed.json"))
        if bf and not bk:
            failed.append({"name": n, "error": bf.get("error")})
            continue
        if not bk or not ms:
            missing.append(n)
            continue
        rep = load(os.path.join(A5, n, "mesh_v5.json"))
        prov = load(os.path.join(A5, n, "provenance.json"))
        g = ms["gates"]
        rr = ms["right_route_detail"]
        rows.append({
            "name": n, "order": order.index(n) if n in order else None,
            "pass": g["all_pass"], "rung": bk["rung"], "flags": bk["flags"], "union": bk["union"],
            "carotid_margin_mm": bk["margin_mm"], "va_margin_mm": bk["va_margin_mm"],
            "far_clamp": bool(bk["far_clamp"]["used"]), "far_clamp_why": bk["far_clamp"].get("why"),
            "bake_one_calls": len(bk["calls"]), "attempts": bk["attempts"],
            "sections_tube_only": bk["sections_tube_only"], "suppressed_stretches_mm": bk["suppressed_stretches_mm"],
            "seconds_bake": bk["seconds"], "seconds_measure": ms["seconds"], "peak_rss_gb_bake": bk["peak_rss_gb"],
            "peak_rss_gb_measure": ms["peak_rss_gb"], "container_memory_peak_gb": bk.get("container_memory_peak_gb"),
            "obj_tris": g["obj_tris"], "obj_budget": g["obj_budget"], "full_tris": g["full_tris"],
            "full_budget": g["full_budget"], "union_tris": g["union_tris"],
            "clean": {k: g[k + "_clean_tree"] for k in ("union", "full", "obj")},
            "self_intersecting_faces": g["self_intersecting_faces"],
            "routes_navigable": g["routes_navigable"], "route_lumen_min_mm": g["route_lumen_min_mm"],
            "centerline_pts_outside": g["centerline_pts_outside"],
            "decimation": {"full": rep["decimation"].get("full"), "obj": rep["decimation"].get("obj")},
            "obj_fix": (rep.get("obj_fix") or {}).get("accepted"),
            "v4_va_kept_by_bake": [s for s in ("right", "left") if (prov["v5_va"].get(s) or {}).get("kept_by")],
            "right_route_detail": {"v5": rr["v5"], "v4": rr["v4"], "identical": rr["identical"],
                                   "ratio": rr["ratio"], "identical_beyond_mm": rr["identical_beyond_mm"],
                                   "excluded_area_frac_8mm_v5": rr["away_from_vas"]["8mm"]["excluded_area_frac_v5"]},
            "v4_rebuild_max_abs_diff": ms.get("v4_rebuild_max_abs_diff"),
            "va": ms["va_detail"],
            "va_tubed_all_sections": bk["va_tubed"],
            "records_written": {"mesh_v5_has_gates": bool(rep and "v5_gates" in rep and "va_detail" in rep),
                                "provenance_bake_v5": bool(prov and "gates" in prov.get("bake_v5", {})
                                                           and "va_detail" in prov.get("bake_v5", {})
                                                           and "rung" in prov.get("bake_v5", {}))},
        })
    vas = [(r["name"], side, v) for r in rows for side, v in r["va"].items() if v.get("donor")]
    kept = [(r["name"], side, v.get("why")) for r in rows for side, v in r["va"].items() if not v.get("donor")]
    fl = Counter(f.split(" (")[0] for r in rows for f in r["flags"])
    summary = {
        "anatomies": len(names), "baked_and_measured": len(rows), "missing": missing, "bake_failed": failed,
        "all_pass": sum(r["pass"] for r in rows), "not_pass": [r["name"] for r in rows if not r["pass"]],
        "records_written_ok": sum(r["records_written"]["mesh_v5_has_gates"] and r["records_written"]["provenance_bake_v5"]
                                  for r in rows),
        "rungs": dict(Counter(r["rung"] for r in rows)),
        "flags": dict(fl),
        "flagged_anatomies": {f: [r["name"] for r in rows if any(x.startswith(f) for x in r["flags"])] for f in fl},
        "far_clamp": [r["name"] for r in rows if r["far_clamp"]],
        "two_bake_one_calls": [r["name"] for r in rows if r["bake_one_calls"] > 1],
        "obj_fix": [[r["name"], r["obj_fix"]] for r in rows if r["obj_fix"]],
        "v4_va_kept_by_bake": [[r["name"], s] for r in rows for s in r["v4_va_kept_by_bake"]],
        "replaced": {"right": sum(1 for r in rows if r["va"]["right"].get("donor")),
                     "left": sum(1 for r in rows if r["va"]["left"].get("donor")),
                     "anatomies_both": sum(1 for r in rows if r["va"]["right"].get("donor") and r["va"]["left"].get("donor")),
                     "anatomies_one": sum(1 for r in rows if bool(r["va"]["right"].get("donor")) != bool(r["va"]["left"].get("donor"))),
                     "anatomies_none": sum(1 for r in rows if not r["va"]["right"].get("donor") and not r["va"]["left"].get("donor"))},
        "route_ratio_below_1": sorted([[r["name"], r["right_route_detail"]["ratio"]["fork"]["frac_0.3mm"],
                                        r["right_route_detail"]["ratio"]["siphon"]["frac_0.3mm"],
                                        r["right_route_detail"]["identical_beyond_mm"], r["flags"]] for r in rows
                                       if min(r["right_route_detail"]["ratio"]["fork"]["frac_0.3mm"] or 1,
                                              r["right_route_detail"]["ratio"]["siphon"]["frac_0.3mm"] or 1) < 0.999],
                                      key=lambda x: min(x[1], x[2])),
        "clean": {k: sum(r["clean"][k] for r in rows) for k in ("union", "full", "obj")},
        "self_intersecting_faces_total": {k: sum(r["self_intersecting_faces"][k] for r in rows) for k in ("union", "full", "obj")},
        "obj_tris": rng([r["obj_tris"] for r in rows]), "obj_budget": dict(Counter(r["obj_budget"] for r in rows)),
        "full_tris": rng([r["full_tris"] for r in rows]), "full_budget": dict(Counter(r["full_budget"] for r in rows)),
        "union_tris": rng([r["union_tris"] for r in rows]),
        "routes_navigable": sum(r["routes_navigable"] for r in rows),
        "route_lumen_min_mm": {k: rng([r["route_lumen_min_mm"][k] for r in rows]) for k in ("RCCA", "LCCA")},
        "centerline_pts_outside_total": sum(r["centerline_pts_outside"] for r in rows),
        "decimation": {k: dict(Counter(r["decimation"][k] for r in rows)) for k in ("full", "obj")},
        "right_route_detail": {
            "identical_to_v4": sum(r["right_route_detail"]["identical"] for r in rows),
            "identical_beyond_mm": dict(Counter(str(r["right_route_detail"]["identical_beyond_mm"]) for r in rows)),
            "not_identical_even_beyond_8mm": [r["name"] for r in rows if r["right_route_detail"]["identical_beyond_mm"] is None],
            "ratio_fork_0.3mm": rng([r["right_route_detail"]["ratio"]["fork"]["frac_0.3mm"] for r in rows]),
            "ratio_siphon_0.3mm": rng([r["right_route_detail"]["ratio"]["siphon"]["frac_0.3mm"] for r in rows]),
            "excluded_area_frac_8mm_v5": rng([r["right_route_detail"]["excluded_area_frac_8mm_v5"] for r in rows])},
        "v4_rebuild_exact": sum(1 for r in rows if r["v4_rebuild_max_abs_diff"] == 0),
        "donor_vas": len(vas), "v4_va_kept_sides": len(kept),
        "donor_vas_real_wall": sum(1 for _, _, v in vas if v["real_surface"]),
        "donor_vas_whole_tube": [(n, s) for n, s, v in vas if v["whole_tube"]],
        "donor_vas_with_windows": sum(1 for _, _, v in vas if v["n_windows"]),
        "windows_per_donor_va": dict(Counter(v["n_windows"] for _, _, v in vas)),
        "tubed_mm": rng([v["tubed_mm"] for _, _, v in vas if v["n_windows"]]),
        "tubed_frac": rng([v["tubed_frac"] for _, _, v in vas if v["n_windows"]]),
        "tubed_mm_max_anatomy": max(((v["tubed_mm"], n, s) for n, s, v in vas), default=None),
        "donor_real_frac_0.3mm": rng([v["donor_real_frac_0.3mm"] for _, _, v in vas]),
        "whole_va_conservative_frac_0.3mm": rng([v["whole_va_conservative_frac_0.3mm"] for _, _, v in vas]),
        "detail_kept_in_situ_over_alone": rng([v["A2_detail_kept_in_situ_over_alone"] for _, _, v in vas]),
        "detail_kept_below_0.9": [(n, s, v["A2_detail_kept_in_situ_over_alone"], v["tubed_mm"]) for n, s, v in vas
                                 if isinstance(v["A2_detail_kept_in_situ_over_alone"], (int, float))
                                 and v["A2_detail_kept_in_situ_over_alone"] < 0.9],
        "A2_errors": [(n, s, v["A2_error"]) for n, s, v in vas if v.get("A2_error")],
        "v4_host_tube_frac_0.3mm": rng([v["v4_host_core_frac_0.3mm"] for _, _, v in vas]),
        "seconds_bake": rng([r["seconds_bake"] for r in rows]),
        "seconds_measure": rng([r["seconds_measure"] for r in rows]),
        "peak_rss_gb_bake": rng([r["peak_rss_gb_bake"] for r in rows]),
        "peak_rss_gb_measure": rng([r["peak_rss_gb_measure"] for r in rows]),
    }
    # windows per donor side
    by_donor = {}
    for n, s, v in vas:
        d = by_donor.setdefault(v["donor"], {"genus": v["donor_surface_genus"], "vas": 0, "with_windows": 0,
                                             "tubed_mm_max": 0.0, "windows_from_ostium_mm": []})
        d["vas"] += 1
        if v["n_windows"]:
            d["with_windows"] += 1
            d["tubed_mm_max"] = max(d["tubed_mm_max"], v["tubed_mm"])
            d["windows_from_ostium_mm"] += v["windows_from_ostium_mm"]
    for d in by_donor.values():
        w = d.pop("windows_from_ostium_mm")
        d["window_from_ostium_mm_range"] = [min(x[0] for x in w), max(x[1] for x in w)] if w else None
    summary["by_donor_side"] = dict(sorted(by_donor.items(), key=lambda kv: -(kv[1]["genus"] or 0)))
    res = {"what": "v5_bake_full: the 223 anatomies_v5 baked in place with /sp/v5_va/bake_v5b.py (user handle rule) "
                   "and measured (gates, right route vs v4, VA wall detail on the undecimated union)",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary, "kept_v4_va_sides": kept,
           "anatomies": rows}
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(json.dumps({k: summary[k] for k in ("anatomies", "baked_and_measured", "all_pass", "bake_failed", "rungs",
                                                "flags", "clean", "right_route_detail")}, indent=0)[:3000])
    if not a.build:
        return 0
    ok = (len(rows) == len(names) == 223 and summary["all_pass"] == 223 and not failed
          and summary["records_written_ok"] == 223)
    if not ok:
        print("NOT writing BUILD_v5.json: not every anatomy is baked, measured and passing")
        return 1
    build_v5(rows, summary, kept)
    return 0


def build_v5(rows, S, kept):
    b4 = json.load(open(os.path.join(A4, "BUILD_v4.json")))
    asg = json.load(open(ASSIGN))
    import sys
    sys.path.insert(0, "/sp/v5_va")
    import bake_v5b as V
    tab = asg["table"]
    lic = {}
    for t in tab:
        lic[t["unit"]] = {"source": t.get("source"), "case": t.get("case"), "licence": t.get("licence"),
                          "non_commercial": t.get("non_commercial")}
    donors = [{"unit": t["unit"], "class": t.get("class"), "source": t.get("source"), "case": t.get("case"),
               "topbrain_patient": t["patient"], "anatomies": t.get("anatomies"), "rva_replaced": t.get("rva_replaced"),
               "lva_replaced": t.get("lva_replaced"), "non_commercial": t.get("non_commercial")} for t in tab]
    sec = [r["seconds_bake"] for r in rows]
    out = {
        "set": "B v5: the 223 v4 anatomies with the shared host lower vertebral artery replaced by real donor cervical "
               "VAs (strategy B; 25 donor pairs one-to-one with the 25 TopBrain patients)",
        "v4_anatomies": b4["v4_anatomies"], "v5_anatomies": len(rows), "dropped": [],
        "grafter": "v5_va/graft_v5_va.py + graft_v5_reg 1.1 (registration / gates) + graft_v5_assign 1.1 (one-to-one "
                   "assignment); on top of v4 (carotid_tools/graft_v4.py)",
        "mesher": "v5_va/bake_v5b.py = topbrain_tools/bake_meshes_v4.py UNCHANGED + bake_v5.py (the donor vaC_R / vaC_L "
                  "real sections) + the user handle rule (D1-D8): carotid sections pinned to v4's bake (margin, "
                  "suppressed stretches, whole tubes, far clamp); the VA sections alone searched for handles (25 rounds "
                  "of 10/20/40/80 mm bins + bisection), every new window pruned / split / shrunk to SHRINK_TOL 2 mm; "
                  "whole donor tube, carotid change, v4 rung 5 / tubes only = flagged fallbacks; then v4's own clean "
                  "decimation to 60000 (collision_full.vtp) and the 23500-tri OBJ",
        "build_images": dict(b4["build_images"], bake="neve-build-meshlab (as v4)",
                             donor_walls="neve-build-stagea / arterial images (va_sources build, see V5_BUILD_PLAN)"),
        "what_v5_adds": "each VA's lower (cervical) stretch, v4's shared host tube, replaced by a real donor VA: the "
                        "donor's closed lumen wall (CT-edge stage-A surface) from the subclavian ostium to a splice in V3, "
                        "rigidly registered onto the anatomy and owned in the SDF union like v4's real sections; v4's "
                        "TopBrain distal piece kept above the splice. Carotids, ECAs, siphons and routes unchanged.",
        "handedness": b4.get("handedness"),
        "flags": dict(b4["flags"], v5_va={"handle_rule": "local tube windows only (user 2026-10-02 16:20)",
                                          "va_rounds": V.VA_ROUNDS, "va_bins_mm": list(V.VA_BINS_MM),
                                          "va_min_window_mm": V.VA_MIN_WINDOW_MM,
                                          "va_split_above_mm": V.VA_SPLIT_ABOVE_MM,
                                          "shrink_tol_mm": V.SHRINK_TOL, "va_margins_mm": list(V.VA_MARGINS)}),
        "sources": dict(b4["sources"], donors_v5="carotid_data/va_sources/v5_reg/donors_v5/<unit>/{R,L}.json + "
                                                 "{R,L}_surface.vtp"),
        "donors": {"units": len(donors), "by_source": dict(Counter(d["source"] for d in donors)), "table": donors},
        "assignment": {"file": "v5_va/results/v5_assignment.json", "objective_order": asg["objective"]["order"],
                       "counts": asg["counts"],
                       "flagged_anatomies_v4_va_kept": [{"anatomy": f["anatomy"], "v4_va_kept": f["v4_va_kept"]}
                                                        for f in asg["flagged_anatomies_v4_va_kept"]]},
        "licences": {"per_unit": lic, "non_commercial_anatomies": asg["counts"].get("non_commercial_anatomies"),
                     "per_anatomy": "provenance.json v5_va.licences + LICENSE_v5_donors.txt in each anatomy folder"},
        "gates": {
            "standard": "v4: one closed genus-0 two-manifold tree (union, collision_full, OBJ), 0 self-intersecting "
                        "faces, OBJ at the 23.5k budget, every centreline point inside, both routes navigable; donor "
                        "VA real wall owned (real_surface true)",
            "all_pass": S["all_pass"], "clean_tree": S["clean"],
            "self_intersecting_faces_total": S["self_intersecting_faces_total"],
            "routes_navigable": S["routes_navigable"], "centerline_pts_outside_total": S["centerline_pts_outside_total"],
            "obj_budget": S["obj_budget"], "full_budget": S["full_budget"],
            "donor_vas": S["donor_vas"], "donor_vas_real_wall_owned": S["donor_vas_real_wall"],
            "right_route_detail": S["right_route_detail"]},
        "fallbacks": {"donor_tube (whole donor section as tube)": S["donor_vas_whole_tube"],
                      "flagged": S["flagged_anatomies"],
                      "v4_va_kept_sides (assignment: no gate-passing donor pose)": [list(k) for k in kept],
                      "v4_va_kept_by_bake (tube-only union not a tree: centreline-level fusion)": S["v4_va_kept_by_bake"],
                      "obj_fix (no clean 23.5k from v4's obj_ladder; stated alternative target/source)": S["obj_fix"]},
        "replaced_after_bake": S["replaced"],
        "handles": {"donor_vas_with_windows": S["donor_vas_with_windows"],
                    "windows_per_donor_va": S["windows_per_donor_va"], "tubed_mm": S["tubed_mm"],
                    "tubed_frac_of_donor_stretch": S["tubed_frac"], "by_donor_side": S["by_donor_side"]},
        "va_detail_undecimated_union": {
            "donor_real_frac_0.3mm": S["donor_real_frac_0.3mm"],
            "whole_va_conservative_frac_0.3mm": S["whole_va_conservative_frac_0.3mm"],
            "detail_kept_in_situ_over_alone": S["detail_kept_in_situ_over_alone"],
            "v4_host_tube_frac_0.3mm": S["v4_host_tube_frac_0.3mm"]},
        "ranges": {"route_lumen_min_mm": S["route_lumen_min_mm"], "obj_tris": S["obj_tris"], "full_tris": S["full_tris"],
                   "union_tris": S["union_tris"]},
        "bake": {"pass": S["all_pass"], "rungs": S["rungs"], "far_clamp": len(S["far_clamp"]),
                 "two_bake_one_calls": len(S["two_bake_one_calls"]), "decimation": S["decimation"],
                 "carotid margin": dict(Counter(str(r["carotid_margin_mm"]) for r in rows)),
                 "va margin": dict(Counter(str(r["va_margin_mm"]) for r in rows))},
        "timings": {"seconds_bake_per_anatomy": S["seconds_bake"], "seconds_measure_per_anatomy": S["seconds_measure"],
                    "bake_cpu_hours": round(sum(sec) / 3600.0, 2), "peak_rss_gb_bake": S["peak_rss_gb_bake"],
                    "peak_rss_gb_measure": S["peak_rss_gb_measure"], "workers": "8 containers (neve-build-meshlab, one anatomy each at a time, hardest first), 10 from 18:34",
                    "wall_clock": "2026-10-02 17:29-20:21 EDT (bake + measure of 223), fixes to 20:26"},
        "bake_failing": [],
        "records": {"per_anatomy": "mesh_v5.json (v4 keys + ladder_v5b + v5_gates + va_detail); provenance.json "
                                   "bake_v5", "collect": "v5_va/results/v5_bake_full.json",
                    "undecimated_unions": "carotid_data/va_sources/v5_reg/iso_v5/<name>.vtp"},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p = os.path.join(A5, "BUILD_v5.json")
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    print("wrote", p)


if __name__ == "__main__":
    raise SystemExit(main())
