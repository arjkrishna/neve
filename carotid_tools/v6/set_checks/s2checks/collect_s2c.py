#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\collect_s2c.py (= /sp/v6/s2checks/collect_s2c.py in the build containers)
"""collect_s2c.py 1.0 (s2_checks, 2026-10-07): the SET-LEVEL CHECKS of carotid_data/anatomies_v6 -> v6_checks.json and
v6_final_summary.json.

Inputs (all read-only):
  /s2c (= carotid_data/v6_work/s2checks, this agent's runs ON anatomies_v6):
    split/split_v6.json + exit.txt         v5_split.py anatomies_v6 --baseline split_v4.json
    loader/loader_set_v6.json              the training loader on the whole folder (loader_set.py)
    sofa/rcca_v6/<n>.json (223)            sofa_rcca_v6 1.0, seeds 777-779, 300 steps (RCCA training route)
    sofa/rcca_v5/<n>.json (25)             the same on anatomies_v5 (reach comparison on the v5 SOFA set)
    sofa/d14_{v6,v5,v4}/<n>.json (25 each) sofa_va_entry 1.1 (D14), same seeds
    entry_geom/geom25_<t>.json, geom223_<t>.json   va_entry_geom 1.0 (geometric entry probe)
    lumen/va_lumen_poly_v6.json            wall/va_lumen_poly.py on every anatomies_v6 OBJ + collision_full
    orient/orient_set_v{6,5}.json          orient_set.py (orientation gates, ICA-terminus AP, fork headings)
    ident/ident_v6_tree3.json              anatomies_v6 == the checked tree (/full/tree3), file for file
    route_spot/<n>.json                    checks_v6 1.5 part route re-run straight from anatomies_v6 (4 anatomies)
  /full/checks4/<n>.json                   the build's per-anatomy checks (checks_v6 1.3 parts + route_d22 1.4 / 1.5)
  /full/load_v6/<n>.json                   the build's per-anatomy loader_one on anatomies_v6
  /sp/v6/results/v6_build.json             the build record (flags)
  /repo/carotid_data/v6_work/wall/lumen/va_lumen_poly_v5.json   v5 polyline lumen (v6_wall c2)
  /sp/v5_va/checks/out/<n>.json            v5's checks (v4 VA wall detail per side: item2.va.<side>.v4)
    python3 collect_s2c.py <out_dir> [--final]
"""
import collections
import glob
import json
import os
import sys
import time

import numpy as np

VERSION = "collect_s2c 1.0 (s2_checks, 2026-10-07)"
S2C, FULL = "/s2c", "/full"
A6 = "/repo/carotid_data/anatomies_v6"
MANIFEST_MD5 = "d65b72096b9555cb960e78f5a5cb1ba4"


def jl(p, d=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def st(v, nd=4):
    v = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)]
    if not v:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": int(len(a)), "min": round(float(a.min()), nd), "p10": round(float(np.percentile(a, 10)), nd),
            "median": round(float(np.median(a)), nd), "p90": round(float(np.percentile(a, 90)), nd),
            "max": round(float(a.max()), nd)}


def short(n):
    return n.replace("case_", "").replace("__topcow_", " ")


def merge(a, b):
    c = dict(a)
    c.update(b)
    return c


names = sorted(d for d in os.listdir(A6) if d.startswith("case_") and os.path.isdir(os.path.join(A6, d)))
assert len(names) == 223, len(names)
sel = jl(os.path.join(S2C, "sofa", "set25.json"))
set25 = sel["set25"]
CK = {n: jl(os.path.join(FULL, "checks4", n + ".json")) for n in names}
B = jl("/sp/v6/results/v6_build.json")
BA = B["anatomies"]
out = {"version": VERSION, "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "tree": "carotid_data/anatomies_v6", "n": len(names),
       "toolchain": {"manifest": "/sp/v6/build/TOOLCHAIN_s2build.md5", "md5": MANIFEST_MD5,
                     "note": "every tool run by this agent is a manifest file (md5sum -c before the runs); the new set-level "
                             "scripts are scratchpad/v6/s2checks/*.py (orient_set, loader_set, ident_check, collect_s2c)"}}
inv = collections.OrderedDict()   # invariant -> {pass, n, source}

# ------------------------------------------------------------------ 0. identity
idn = jl(os.path.join(S2C, "ident", "ident_v6_tree3.json"), {})
out["identity"] = {"ok": idn.get("ok"), "files_compared": idn.get("files_compared"), "bad": idn.get("bad"),
                   "meaning": "anatomies_v6 == /full/tree3 (the tree checks_v6 measured) file for file (sha256); provenance = "
                              "tree3's + only v6_checks; the undecimated union and a complete passing checks4 record exist for "
                              "every anatomy -> the per-anatomy check records hold for anatomies_v6"}

# ------------------------------------------------------------------ 1. split
sp = jl(os.path.join(S2C, "split", "split_v6.json"), {})
try:
    sp_exit = int(open(os.path.join(S2C, "split", "exit.txt")).read().split()[-1])
except Exception:  # noqa: BLE001
    sp_exit = None
bl = sp.get("baseline") or {}
rows = bl.get("rows") or []
raised = [r for r in rows if isinstance(r, dict) and (r.get("now") or 0) > (r.get("baseline") or 0)]
nest = {d: g(sp, "defs", d, "extra_nesting") or g(sp, "defs", d, "nesting") for d in ("v3", "v4side", "strict")}
out["split"] = {"tool": "/sp/v5_va/v5_split.py (manifest)", "args": "anatomies_v6 --baseline /sp/v5_va/results/split_v4.json",
                "exit": sp_exit, "compared": bl.get("compared"), "lowered": bl.get("lowered"),
                "raised": raised, "rows": rows, "topbrain_patients": sp.get("topbrain_patients"),
                "extra_donor_nesting": nest, "file": "/s2c/split/split_v6.json"}
low = bl.get("lowered")
inv["split_exit0_none_of_45_lowered"] = {"pass": sp_exit == 0 and bl.get("compared") == 45 and not low,
                                         "detail": "exit %s, %s compared, lowered %s" % (sp_exit, bl.get("compared"), low)}
d1 = g(B, "compose_summary", "donors_one_to_one") or {}
inv["donor_units_one_to_one_25"] = {"pass": bool(d1.get("one_to_one")) and d1.get("donor_units") == 25,
                                    "detail": "%s donor units onto %s TopBrain patients (build record); split: V: donors nested "
                                              "one-to-one on every definition" % (d1.get("donor_units"), d1.get("patients_with_donor"))}

# ------------------------------------------------------------------ 2. loader
ls_ = jl(os.path.join(S2C, "loader", "loader_set_v6.json"), {})
lo1 = {n: jl(os.path.join(FULL, "load_v6", n + ".json")) for n in names}
lo1_ok = [n for n in names if lo1[n] and (lo1[n].get("ok") or lo1[n].get("pass") or lo1[n].get("problems") == [])]
ck_loader = sum(1 for n in names if g(CK[n], "gates", "loader_ok"))
out["loader"] = {"whole_folder": {k: ls_.get(k) for k in ("pass", "folders", "anatomy_count", "checked", "ok", "bad",
                                                          "names_equal_folders", "fingerprints_unique",
                                                          "schedule_covers_all_in_one_pass", "rcca_len_mm", "seconds")},
                 "per_anatomy_loader_one_on_anatomies_v6": {"present": sum(1 for n in names if lo1[n]), "ok": len(lo1_ok),
                                                            "source": "/full/load_v6 (s2_build load_v6_all.sh, loader_one 1.1)"},
                 "checks_v6_loader_gate": ck_loader, "file": "/s2c/loader/loader_set_v6.json"}
inv["loader_223"] = {"pass": bool(ls_.get("pass")) and ls_.get("ok") == 223 and ck_loader == 223,
                     "detail": "whole-folder training loader %s/%s; loader_one on anatomies_v6 %d/223; checks gate %d/223" % (
                         ls_.get("ok"), ls_.get("anatomy_count"), len(lo1_ok), ck_loader)}

# ------------------------------------------------------------------ 3. SOFA RCCA (all 223) + v5 on the 25
REACH = ("past_seam1_20mm", "past_bifurcation", "reached_siphon", "reached_target")


def rcca_rec(p):
    r = jl(p)
    if not r:
        return None
    ps = r.get("per_seed") or []
    return {"passed": r.get("passed"), "exercised": r.get("rcca_exercised"), "exception": r.get("exception"),
            "branches": r.get("branches"), "loader_mesh_is_this_obj": r.get("loader_mesh_is_this_obj"),
            "insertion_inside": r.get("insertion_inside"), "nonfinite_steps": r.get("nonfinite_steps"),
            "tip_outside_max_mm": r.get("tip_outside_max_mm"), "tip_outside_gt_0.5mm_steps": r.get("tip_outside_gt_0.5mm_steps"),
            "dev_outside_max_mm": r.get("dev_outside_max_mm"), "max_rcca_s_mm": r.get("max_rcca_s_mm"),
            "route": r.get("route"), "rcca": r.get("rcca"), "target_s_mm": g(r, "target_vessel_frame", "rcca_s_mm"),
            "per_seed": [{k: s.get(k) for k in ("seed", "max_rcca_s_mm") + REACH} for s in ps],
            "seconds": r.get("seconds"), "ms_per_step": r.get("ms_per_step")}


R6 = {n: rcca_rec(os.path.join(S2C, "sofa", "rcca_v6", n + ".json")) for n in names}
R5 = {n: rcca_rec(os.path.join(S2C, "sofa", "rcca_v5", n + ".json")) for n in names}
R4 = {n: rcca_rec(os.path.join(S2C, "sofa", "rcca_v4", n + ".json")) for n in names}
done6 = [n for n in names if R6[n]]


def reach_counts(recs, ns):
    c = {"anatomies": len(ns), "seeds": 0, **{k: 0 for k in REACH}, **{"anat_any_" + k: 0 for k in REACH}}
    for n in ns:
        r = recs.get(n)
        if not r:
            continue
        for s in r["per_seed"]:
            c["seeds"] += 1
            for k in REACH:
                c[k] += bool(s.get(k))
        for k in REACH:
            c["anat_any_" + k] += any(s.get(k) for s in r["per_seed"])
    return c


rc6 = reach_counts(R6, done6)
not_passed = [n for n in done6 if not R6[n]["passed"]]
not_ex = [n for n in done6 if not R6[n]["exercised"]]
both25 = [n for n in set25 if R6.get(n) and R5.get(n)]
cmp25 = {"v6": reach_counts(R6, both25), "v5": reach_counts(R5, both25), "per_anatomy": {}}
for n in both25:
    cmp25["per_anatomy"][short(n)] = {t: merge({k: sum(bool(s.get(k)) for s in R[n]["per_seed"]) for k in REACH},
                                               {"max_s": R[n]["max_rcca_s_mm"]}) for t, R in (("v6", R6), ("v5", R5))}
fewer_target = [short(n) for n in both25 if sum(bool(s.get("reached_target")) for s in R6[n]["per_seed"])
                < sum(bool(s.get("reached_target")) for s in R5[n]["per_seed"])]
more_target = [short(n) for n in both25 if sum(bool(s.get("reached_target")) for s in R6[n]["per_seed"])
               > sum(bool(s.get("reached_target")) for s in R5[n]["per_seed"])]
cmp25["target_seeds_fewer_in_v6"] = fewer_target
cmp25["target_seeds_more_in_v6"] = more_target
cmp25["soundness"] = {t: {"passed": sum(1 for n in both25 if R[n]["passed"]),
                          "tip_outside_max_mm": st([R[n]["tip_outside_max_mm"] for n in both25], 3),
                          "dev_outside_max_mm": st([R[n]["dev_outside_max_mm"] for n in both25], 3)} for t, R in (("v6", R6), ("v5", R5))}
out["sofa_rcca"] = {
    "tool": "sofa/sofa_rcca_v6.py 1.0 (manifest), seeds 777,778,779 x 300 steps, from the training insertion; PASSED = soundness "
            "(18 branches incl. RVA/LVA, the loader's mesh is this OBJ, insertion inside, 0 exceptions, 0 non-finite steps, tip never "
            "> 0.5 mm outside the wall) over every step of every seed; reach reported beside it",
    "complete": len(done6), "passed": sum(1 for n in done6 if R6[n]["passed"]), "not_passed": not_passed,
    "exercised_past_bifurcation_any_seed": sum(1 for n in done6 if R6[n]["exercised"]), "not_exercised": not_ex,
    "reach_v6_223": rc6,
    "tip_outside_max_mm": st([R6[n]["tip_outside_max_mm"] for n in done6], 3),
    "dev_outside_max_mm": st([R6[n]["dev_outside_max_mm"] for n in done6], 3),
    "max_rcca_s_mm": st([R6[n]["max_rcca_s_mm"] for n in done6], 1),
    "seconds": st([R6[n]["seconds"] for n in done6], 1),
    "v5_comparison_set25": cmp25,
    "anatomies_zero_seeds_reaching_target": sorted(short(n) for n in done6 if not any(s.get("reached_target") for s in R6[n]["per_seed"])),
    "anatomies_zero_seeds_past_bifurcation": sorted(short(n) for n in not_ex),
    "not_exercised_references": {short(n): {t: (None if not R.get(n) else merge({k: sum(bool(s.get(k)) for s in R[n]["per_seed"]) for k in REACH}, {"max_s": R[n]["max_rcca_s_mm"], "passed": R[n]["passed"], "eca_takeoff_mm": g(R[n], "rcca", "eca_takeoff_mm"), "fork_mirror": g(R[n], "rcca", "fork_mirror")})) for t, R in (("v6", R6), ("v5", R5), ("v4", R4))} for n in not_ex},
    "per_anatomy": {n: R6[n] for n in names}}
inv["sofa_rcca_pass_223"] = {"pass": len(done6) == 223 and not not_passed,
                             "detail": "%d/%d complete, %d passed (soundness); exercised past the bifurcation on %d" % (
                                 len(done6), 223, out["sofa_rcca"]["passed"], out["sofa_rcca"]["exercised_past_bifurcation_any_seed"])}

# ------------------------------------------------------------------ 4. D14 VA entry (25 set + v4 controls), geometric probe


def d14_rec(p):
    r = jl(p)
    if not r:
        return None
    return {"passed": r.get("passed"), "exception": r.get("exception"), "d14": r.get("d14"),
            "per_seed": [{k: s.get(k) for k in ("seed", "entered_mm", "entered_20mm", "reached_donor_20mm")} for s in r.get("per_seed") or []],
            "rva_donor": g(r, "rva", "donor"), "v4_va_kept": g(r, "rva", "v4_va_kept"), "v6_junction": r.get("v6_junction"),
            "tip_outside_max_mm": r.get("tip_outside_max_mm"), "nonfinite_steps": r.get("nonfinite_steps")}


D = {t: {n: d14_rec(os.path.join(S2C, "sofa", "d14_" + t, n + ".json")) for n in set25} for t in ("v6", "v5", "v4")}


def geom_tab(p):
    r = jl(p)
    if not r:
        return None
    return {a["name"]: {sd: {"enters": g(a, "obj", sd, "enters"), "enters_any": g(a, "obj", sd, "enters_any_direction"),
                             "ante_depth_mm": g(a, "obj", sd, "antegrade", "depth_mm"),
                             "retro_depth_mm": g(a, "obj", sd, "retrograde", "depth_mm"),
                             "lumen_exit15_min_mm": g(a, "obj", sd, "va_lumen_exit_15mm_min")} for sd in ("right", "left")}
            for a in r.get("anatomies") or []}


G25 = {t: geom_tab(os.path.join(S2C, "entry_geom", "geom25_%s.json" % t)) for t in ("v6", "v5", "v4")}
G223 = {t: geom_tab(os.path.join(S2C, "entry_geom", "geom223_%s.json" % t)) for t in ("v6", "v5", "v4")}


def d14_counts(t):
    recs = [D[t][n] for n in set25 if D[t][n]]
    c = {"complete": len(recs), "passed": sum(1 for r in recs if r["passed"]), "seeds": sum(len(r["per_seed"]) for r in recs),
         "seeds_entered_20mm": sum(sum(bool(s["entered_20mm"]) for s in r["per_seed"]) for r in recs),
         "anatomies_any_entry": sum(1 for r in recs if any(s["entered_20mm"] for s in r["per_seed"])),
         "entered_mm_median_of_medians": st([g(r, "d14", "entered_mm_median") for r in recs], 1).get("median")}
    don = [r for r in recs if r["rva_donor"] and not r["v4_va_kept"]]
    c["donor_rva_anatomies"] = len(don)
    c["donor_rva_seeds_entered_20mm"] = sum(sum(bool(s["entered_20mm"]) for s in r["per_seed"]) for r in don)
    c["donor_rva_anatomies_any_entry"] = sum(1 for r in don if any(s["entered_20mm"] for s in r["per_seed"]))
    return c


def geom_counts(G):
    if not G:
        return None
    return {"anatomies": len(G), "right_enters": sum(1 for v in G.values() if v["right"]["enters"]),
            "left_enters": sum(1 for v in G.values() if v["left"]["enters"]),
            "right_enters_any_direction": sum(1 for v in G.values() if v["right"]["enters_any"]),
            "left_enters_any_direction": sum(1 for v in G.values() if v["left"]["enters_any"])}


per25 = {}
for n in set25:
    per25[short(n)] = {t: None if not D[t][n] else {"seeds_entered_20mm": sum(bool(s["entered_20mm"]) for s in D[t][n]["per_seed"]),
                                                     "entered_mm": [s["entered_mm"] for s in D[t][n]["per_seed"]],
                                                     "passed": D[t][n]["passed"], "donor": D[t][n]["rva_donor"],
                                                     "geom_R_enters": g(G25, t, n, "right", "enters")}
                       for t in ("v6", "v5", "v4")}
ch = {"v6_more_than_v5": [], "v6_fewer_than_v5": []}
for n in set25:
    if D["v6"][n] and D["v5"][n]:
        a6 = sum(bool(s["entered_20mm"]) for s in D["v6"][n]["per_seed"])
        a5 = sum(bool(s["entered_20mm"]) for s in D["v5"][n]["per_seed"])
        if a6 > a5:
            ch["v6_more_than_v5"].append([short(n), a5, a6])
        elif a6 < a5:
            ch["v6_fewer_than_v5"].append([short(n), a5, a6])
out["d14_va_entry"] = {
    "protocol": "D14 (orchestrator, 2026-10-04): sofa_va_entry 1.1 (manifest; device reset per seed) with seeds 777,778,779 x 300 "
                "steps on the v5 25-anatomy SOFA set (/sp/v5_va/checks/sofa_selection.json: 15 most changed + 10 random), the SAME "
                "anatomies and seeds on anatomies_v5 and the v4 controls (anatomies_v4); entry = the wire >= 20 mm past the VA "
                "lumen start; plus the geometric probe va_entry_geom 1.0 (straight 0.65 mm probe enters the RVA / LVA ostium >= 5 mm). "
                "Reported, NOT a gate.",
    "set25": [short(n) for n in set25],
    "sofa": {t: d14_counts(t) for t in ("v6", "v5", "v4")},
    "geometric_set25": {t: geom_counts(G25[t]) for t in ("v6", "v5", "v4")},
    "geometric_all223": {t: geom_counts(G223[t]) for t in ("v6", "v5", "v4")},
    "changes_v6_vs_v5": ch, "per_anatomy": per25,
    "v5_reference_recorded": "v5 build (sofa_rva, schedule without device reset): 10/25 anatomies entered (8/23 donor RVAs) vs v4 24/25; "
                             "not comparable seed for seed -- the v5 and v4 numbers above are re-measured with the same tool and seeds"}
geom_bad = []
if G223["v6"]:
    geom_bad = sorted("%s %s" % (short(n), sd) for n, v in G223["v6"].items() for sd in ("right", "left") if not v[sd]["enters"])
out["d14_va_entry"]["geometric_v6_sides_not_entering"] = geom_bad

# ------------------------------------------------------------------ 5. right route (D22 / D26 / D27, D13, relief)
rr = {}
for n in names:
    r = CK[n]
    d22 = g(r, "route", "d22") or {}
    rec = {"baseline_d13": g(r, "route", "d13", "baseline_used") or d22.get("baseline"),
           "rcca_prefix_identical_v4": g(r, "route", "rcca_prefix", "identical_v4"),
           "relief_ratio_info": g(r, "route", "gate_ratio"), "d26": d22.get("d26_flag"), "d27": d22.get("d27_flag"),
           "part_versions": r.get("part_versions")}
    for pc in ("fork", "siphon"):
        p = d22.get(pc) or {}
        rw = p.get("d22_repair_window") or {}
        rec[pc] = {"value": p.get("value"), "value_v4": p.get("value_v4"), "value_own": p.get("value_own"),
                   "value_before_d27": p.get("value_before_d27"), "reference": p.get("d22_reference"), "ok": p.get("ok"),
                   "d26": p.get("d26_flag"), "d27": p.get("d27_flag"), "gated_area_mm2": g(p, "gate", "area_mm2"),
                   "gate_max_dev_mm": g(p, "gate", "max_mm"),
                   "zone": {k: g(p, "zone", k) for k in ("area_mm2", "within_0.3", "max_mm", "area_gt_0.3_mm2")},
                   "handle_windows_v6": p.get("handle_windows_v6") or [], "handle_windows_v4_mapped": p.get("handle_windows_v4_mapped") or [],
                   "windows": {k: g(p, "windows", k) for k in ("area_mm2", "within_0.3", "max_mm")} if p.get("windows") else None,
                   "repair_window": {k: rw.get(k) for k in ("area_mm2", "cells", "s_range_mm", "clipped_in_v4_world_frac")} if rw else None,
                   "coverage": g(p, "coverage", "within_0.3"), "coverage_area_off_mm2": g(p, "coverage", "area_gt_0.3_mm2"),
                   "mirror": p.get("mirror_v6"), "same_surface_as_v4": p.get("same_surface")}
    rr[n] = rec
rt = {"gate_pass": {pc: sum(1 for n in names if g(CK[n], "gates", "route_%s_d22_ge_0.99" % pc)) for pc in ("fork", "siphon")},
      "rcca_prefix_unchanged": sum(1 for n in names if g(CK[n], "gates", "rcca_prefix_unchanged"))}
for pc in ("fork", "siphon"):
    rt[pc] = {"reference": dict(collections.Counter(rr[n][pc]["reference"] for n in names)),
              "value_under_reference": st([rr[n][pc]["value"] for n in names]),
              "value_v4_only": st([rr[n][pc]["value_v4"] for n in names]),
              "value_v4_only_lt_0.99": sorted([short(n), rr[n][pc]["value_v4"], rr[n][pc]["reference"]] for n in names
                                               if (rr[n][pc]["value_v4"] or 1) < 0.99),
              "coverage": st([rr[n][pc]["coverage"] for n in names]),
              "coverage_lt_0.99": sorted([short(n), rr[n][pc]["coverage"], rr[n][pc]["coverage_area_off_mm2"]] for n in names
                                         if (rr[n][pc]["coverage"] or 1) < 0.99),
              "blend_zone_reported": {"area_mm2": st([rr[n][pc]["zone"]["area_mm2"] for n in names], 1),
                                      "within_0.3": st([rr[n][pc]["zone"]["within_0.3"] for n in names]),
                                      "max_dev_mm": st([rr[n][pc]["zone"]["max_mm"] for n in names], 3)},
              "handle_windows_reported": {"anatomies_with_v6_window": sorted(short(n) for n in names if rr[n][pc]["handle_windows_v6"]),
                                          "anatomies_with_v4_window_mapped": sorted(short(n) for n in names if rr[n][pc]["handle_windows_v4_mapped"]),
                                          "anatomies_with_window_cells": sum(1 for n in names if rr[n][pc]["windows"]),
                                          "window_area_mm2": st([(rr[n][pc]["windows"] or {}).get("area_mm2") for n in names if rr[n][pc]["windows"]], 2),
                                          "window_within_0.3": st([(rr[n][pc]["windows"] or {}).get("within_0.3") for n in names if rr[n][pc]["windows"]]),
                                          "window_max_dev_mm": st([(rr[n][pc]["windows"] or {}).get("max_mm") for n in names if rr[n][pc]["windows"]], 3)},
              "repair_window_d27": sorted([short(n), rr[n][pc]["repair_window"]] for n in names if rr[n][pc]["repair_window"]),
              "relief_ratio_info": st([g(rr[n], "relief_ratio_info", pc) for n in names]),
              "relief_ratio_info_lt_0.99": sum(1 for n in names if (g(rr[n], "relief_ratio_info", pc) or 1) < 0.99),
              "mirrored_v6": sum(1 for n in names if (rr[n][pc]["mirror"] or [1])[0] < 0)}
rt["d13_baseline"] = dict(collections.Counter(rr[n]["baseline_d13"] for n in names))
rt["d26_anatomies"] = sorted(short(n) for n in names if rr[n]["fork"]["d26"] or rr[n]["siphon"]["d26"])
rt["d27_anatomies"] = sorted(short(n) for n in names if rr[n]["fork"]["d27"] or rr[n]["siphon"]["d27"])
rt["part_versions"] = dict(collections.Counter(json.dumps(rr[n]["part_versions"], sort_keys=True) for n in names))
# the spot re-run straight from anatomies_v6 (full part route, checks_v6 1.5)
spot = {}
for p in sorted(glob.glob(os.path.join(S2C, "route_spot", "case_*.json"))):
    n = os.path.basename(p)[:-5]
    r = jl(p) or {}
    a, b = g(r, "route", "d22") or {}, g(CK[n], "route", "d22") or {}
    keys = ("value", "value_v4", "value_own", "d22_reference", "d26_flag", "d27_flag", "ok")

    def _norm(p):   # 1.4 records have no d27_flag key; 1.5 writes d27_flag false on every passing piece (same code path)
        p = dict(p or {})
        p["d27_flag"] = bool(p.get("d27_flag"))
        return json.dumps(p, sort_keys=True, default=str)
    spot[short(n)] = {"error": g(r, "route", "error"),
                      "equal_to_checks4": {pc: _norm(a.get(pc)) == _norm(b.get(pc)) for pc in ("fork", "siphon")},
                      "equality_note": "the whole d22 piece record compared (every key), d27_flag absent == false",
                      "new": {pc: {k: (a.get(pc) or {}).get(k) for k in keys} for pc in ("fork", "siphon")},
                      "relief_new": g(r, "route", "gate_ratio"), "relief_checks4": g(CK[n], "route", "gate_ratio"),
                      "d13_new": g(r, "route", "d13", "baseline_used"), "d13_checks4": g(CK[n], "route", "d13", "baseline_used")}
rt["spot_rerun_from_anatomies_v6"] = spot
_dc = next((d_ for d_ in ("diag_cov12", "diag_cov11", "diag_cov") if glob.glob(os.path.join(S2C, d_, "case_*.json"))), "diag_cov")
rt["coverage_diagnosis"] = {"tool": "s2checks/diag_cov.py %s (checks_v6 1.5 route_d22 recompiled in memory with one capture; reported, not gated)"
                                    % {"diag_cov12": "1.2", "diag_cov11": "1.1"}.get(_dc, "1.0"),
                            "pieces": {os.path.basename(p)[:-5]: {k: v for k, v in (jl(p) or {}).items() if k not in ("name",)}
                                       for p in sorted(glob.glob(os.path.join(S2C, _dc, "case_*.json")))}}
rt["per_anatomy"] = {n: rr[n] for n in names}
out["right_route"] = rt
inv["right_route_d22_fork_223"] = {"pass": rt["gate_pass"]["fork"] == 223, "detail": "%d/223 (reference v4 %s, own surface %s)" % (
    rt["gate_pass"]["fork"], rt["fork"]["reference"].get("v4"), rt["fork"]["reference"].get("own_surface"))}
inv["right_route_d22_siphon_223"] = {"pass": rt["gate_pass"]["siphon"] == 223, "detail": "%d/223 (reference v4 %s, own surface %s; D27 %d)" % (
    rt["gate_pass"]["siphon"], rt["siphon"]["reference"].get("v4"), rt["siphon"]["reference"].get("own_surface"), len(rt["d27_anatomies"]))}
inv["rcca_prefix_bit_identical_223"] = {"pass": rt["rcca_prefix_unchanged"] == 223, "detail": "%d/223" % rt["rcca_prefix_unchanged"]}

# ------------------------------------------------------------------ 6. orientation, ICA AP, fork headings
O6 = jl(os.path.join(S2C, "orient", "orient_set_v6.json"), {})
O5 = jl(os.path.join(S2C, "orient", "orient_set_v5.json"), {})
ap_rec_diff = []
for n in names:
    rec_ap = g(BA, n, "compose", "ica_ap_post_mm") or g(BA, n, "compose", "carotid", "ica_ap_post_mm")
    m_ap = g(O6, "anatomies", n, "ica_ap")
    if isinstance(rec_ap, dict) and m_ap:
        for p in ("RICA", "LICA"):
            if rec_ap.get(p) is not None and (m_ap.get(p) or {}).get("post_mm") is not None:
                ap_rec_diff.append(abs(rec_ap[p] - m_ap[p]["post_mm"]))
out["orientation"] = {
    "tool": "s2checks/orient_set.py 1.0 = gates/orient_gates.py 1.2 measure_tree (manifest) on the whole tree + checks_v6 part orient's "
            "D8 / D23 post-processing; fork gates informational (D11)",
    "v6": O6.get("counts"), "v5": O5.get("counts"),
    "v6_tierA_anatomies": sorted([short(n), v["tierA_pieces"], v["d23_accepted"]] for n, v in (O6.get("anatomies") or {}).items() if v["tierA_pieces"]),
    "v6_blocking_after_D8_D23": sorted(short(n) for n, v in (O6.get("anatomies") or {}).items() if v["v6_blocking_after_D8"]),
    "v6_accepted_flags": sorted([short(n), v["accepted_flags"]] for n, v in (O6.get("anatomies") or {}).items() if v["accepted_flags"]),
    "og_counts_v6": {k: g(O6, "og_counts", k) for k in ("anatomies_tierA", "anatomies_primary", "pieces_by_flag", "v6_blocking_by_flag",
                                                         "class_anatomies", "anchor_elev", "va_tip_move_anat", "va_lumen_gap", "forks")},
    "og_counts_v5": {k: g(O5, "og_counts", k) for k in ("anatomies_tierA", "anatomies_primary", "v6_blocking_by_flag", "class_anatomies",
                                                         "va_tip_move_anat", "va_lumen_gap", "forks")},
    "files": ["/s2c/orient/orient_set_v6.json", "/s2c/orient/orient_set_v5.json"]}
out["ica_ap_and_fork_headings"] = {
    "definition": "ICA-terminus AP offset (D11 record): post_mm = placed siphon end minus its anatomical target (C_HOST + A0 (q - mid), q = the "
                  "back-mapped piece end, mid = the patient's two native VA ends), +y = posterior; flag > 40 mm. Fork heading = |yaw| of "
                  "the seam-1 fork rotation against the per-patient Zenodo map (gate value 45 deg, informational)",
    "ica_ap_gt40": {"v6": {k: g(O6, "counts", k) for k in ("ica_ap_gt40_any", "ica_ap_gt40_RICA", "ica_ap_gt40_LICA")},
                    "v5": {k: g(O5, "counts", k) for k in ("ica_ap_gt40_any", "ica_ap_gt40_RICA", "ica_ap_gt40_LICA")},
                    "v5_reference_compose": {"any": 108, "RICA": 66, "LICA": 59},
                    "v6_list": sorted([short(n), {p: (a or {}).get("post_mm") for p, a in v["ica_ap"].items()}] for n, v in (O6.get("anatomies") or {}).items()
                                      if any(((a or {}).get("post_mm") or -1e9) > 40 for a in v["ica_ap"].values()))},
    "ica_ap_post_mm": {"v6": O6.get("ica_ap_post_mm"), "v5": O5.get("ica_ap_post_mm")},
    "tree_vs_compose_record_max_abs_diff_mm": round(max(ap_rec_diff), 3) if ap_rec_diff else None,
    "tree_vs_compose_record_n": len(ap_rec_diff),
    "fork_heading": {t: merge({k: g(O, "counts", k) for k in ("lower_fork_heading_gt45", "lower_L_fork_heading_gt45", "lower_fork_tilt_gt40",
                                                              "lower_L_fork_tilt_gt40", "lower_mirrored", "lower_L_mirrored",
                                                              "lower_contralateral_effective", "lower_L_contralateral_effective",
                                                              "lower_eca_anterior", "lower_L_eca_anterior")},
                               {"abs_yaw_deg": O.get("fork_abs_yaw_deg")}) for t, O in (("v6", O6), ("v5", O5))},
    "v6_right_fork_heading_gt45": sorted([short(n), round(v["forks"]["lower"]["yaw_deg"], 1), v["forks"]["lower"]["placement"]]
                                         for n, v in (O6.get("anatomies") or {}).items()
                                         if "fork_heading" in ((v["forks"].get("lower") or {}).get("flags") or [])),
    "v6_left_fork_heading_gt45": sorted([short(n), round(v["forks"]["lower_L"]["yaw_deg"], 1), v["forks"]["lower_L"]["placement"]]
                                        for n, v in (O6.get("anatomies") or {}).items()
                                        if "fork_heading" in ((v["forks"].get("lower_L") or {}).get("flags") or []))}
oc = O6.get("counts") or {}
inv["orientation_0_tierA_except_D23"] = {"pass": oc.get("tierA_after_D23") == 0 and oc.get("blocking_after_D8_D23") == 0 and oc.get("anatomies") == 223,
                                         "detail": "tier A %s anatomy (D23 accepted) -> %s after D23; v6-blocking after D8/D23 %s" % (
                                             oc.get("tierA_any"), oc.get("tierA_after_D23"), oc.get("blocking_after_D8_D23"))}

# ------------------------------------------------------------------ 7. wall detail (VA), v6 / v5 / v4
V5C = {n: jl("/sp/v5_va/checks/out/%s.json" % n) for n in names}
wrows = []
for n in names:
    va = g(CK[n], "va") or {}
    for side in ("right", "left"):
        s6, s5 = g(va, "v6", side) or {}, g(va, "v5", side) or {}
        v4 = g(V5C[n], "item2", "va", side, "v4") or {}
        row = {"name": n, "side": side, "donor": s6.get("donor"), "donor_v5": s5.get("donor"),
               "v6_donor_real": g(s6, "donor_real", "frac_0.3mm"), "v5_donor_real": g(s5, "donor_real", "frac_0.3mm"),
               "v6_donor_stretch": g(s6, "donor_stretch", "frac_0.3mm"), "v5_donor_stretch": g(s5, "donor_stretch", "frac_0.3mm"),
               "v6_subclavian": g(s6, "subclavian", "subclavian_real_combined", "frac_0.3mm"),
               "v6_subclavian_area": g(s6, "subclavian", "subclavian_real_combined", "area_mm2"),
               "v6_topbrain": g(s6, "topbrain", "frac_0.3mm"), "v5_topbrain": g(s5, "topbrain", "frac_0.3mm"),
               "v6_whole": g(s6, "whole_from_origin", "frac_0.3mm"), "v5_whole": g(s5, "whole_from_origin", "frac_0.3mm"),
               "v4_whole": g(v4, "whole_va", "frac_0.3mm"), "v4_host_tube": g(v4, "host_tube", "frac_0.3mm"),
               "v4_topbrain": g(v4, "topbrain", "frac_0.3mm"),
               "v6_d15_outside": g(s6, "d15_split", "outside", "frac_0.3mm"), "v5_d15_outside": g(s5, "d15_split", "outside", "frac_0.3mm"),
               "v6_d15_zone": g(s6, "d15_split", "zone", "frac_0.3mm"), "v5_d15_zone": g(s5, "d15_split", "zone", "frac_0.3mm"),
               "v6_d15_zone_area": g(s6, "d15_split", "zone", "area_mm2"), "tubed_mm": s6.get("tubed_mm")}
        wrows.append(row)
don = [r for r in wrows if r["donor"]]
both = [r for r in don if r["v5_donor_real"] is not None and r["v6_donor_real"] is not None]


def ratio(a, b):
    return [x / y for x, y in zip(a, b) if x is not None and y not in (None, 0)]


wall = {"definition": "wall detail = share of the wall area > 0.3 mm off the centreline tube (frac_0.3mm), exact ownership on the "
                      "UNDECIMATED unions (checks_v6 part va: v6 /full/iso, v5 iso_v5); v4 from v5's checks (item2.va.<side>.v4)",
        "sides": len(wrows), "donor_sides": len(don), "host_sides": len(wrows) - len(don),
        "donor_real": {"v6": st([r["v6_donor_real"] for r in don]), "v5": st([r["v5_donor_real"] for r in both]),
                       "v6_over_v5": st(ratio([r["v6_donor_real"] for r in both], [r["v5_donor_real"] for r in both])),
                       "v4_host_tube_same_side": st([r["v4_host_tube"] for r in don]),
                       "v6_gt_v4_host_tube": sum(1 for r in don if (r["v6_donor_real"] or 0) > (r["v4_host_tube"] or 0))},
        "donor_stretch_as_baked": {"v6": st([r["v6_donor_stretch"] for r in don]), "v5": st([r["v5_donor_stretch"] for r in both])},
        "subclavian_stretch_v6": {"frac_0.3mm": st([r["v6_subclavian"] for r in don]), "area_mm2": st([r["v6_subclavian_area"] for r in don], 1),
                                  "sides_with_stretch": sum(1 for r in don if r["v6_subclavian"] is not None),
                                  "reference": "v5 / v4: the host's subclavian tube (v4 host tube 0.000-0.0005 on the VA side)"},
        "topbrain_piece": {"v6": st([r["v6_topbrain"] for r in wrows]), "v5": st([r["v5_topbrain"] for r in wrows]),
                           "v4": st([r["v4_topbrain"] for r in wrows])},
        "whole_va": {"v6": st([r["v6_whole"] for r in wrows]), "v5": st([r["v5_whole"] for r in wrows]), "v4": st([r["v4_whole"] for r in wrows]),
                     "v6_donor_sides": st([r["v6_whole"] for r in don]),
                     "v6_gt_v4_same_side": sum(1 for r in wrows if r["v6_whole"] is not None and r["v4_whole"] is not None and r["v6_whole"] > r["v4_whole"]),
                     "compared_v4": sum(1 for r in wrows if r["v6_whole"] is not None and r["v4_whole"] is not None),
                     "v6_over_v5": st(ratio([r["v6_whole"] for r in wrows], [r["v5_whole"] for r in wrows]))},
        "d15_split": {"outside_zones_v6": st([r["v6_d15_outside"] for r in both]), "outside_zones_v5": st([r["v5_d15_outside"] for r in both]),
                      "outside_v6_over_v5": st(ratio([r["v6_d15_outside"] for r in both], [r["v5_d15_outside"] for r in both])),
                      "zone_v6": st([r["v6_d15_zone"] for r in both]), "zone_v5": st([r["v5_d15_zone"] for r in both]),
                      "sides_with_zone": sum(1 for r in both if (r["v6_d15_zone_area"] or 0) > 0)},
        "d5_donor_real_median_ge_0.10": (st([r["v6_donor_real"] for r in don]).get("median") or 0) >= 0.10,
        "lowest_donor_real_v6": sorted([[short(r["name"]), r["side"], r["donor"], r["v6_donor_real"], r["v5_donor_real"]] for r in don],
                                       key=lambda x: x[3] if x[3] is not None else 9)[:8],
        "per_side": wrows}
out["wall_detail"] = wall

# ------------------------------------------------------------------ 8. VA polyline lumen (direct on anatomies_v6) vs v5
L6 = jl(os.path.join(S2C, "lumen", "va_lumen_poly_v6.json"), [])
L5 = jl("/repo/carotid_data/v6_work/wall/lumen/va_lumen_poly_v5.json", [])


def lum(L):
    m = {}
    for a in L or []:
        for va in ("RVA", "LVA"):
            for mesh in ("obj", "full"):
                v = g(a, va, mesh, "lumen_min_mm")
                if v is not None:
                    m[(a["name"], va, mesh)] = (v, g(a, va, mesh, "section"), g(a, va, mesh, "n_outside"))
    return m


m6, m5 = lum(L6), lum(L5)
ck_lum_diff = []
for (n, va, mesh), (v, _, _) in m6.items():
    c = g(CK.get(n) or {}, "lumen", "min_mm", va, mesh)
    if c is not None:
        ck_lum_diff.append(abs(c - v))
lumen = {"tool": "wall/va_lumen_poly.py (manifest; 0.05 mm polyline + control points, exact point-triangle distance, winding, 3 mm caps out)",
         "anatomies_v6": len({k[0] for k in m6}), "vas_obj": sum(1 for k in m6 if k[2] == "obj"),
         "obj_min_mm": st([v[0] for k, v in m6.items() if k[2] == "obj"], 3),
         "full_min_mm": st([v[0] for k, v in m6.items() if k[2] == "full"], 3),
         "obj_lt_0.65": sorted([short(k[0]), k[1], v[0], v[1]] for k, v in m6.items() if k[2] == "obj" and v[0] < 0.65),
         "full_lt_0.65": sorted([short(k[0]), k[1], v[0], v[1]] for k, v in m6.items() if k[2] == "full" and v[0] < 0.65),
         "samples_outside_obj": sum(v[2] or 0 for k, v in m6.items() if k[2] == "obj"),
         "lowest5_obj": sorted([[short(k[0]), k[1], v[0], v[1]] for k, v in m6.items() if k[2] == "obj"], key=lambda x: x[2])[:5],
         "v5": {"obj_min_mm": st([v[0] for k, v in m5.items() if k[2] == "obj"], 3),
                "obj_lt_0.65": sorted([short(k[0]), k[1], v[0], v[1]] for k, v in m5.items() if k[2] == "obj" and v[0] < 0.65)},
         "equal_to_checks4_lumen_part_max_abs_diff_mm": round(max(ck_lum_diff), 4) if ck_lum_diff else None,
         "file": "/s2c/lumen/va_lumen_poly_v6.json"}
out["va_lumen_polyline"] = lumen
inv["va_polyline_lumen_ge_0.65_every_obj"] = {"pass": lumen["vas_obj"] == 446 and not lumen["obj_lt_0.65"] and lumen["samples_outside_obj"] == 0,
                                              "detail": "%d VAs on %d OBJ, min %s mm; v5: %d below 0.65" % (
                                                  lumen["vas_obj"], lumen["anatomies_v6"], lumen["obj_min_mm"].get("min"), len(lumen["v5"]["obj_lt_0.65"]))}

# ------------------------------------------------------------------ 9. per-anatomy mesh / geometry gates (checks4)
gk = collections.Counter()
for n in names:
    for k, v in (CK[n].get("gates") or {}).items():
        gk[k] += bool(v)
i1 = {"genus0_obj": sum(1 for n in names if g(CK[n], "item1", "obj", "pml_raw", "genus") == 0),
      "closed_obj": sum(1 for n in names if g(CK[n], "item1", "obj", "pml_raw", "boundary_edges") == 0),
      "selfint_obj_full_union_0": sum(1 for n in names if all((g(CK[n], "item1", "gates", "selfint", m) or 0) == 0 for m in ("obj", "full", "union"))),
      "centrelines_inside_obj": sum(1 for n in names if g(CK[n], "item1", "gates", "centreline_outside_obj_all") == 0),
      "strict_pass": sum(1 for n in names if g(CK[n], "item1", "gates", "strict_pass"))}
kgt = []
for n in names:
    for cv, v in (g(CK[n], "geom", "kink") or {}).items():
        m = (v or {}).get("max_new_deg")
        if m is not None and m > 60.0:
            kgt.append([short(n), cv, m, v.get("max_new_form"), v.get("at_s_mm"), v.get("v5_max_deg")])
geo = {"clear_min_new_mm": st([g(CK[n], "geom", "clear_min_new_mm") for n in names], 3),
       "kink_max_new_deg": st([max([(v or {}).get("max_new_deg") or 0 for v in (g(CK[n], "geom", "kink") or {}).values()] or [0]) for n in names], 1),
       "kink_fails": sum(len(g(CK[n], "geom", "kink_fails") or []) for n in names),
       "clear_fails": sum(len(g(CK[n], "geom", "clear_fails") or []) for n in names),
       "kink_rule": "geom_v6 1.1 (manifest): a new 1 mm turn (every sampling phase) fails when > 60 deg AND sharper than the same v5 "
                    "curve's own maximum + 1 deg",
       "new_turns_gt60_not_sharper_than_v5": sorted(kgt),
       "n_new_turns_gt60_not_sharper_than_v5": len(kgt)}
out["per_anatomy_gates"] = {"source": "/full/checks4 (checks_v6 1.3 parts + route_d22 1.4 / 1.5; anatomies_v6 identical to the checked tree)",
                            "gate_pass_counts": dict(gk), "pass": sum(1 for n in names if CK[n].get("pass")),
                            "strict_audit": i1, "geom": geo}
inv["genus0_closed_0SI_centrelines_inside_223"] = {"pass": all(v == 223 for v in i1.values()), "detail": json.dumps(i1)}
inv["clearance_new_ge_0.6_223"] = {"pass": gk["geom_clearance_new_ge_0.6"] == 223, "detail": "min %s mm" % geo["clear_min_new_mm"].get("min")}
inv["joins_1mm_le_60_every_phase_223"] = {"pass": gk["geom_kink_new_le_60"] == 223,
                                          "detail": "%d/223 under geom_v6's rule (> 60 fails only if sharper than v5's same curve); %d new "
                                                    "turns > 60 no sharper than v5 (listed, max %s deg)" % (
                                              gk["geom_kink_new_le_60"], len(kgt), geo["kink_max_new_deg"].get("max"))}

# ------------------------------------------------------------------ 10. flags
fl = {k[5:]: v for k, v in (B.get("counts") or {}).items() if k.startswith("flag:")}
prov_flags = collections.Counter()
for n in names:
    p = jl(os.path.join(A6, n, "provenance.json")) or {}
    for f in (g(p, "v6_checks", "flags") or []):
        prov_flags[f if isinstance(f, str) else json.dumps(f)] += 1
new_flags = {"sofa_rcca_not_exercised_past_bifurcation": len(not_ex), "sofa_rcca_no_seed_reaches_target":
             len(out["sofa_rcca"]["anatomies_zero_seeds_reaching_target"]),
             "geometric_va_entry_not_entering_v6_sides": len(geom_bad),
             "d14_fewer_entering_seeds_than_v5_set25": len(ch["v6_fewer_than_v5"])}
out["flags"] = {"build_record_counts": fl, "provenance_v6_checks_flags": dict(prov_flags),
                "prov_stamp_flag_counts": g(B, "prov_stamp", "flag_counts"), "set_check_flags_new": new_flags,
                "note": "build_record_counts = results/v6_build.json counts (flag:*), per anatomy in v6_build.json; provenance_v6_checks_flags = "
                        "the v6_checks key stamped into each anatomies_v6 provenance.json"}

out["invariants"] = inv
out["all_invariants_pass"] = all(v["pass"] for v in inv.values())
od = sys.argv[1]
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "v6_checks.json.tmp"), "w"), indent=1, default=float)
os.replace(os.path.join(od, "v6_checks.json.tmp"), os.path.join(od, "v6_checks.json"))

# ------------------------------------------------------------------ the final summary
summ = {"version": VERSION, "date": out["date"], "tree": out["tree"], "n": 223,
        "verdict": "ALL INVARIANTS PASS" if out["all_invariants_pass"] else "INVARIANT FAILURE(S): " + ", ".join(k for k, v in inv.items() if not v["pass"]),
        "invariants": {k: [v["pass"], v["detail"]] for k, v in inv.items()},
        "sofa_rcca": merge({k: out["sofa_rcca"][k] for k in ("complete", "passed", "not_passed", "exercised_past_bifurcation_any_seed", "not_exercised")},
                           {"reach_seeds_v6_223": rc6, "set25_v6_vs_v5": {"v6": cmp25["v6"], "v5": cmp25["v5"]},
                            "no_seed_reaches_target": out["sofa_rcca"]["anatomies_zero_seeds_reaching_target"],
                            "not_exercised_references": out["sofa_rcca"]["not_exercised_references"]}),
        "d14_va_entry_set25": merge({t: out["d14_va_entry"]["sofa"][t] for t in ("v6", "v5", "v4")},
                                    {"geometric_set25": out["d14_va_entry"]["geometric_set25"],
                                     "geometric_all223": out["d14_va_entry"]["geometric_all223"], "changes_v6_vs_v5": ch}),
        "right_route": {"gate_pass": rt["gate_pass"], "reference": {pc: rt[pc]["reference"] for pc in ("fork", "siphon")},
                        "value_min": {pc: rt[pc]["value_under_reference"].get("min") for pc in ("fork", "siphon")},
                        "value_v4_only_min": {pc: rt[pc]["value_v4_only"].get("min") for pc in ("fork", "siphon")},
                        "d26": rt["d26_anatomies"], "d27": rt["d27_anatomies"], "d13_baseline": rt["d13_baseline"],
                        "coverage_lt_0.99": {pc: rt[pc]["coverage_lt_0.99"] for pc in ("fork", "siphon")},
                        "relief_ratio_info_lt_0.99": {pc: rt[pc]["relief_ratio_info_lt_0.99"] for pc in ("fork", "siphon")},
                        "spot_rerun_from_anatomies_v6": {k: v["equal_to_checks4"] for k, v in spot.items()}},
        "ica_ap_gt40": out["ica_ap_and_fork_headings"]["ica_ap_gt40"],
        "fork_heading_gt45": {t: {"right": g(out, "ica_ap_and_fork_headings", "fork_heading", t, "lower_fork_heading_gt45"),
                                  "left": g(out, "ica_ap_and_fork_headings", "fork_heading", t, "lower_L_fork_heading_gt45")} for t in ("v6", "v5")},
        "orientation": {"v6": {k: oc.get(k) for k in ("tierA_any", "tierA_after_D23", "blocking_after_D8_D23", "va_target_20_30_flag")},
                        "v5": {k: (O5.get("counts") or {}).get(k) for k in ("tierA_any", "blocking_any")}},
        "wall_detail_median": {"donor_real": {"v6": wall["donor_real"]["v6"].get("median"), "v5": wall["donor_real"]["v5"].get("median"),
                                              "v4_host_tube": wall["donor_real"]["v4_host_tube_same_side"].get("median")},
                               "subclavian_v6": wall["subclavian_stretch_v6"]["frac_0.3mm"].get("median"),
                               "whole_va": {t: wall["whole_va"][t].get("median") for t in ("v6", "v5", "v4")},
                               "topbrain": {t: wall["topbrain_piece"][t].get("median") for t in ("v6", "v5", "v4")},
                               "d15_outside_v6_over_v5_median": wall["d15_split"]["outside_v6_over_v5"].get("median")},
        "va_lumen_polyline": {"obj_min_mm": lumen["obj_min_mm"].get("min"), "obj_lt_0.65": lumen["obj_lt_0.65"],
                              "v5_obj_lt_0.65": len(lumen["v5"]["obj_lt_0.65"])},
        "split": {"exit": sp_exit, "compared": bl.get("compared"), "lowered": bl.get("lowered"), "raised": [[r.get("defn"), r.get("quantity"), r.get("field"), r.get("baseline"), r.get("now")] for r in raised]},
        "loader": out["loader"]["whole_folder"],
        "flags_build_record": fl, "flags_set_checks": new_flags,
        "files": {"v6_checks": "/sp/v6/results/v6_checks.json", "work": "carotid_data/v6_work/s2checks", "notes": "/sp/v6/notes/s2_checks.md"}}
json.dump(summ, open(os.path.join(od, "v6_final_summary.json.tmp"), "w"), indent=1, default=float)
os.replace(os.path.join(od, "v6_final_summary.json.tmp"), os.path.join(od, "v6_final_summary.json"))
print(json.dumps({"verdict": summ["verdict"], "invariants": summ["invariants"]}, indent=1, default=float))
