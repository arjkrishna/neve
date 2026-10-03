#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\v5_assign_report.py (= /sp/v5_va/v5_assign_work/v5_assign_report.py in the build containers)
"""v5_assign: the final assignment record /sp/v5_va/results/v5_assignment.json, built from the plan, the assignment
summary, the written anatomies_v5 provenance, the donor records, the dry-run stats, the split check and the
sensitivity runs. Prints the patient table."""
import json
import os
import statistics
import time
from collections import Counter, defaultdict

RES = "/sp/v5_va/results"
REG = "/repo/carotid_data/va_sources/v5_reg"
V5 = "/repo/carotid_data/anatomies_v5"


def J(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


summ = J(os.path.join(RES, "graft_reg_assign_v5.json"))
plan = J(os.path.join(REG, "plan_v5.json"))
el = J(os.path.join(RES, "v5_eligible_final.json"))
dry = J(os.path.join(RES, "v5_assign_dry_stats.json"))
recs = J(os.path.join(RES, "v5_assign_records.json"))
wchk = J(os.path.join(RES, "v5_assign_written_check.json"))
altA = J(os.path.join(RES, "v5_assign_alt_all_served.json"))
alto = J(os.path.join(RES, "v5_assign_alt_orders.json"))
sp = os.path.join(RES, "split_v5.json")
split = J(sp) if os.path.isfile(sp) else None
pairs = {}
for p in el["primary"] + el["reserve"]:
    if p["source"] == "vmr0248":
        u = "vmr0248"
    else:
        u = p["case"] if p["case"].startswith(p["source"]) else "%s_%s" % (p["source"], p["case"])
    pairs[u] = p
prov = {n: J(os.path.join(V5, n, "provenance.json")) for n in sorted(os.listdir(V5)) if os.path.isdir(os.path.join(V5, n))}
by_pat = defaultdict(list)
for n, p in prov.items():
    by_pat[p["patient"]].append(n)
unit_of = {t["patient"]: t["unit"] for t in summ["assignment"]}
rows, flagged = [], []
tk_np, tk_d, stubs, cal2, adv = 0, [], [], 0, Counter()
for P in sorted(by_pat, key=lambda q: (-len(by_pat[q]), q)):
    names = by_pat[P]
    u = unit_of.get(P)
    pr = pairs.get(u, {})
    co = pr.get("coarseness") or {}
    r = {"patient": P, "anatomies": len(names), "native": sum(prov[n]["handedness"] == "native" for n in names),
         "mirrored": sum(prov[n]["handedness"] == "mirrored" for n in names), "unit": u, "class": pr.get("class"),
         "source": pr.get("source"), "case": pr.get("case"), "patient_key": pr.get("patient_key"),
         "coarseness_contact_free_R_L": co.get("R_L_contact_free"),
         "coarseness_mean_contact_free": co.get("score_mean_contact_free"),
         "rva_replaced": 0, "lva_replaced": 0, "both": 0, "one": 0, "none": 0, "takeoff_not_preserved": 0,
         "rva_from_donor_side": Counter(), "lva_from_donor_side": Counter(),
         "non_commercial": pr.get("non_commercial"), "licence": pr.get("licence"),
         "provenance_flags": pr.get("provenance_flags") or [], "pair_flags": pr.get("flags") or [],
         "scanner": pr.get("scanner")}
    for n in names:
        v = prov[n]["v5_va"]
        s = v["summary"]
        k = sum(s["replaced"].values())
        r["both" if k == 2 else ("one" if k == 1 else "none")] += 1
        for side, key in (("right", "rva"), ("left", "lva")):
            if s["replaced"][side]:
                r[key + "_replaced"] += 1
                d = v[side]
                r[key + "_from_donor_side"][d["donor_side"] + (" mirrored" if s["donor"][side]["mirrored"] else "")] += 1
                if d.get("takeoff_not_preserved"):
                    r["takeoff_not_preserved"] += 1
                    tk_np += 1
                tk_d.append(abs(d["takeoff_delta_10mm_deg"]))
                stubs.append(d["stub_mm"])
                cal2 += d.get("splice_calibre_pass") == 2
                adv[str(d.get("ostium_advance_mm"))] += 1
        if s["flags"]:
            flagged.append({"anatomy": n, "patient": P, "unit": u, "v4_va_kept": s["v4_va_kept"], "flags": s["flags"]})
    r["rva_from_donor_side"] = dict(r["rva_from_donor_side"])
    r["lva_from_donor_side"] = dict(r["lva_from_donor_side"])
    rows.append(r)
tot = {"rva_replaced": sum(r["rva_replaced"] for r in rows), "lva_replaced": sum(r["lva_replaced"] for r in rows),
       "anatomies_both": sum(r["both"] for r in rows), "anatomies_one": sum(r["one"] for r in rows),
       "anatomies_none": sum(r["none"] for r in rows), "anatomies": len(prov),
       "anatomies_one_rva_only": sum(1 for f in flagged if not f["v4_va_kept"]["right"] and f["v4_va_kept"]["left"]),
       "anatomies_one_lva_only": sum(1 for f in flagged if f["v4_va_kept"]["right"] and not f["v4_va_kept"]["left"]),
       "non_commercial_anatomies": sum(1 for p in prov.values() if p["v5_va"]["summary"]["non_commercial"]),
       "vmr_anatomies_with_licence_file": sum(1 for n in prov if os.path.isfile(os.path.join(V5, n, "LICENSE_VMR_0248.txt"))),
       "takeoff_not_preserved_vas": tk_np, "takeoff_delta_abs_median_deg": round(statistics.median(tk_d), 1),
       "stub_mm_median": round(statistics.median(stubs), 2), "stub_mm_max": round(max(stubs), 2),
       "calibre_pass2_vas": cal2, "ostium_advance_mm": dict(adv)}
split_sum = None
if split:
    defs = {}
    for d, R in split["defs"].items():
        ed = R.get("extra_donors") or {}
        defs[d] = {"roster_v3": {k: R["roster_v3"].get(k) for k in ("train", "test", "excluded")},
                   "max": {k: {f: m.get(f) for f in ("train", "test", "kept", "status")} for k, m in (R.get("max") or {}).items()},
                   "extra_donors": {k: ed.get(k) for k in ("n", "n_spanning_more_than_one_topbrain_patient",
                                                           "topbrain_patients_with_extra_donor", "one_to_one_nested")}}
    split_sum = {"defs": defs, "lowered": (split.get("baseline") or {}).get("lowered"),
                 "compared": (split.get("baseline") or {}).get("compared"), "checks": split.get("checks")}
out = {
    "what": "v5 donor assignment (v5_assign, 2026-10-02): strict one-to-one donor pair <-> TopBrain patient, and the "
            "unbaked anatomies_v5 written from it (strategy B; graft_v5_reg 1.1)",
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "inputs": {"eligible": os.path.join(RES, "v5_eligible_final.json"), "donors": os.path.join(REG, "donors_v5"),
               "dry": os.path.join(REG, "dry_v5"), "plan": os.path.join(REG, "plan_v5.json"),
               "assignment_summary": os.path.join(RES, "graft_reg_assign_v5.json"), "anatomies": V5,
               "records_check": os.path.join(RES, "v5_assign_records.json"),
               "dry_stats": os.path.join(RES, "v5_assign_dry_stats.json"),
               "written_check": os.path.join(RES, "v5_assign_written_check.json"), "split": sp},
    "objective": {
        "solver": "exact lexicographic milp (graft_v5_assign 1.1 solve_lex), one unit per patient, one patient per unit",
        "order": ["0 patients served by a PRIMARY pair (>= 1 VA passing); reserves only for patients no one-to-one "
                  "assignment of the primaries can serve",
                  "(a) anatomies with both VAs passing every gate", "VAs passing", "RVAs passing (training-reachable side)",
                  "(b) sum over passing VAs of the donor side's contact-free coarseness (tally): coarser pairs to the "
                  "patients with more anatomies",
                  "(c) sum over passing VAs of (2 - flag level): unflagged before flagged", "take-off preserved (tie-break)"],
        "optima": summ["lexicographic"]["optima"],
        "summed_weight_hungarian_gives_same_assignment": summ["lexicographic"]["hungarian_summed_weight_same_assignment"]},
    "counts": {"units": len(pairs), "primary": sum(p["class"] == "primary" for p in pairs.values()),
               "reserve": sum(p["class"] == "reserve" for p in pairs.values()), "patients": len(by_pat),
               "assigned": len(summ["assignment"]), "units_unassigned": summ.get("units_unassigned"),
               "reserves_used": (summ.get("reserve") or {}).get("used"),
               "patients_unserved_by_primaries": (summ.get("reserve") or {}).get("patients_unserved_by_primaries"),
               "verify_log": summ["verify_log"], **tot},
    "table": rows,
    "flagged_anatomies_v4_va_kept": flagged,
    "dry_run": {"unit_anatomy_sides": dry["unit_anatomy_sides"], "pass": dry["pass"], "pass_rate": dry["pass_rate"],
                "base_gate_pass_rate": dry["base_gate_pass_rate"], "errors": dry["errors"],
                "per_unit": {u: dict({"class": r["class"], "both_unit_dry": r["anatomies_both_pass_unit_dry"]},
                                     **{s: {k: r[s][k] for k in ("pass", "candidates", "as_RVA", "as_LVA", "base_all_pass",
                                                                 "dominant_failing_gate", "calibre_pass2",
                                                                 "takeoff_not_preserved", "takeoff_delta_abs_median",
                                                                 "ostium_advance", "stub_mm_max")}
                                        for s in ("R", "L")}) for u, r in dry["per_unit"].items()}},
    "records": {"sides": recs["n_sides"], "all_checks_pass": recs["n_sides_all_checks_pass"], "failures": recs["failures"],
                "duplicate_patient_keys": recs["duplicate_patient_keys"]},
    "written": {k: wchk[k] for k in ("n_v4", "n_v5", "names_equal", "n_ok", "bad", "units_on_several_patients",
                                     "replaced_right", "replaced_left")},
    "split": split_sum,
    "alternatives": {
        "A_lex_over_primaries_then_reserve": {
            "optima_primaries": altA["optima_A_primaries"], "reserve_used": altA["A_reserve"],
            "totals_vs_chosen": "both 209 vs 208, VAs 432 vs 431, RVAs 213 vs 212; uses reserve ia172_172 (its R, mirrored, "
                                "as the LVA of case_w_035_left__topcow_mr_015_L) while primary ia172_039 can serve "
                                "topcow_mr_015 (it then leaves topcow_mr_025 to ia172_069, LVAs only)",
            "files": ["/repo/carotid_data/va_sources/v5_reg/plan_v5_A_reserve_on_mr015.json",
                      os.path.join(RES, "graft_reg_assign_v5_A_reserve_on_mr015.json")]},
        "orders": {k: {"order": v["order"], "totals": v["totals"], "units_moved": v["n_units_moved_vs_chosen"]}
                   for k, v in alto.items() if k != "what"}},
}
with open(os.path.join(RES, "v5_assignment.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("totals", json.dumps(tot))
print("%-14s %3s %3s %3s  %-18s %-3s  %6s %6s  RVA    LVA    both one none tknp" % (
    "patient", "n", "nat", "mir", "unit", "cls", "cfR", "cfL"))
for r in rows:
    cf = r["coarseness_contact_free_R_L"] or [None, None]
    print("%-14s %3d %3d %3d  %-18s %-3s  %6s %6s  %2d/%2d  %2d/%2d  %3d %3d %3d  %3d" % (
        r["patient"], r["anatomies"], r["native"], r["mirrored"], r["unit"], (r["class"] or "")[:3], cf[0], cf[1],
        r["rva_replaced"], r["anatomies"], r["lva_replaced"], r["anatomies"], r["both"], r["one"], r["none"],
        r["takeoff_not_preserved"]))
