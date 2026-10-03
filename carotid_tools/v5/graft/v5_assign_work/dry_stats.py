#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\dry_stats.py (= /sp/v5_va/v5_assign_work/dry_stats.py in the build containers)
"""v5_assign: per-unit / per-side / per-gate statistics of the dry run v5_reg/dry_v5 (27 units x 223 anatomies).
Output /sp/v5_va/results/v5_assign_dry_stats.json + table on stdout."""
import glob, json, os
from collections import Counter, defaultdict
DRY = "/repo/carotid_data/va_sources/v5_reg/dry_v5"
DON = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
GATES = ("rotation_proper", "origin_turn", "course_rms", "takeoff_band", "rise", "splice_window", "calibre",
         "join_kink", "tip_move", "clearance")
units = sorted(d for d in os.listdir(DRY) if os.path.isfile(os.path.join(DRY, d, "_done.json")))
out, agg_base, agg_n = {}, Counter(), 0
agg_final = Counter()
errors = []
for u in units:
    rec = json.load(open(os.path.join(DON, u, "record.json")))
    per = {s: {"candidates": 0, "pass": 0, "base_all_pass": 0, "base_gate_pass": Counter(), "dominant_fail": Counter(),
               "by_host_side": Counter(), "calibre_pass2": 0, "takeoff_not_preserved": 0, "ostium_advance": Counter(),
               "takeoff_delta_abs": [], "stub_mm": [], "final_fail": 0} for s in ("R", "L")}
    n_anat, both = 0, 0
    for f in sorted(glob.glob(os.path.join(DRY, u, "case_*.json"))):
        d = json.load(open(f))
        n_anat += 1
        if "error" in d:
            errors.append((u, os.path.basename(f), d["error"]))
            continue
        ok_both = True
        for hs in ("right", "left"):
            x = d["sides"][hs]
            ds = x.get("donor_side")
            if ds is None:
                ok_both = False
                continue
            p = per[ds]
            p["candidates"] += 1
            b = x.get("base") or {}
            if b.get("all_pass"):
                p["base_all_pass"] += 1
            for g in GATES:
                if g in b and b[g].get("pass"):
                    p["base_gate_pass"][g] += 1
                    agg_base[g] += 1
            agg_n += 1
            if x.get("pass"):
                p["pass"] += 1
                p["by_host_side"][hs] += 1
                agg_final["pass"] += 1
                if x.get("splice_calibre_pass") == 2:
                    p["calibre_pass2"] += 1
                if x.get("takeoff_not_preserved"):
                    p["takeoff_not_preserved"] += 1
                p["ostium_advance"][str(x.get("ostium_advance_mm"))] += 1
                p["takeoff_delta_abs"].append(abs(x.get("takeoff_delta_10mm_deg", 0.0)))
                p["stub_mm"].append(x.get("stub_mm"))
            else:
                ok_both = False
                if x.get("replaced") and x.get("final_fail"):
                    p["final_fail"] += 1
                    p["dominant_fail"]["final clearance"] += 1
                else:
                    ff = {k: v for k, v in (x.get("first_fail_counts") or {}).items() if k != "origin_turn_ostium"}
                    p["dominant_fail"][max(ff, key=ff.get) if ff else "none evaluated"] += 1
        both += ok_both
    row = {"anatomies": n_anat, "anatomies_both_pass_unit_dry": both}
    for s in ("R", "L"):
        p = per[s]
        td = sorted(p["takeoff_delta_abs"])
        sm = sorted(v for v in p["stub_mm"] if v is not None)
        row[s] = {"candidates": p["candidates"], "pass": p["pass"], "pass_rate": round(p["pass"] / p["candidates"], 3) if p["candidates"] else None,
                  "as_RVA": p["by_host_side"]["right"], "as_LVA": p["by_host_side"]["left"],
                  "base_all_pass": p["base_all_pass"],
                  "base_gate_pass_rate": {g: round(p["base_gate_pass"][g] / p["candidates"], 3) for g in GATES} if p["candidates"] else {},
                  "dominant_failing_gate": dict(p["dominant_fail"].most_common()),
                  "calibre_pass2": p["calibre_pass2"], "takeoff_not_preserved": p["takeoff_not_preserved"],
                  "takeoff_delta_abs_median": round(td[len(td) // 2], 1) if td else None,
                  "ostium_advance": dict(p["ostium_advance"]), "stub_mm_max": round(sm[-1], 2) if sm else None,
                  "stub_mm_median": round(sm[len(sm) // 2], 2) if sm else None,
                  "flag_level": rec["sides"][s].get("flag_level"), "score": rec["sides"][s].get("coarseness_score")}
    row["class"] = rec.get("eligible_class")
    out[u] = row
res = {"what": "v5_assign dry run (graft_v5_reg 1.1, donors_v5) per unit / side / gate", "dry": DRY, "units": len(units),
       "unit_anatomy_sides": agg_n, "pass": agg_final["pass"], "pass_rate": round(agg_final["pass"] / agg_n, 4) if agg_n else None,
       "base_gate_pass_rate": {g: round(agg_base[g] / agg_n, 4) for g in GATES}, "errors": errors, "per_unit": out}
json.dump(res, open("/sp/v5_va/results/v5_assign_dry_stats.json", "w"), indent=1)
print("units %d, unit-anatomy-sides %d, pass %d (%.3f), errors %d" % (len(units), agg_n, agg_final["pass"], res["pass_rate"] or 0, len(errors)))
print("base per-gate pass:", res["base_gate_pass_rate"])
for u, r in sorted(out.items(), key=lambda t: (t[1]["class"] != "primary", t[0])):
    for s in ("R", "L"):
        x = r[s]
        print("%-18s %-3s %s %3d/%3d (RVA %3d LVA %3d) base %3d cal2 %3d tknp %3d |dtk| %5s adv %s stub max %s fails %s" % (
            u, (r["class"] or "")[:3], s, x["pass"], x["candidates"], x["as_RVA"], x["as_LVA"], x["base_all_pass"], x["calibre_pass2"],
            x["takeoff_not_preserved"], x["takeoff_delta_abs_median"], x["ostium_advance"], x["stub_mm_max"],
            dict(list(x["dominant_failing_gate"].items())[:4])))
    print("%-18s     both (unit dry) %d / %d" % (u, r["anatomies_both_pass_unit_dry"], r["anatomies"]))
