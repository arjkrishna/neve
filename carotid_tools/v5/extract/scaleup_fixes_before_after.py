#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_before_after.py (= /sp/v5_va/scaleup_fixes_before_after.py in the build containers)
"""scaleup_fixes_before_after.py -- (scaleup_fixes, 2026-10-02) BEFORE (pilot) / AFTER (pipeline v2) record of the two pre-bulk
fixes on every pilot case. Read-only on every input; writes results/scaleup_fixes_before_after.json (+ console table).

Inputs: results/pilot_proc_<src>_grade.json (pilot rows, wall ct_seg4), results/scaleup_fixes_cases_<src>_grade.json (the same
cases under v2: qc/select_v2.json, V1 rescue, wall ct_seg/ct_seg4v2 with the pinch fix), <der>/qc/v1rescue.json,
<der>/qc/select_v2.json, results/scaleup_fixes_ctseg_compare.json (label-level pinch check, PILOT seeds).
Per side: QC status / failures / review, trace set, V1 bridges, wall gates, label s-range, donor-stretch union > 0.3 mm
(all bins = the exact donor stretch, else the 10 mm bins; contact-free), whole-VA union > 0.3 mm, pair eligibility.
Groups: "v1_rescue" (the rescue applied to the side), "pinch" (the pinch fix triggered on the side's wall), "unchanged".
any python3 (json only)."""
import json
import os

RES = "/sp/v5_va/results"
DER = "/repo/carotid_data/va_sources/derived/thin_slice"


def L(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def view(r):
    if r is None:
        return None
    w = r.get("wall") or {}
    don = g(w, "donor_stretch_origin_to_C1m25", "0.3mm")
    return {"trace_set": r.get("trace_set") or "tsba", "qc_status": g(r, "qc_recalibrated", "status"),
            "qc_pass": bool(g(r, "qc_recalibrated", "passes")), "qc_failed": g(r, "qc_recalibrated", "failed"),
            "qc_review": g(r, "qc_recalibrated", "review"),
            "provisional_only": bool(g(r, "qc_recalibrated", "passes")) and not bool(g(r, "qc_two_patient_backed_only", "passes")),
            "has_wall": bool(w), "wall_valid": bool(w.get("gates_V1_V6_whole")) if w else False,
            "label_s_range_mm": g(w, "label_truncation", "label_s_range_mm"),
            "wall_truncated": bool(g(w, "label_truncation", "truncated")),
            "donor_u03_all_bins": don if don is not None else g(w, "contact_free", "union_0.3_all_bins"),
            "donor_u03_contact_free": g(w, "contact_free", "union_0.3_contact_free"),
            "whole_u03": g(w, "whole", "0.3mm"), "length_trace_mm": r.get("length_trace_mm")}


cmp_rows = {(r["source"], r["case"], r["side"]): r for r in (L(RES + "/scaleup_fixes_ctseg_compare.json") or {}).get("rows", [])}
out = {"what": __doc__.split("\n\n")[0], "sides": [], "pairs": []}
for src in ("ia172", "song"):
    P = {(r["case"], r["side"]): r for r in (L(RES + "/pilot_proc_%s_grade.json" % src) or {}).get("rows", [])}
    V = {(r["case"], r["side"]): r for r in (L(RES + "/scaleup_fixes_cases_%s_grade.json" % src) or {}).get("rows", [])}
    cases = sorted({c for c, _ in P} | {c for c, _ in V})
    for c in cases:
        resc = L(os.path.join(DER, src, c, "qc", "v1rescue.json")) or {}
        sel = L(os.path.join(DER, src, c, "qc", "select_v2.json")) or {}
        pair = {"source": src, "case": c}
        for s in ("R", "L"):
            b, a = view(P.get((c, s))), view(V.get((c, s)))
            rs = (resc.get("sides") or {}).get(s) or {}
            cr = cmp_rows.get((src, c, s)) or {}
            grp = ("v1_rescue" if rs.get("applies") else "pinch" if cr.get("pinch_triggered") else "unchanged")
            rec = {"source": src, "case": c, "side": s, "group": grp, "before_pilot": b, "after_v2": a,
                   "v2_graded": a is not None,
                   "v1_rescue": {"applies": rs.get("applies"), "why": rs.get("why"), "rescued": rs.get("rescued"),
                                 "bridges": [{k: x.get(k) for k in ("s_mm", "length_mm", "HU_median_native", "ratio_to_CCA",
                                                                    "span_mm", "capsule_radius_mm", "connected", "added_voxels",
                                                                    "added_HU_median", "cross_section_r_eq_min_mm", "verdict")}
                                             for x in rs.get("bridges_rescue") or []] or rs.get("v1_bridges"),
                                 "select_v2": g(sel, "sides", s)} if rs else None,
                   "pinch_label_check_pilot_seeds": {k: cr.get(k) for k in ("pinch_triggered", "identical", "dice",
                                                                             "label_s_range_pilot", "label_s_range_v2")} if cr else None}
            if b and a:
                ch = []
                for k in ("trace_set", "qc_status", "qc_pass", "wall_valid", "label_s_range_mm", "donor_u03_all_bins",
                          "donor_u03_contact_free", "whole_u03"):
                    if b.get(k) != a.get(k):
                        ch.append(k)
                rec["changed_fields"] = ch
            out["sides"].append(rec)
            pair[s] = rec
        def elig(v):
            return bool(v and v["qc_pass"] and v["wall_valid"])
        pb = elig(pair["R"]["before_pilot"]) and elig(pair["L"]["before_pilot"])
        pa = (elig(pair["R"]["after_v2"]) and elig(pair["L"]["after_v2"])) if pair["R"]["v2_graded"] and pair["L"]["v2_graded"] else None
        out["pairs"].append({"source": src, "case": c, "eligible_pilot": pb, "eligible_v2": pa,
                             "groups": [pair["R"]["group"], pair["L"]["group"]]})
cnt = {"sides": len(out["sides"]),
       "v1_rescue_applied": [{"case": r["case"], "side": r["side"], "rescued": (r["v1_rescue"] or {}).get("rescued"),
                              "used": g(r, "v1_rescue", "select_v2", "v1_rescue", "used"),
                              "why": g(r, "v1_rescue", "select_v2", "v1_rescue", "why"),
                              "qc_before": (r["before_pilot"] or {}).get("qc_status"),
                              "qc_after": (r["after_v2"] or {}).get("qc_status")}
                             for r in out["sides"] if r["group"] == "v1_rescue"],
       "pairs_eligible_pilot": sum(p["eligible_pilot"] for p in out["pairs"]),
       "pairs_eligible_v2": sum(bool(p["eligible_v2"]) for p in out["pairs"]),
       "pairs_not_v2_graded": [p["case"] for p in out["pairs"] if p["eligible_v2"] is None],
       "newly_eligible": [(p["source"], p["case"]) for p in out["pairs"] if p["eligible_v2"] and not p["eligible_pilot"]],
       "lost": [(p["source"], p["case"]) for p in out["pairs"] if p["eligible_pilot"] and p["eligible_v2"] is False],
       "unchanged_sides_with_changed_fields": [(r["case"], r["side"], r.get("changed_fields")) for r in out["sides"]
                                               if r["group"] == "unchanged" and r.get("changed_fields")]}
out["counts"] = cnt
json.dump(out, open(RES + "/scaleup_fixes_before_after.json", "w"), indent=1, default=str)
print(json.dumps(cnt, indent=1, default=str))
for r in out["sides"]:
    b, a = r["before_pilot"] or {}, r["after_v2"] or {}
    if r["group"] == "unchanged" and not r.get("changed_fields"):
        continue
    print("%-6s %-12s %s %-9s | pilot %s %-12s wall %s s %s all %s cf %s | v2 %s %-12s wall %s s %s all %s cf %s | changed %s" % (
        r["source"], r["case"], r["side"], r["group"], b.get("trace_set"), b.get("qc_status"), b.get("wall_valid"),
        b.get("label_s_range_mm"), b.get("donor_u03_all_bins"), b.get("donor_u03_contact_free"),
        a.get("trace_set"), a.get("qc_status"), a.get("wall_valid"), a.get("label_s_range_mm"), a.get("donor_u03_all_bins"),
        a.get("donor_u03_contact_free"), r.get("changed_fields")))
