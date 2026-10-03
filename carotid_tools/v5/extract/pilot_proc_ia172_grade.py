#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_grade.py (= /sp/v5_va/pilot_proc_ia172_grade.py in the build containers)
"""pilot_proc_ia172_grade.py -- per case and side GRADE + donor-ranking table for the 172-series pilot (fourth instance of
pilot_proc_ia172, 2026-10-02). NEW file; reads only (results/pilot_proc_ia172.json from pilot_proc_ia172_collect.py, and
per walled side <der>/ct_seg/ct_seg4/{measure,edge,protrusion}.json). Writes results/pilot_proc_ia172_grade.json + .txt.
eve-training-fixed or neve-build-stagea (json + numpy).

Grading conventions (the pilot brief): native thresholds only (W1 0.0966 / W2 0.0204 / W3 0.1416 -- these are SET-median
bars; per-side flags below are only "this side alone would reach the bar"), acquisition voxel (measure.json), exact
ownership (k 64; measure.json name ends "(k64)"), seam zone separate (C1 - 25 mm -> C1, landmark set "rec" = Erec / C1f),
W3 read only together with W1, W7 by hand (distal 0-40 / 40-80 mm >= 0.0785 / 0.2183; whole-VA only -- strategy B discards
the donor above the V3 splice). The va_standard grade block in the record ('grade', W1 min 0.0159 etc.) carries the
grader's voxel factors (wrong-signed per the feasibility report) and is NOT used.

QC verdicts under both gates: recalibrated (primary; tag_calibration.apply_gate 'recalibrated' + the brief's HARD checks)
and the study draft (+ the report's additions A1 V3 loop 15 / 5 on the study cord, A2 C1 threaded), with the reasons where
they differ.

Donor-ranking quantity (the user's rule: coarsest QC-passing pairs, relief from vein / carotid / bone contamination
excluded, ranked within source): the DONOR STRETCH = origin -> (C1f - 25 mm) on the ct_seg4 centreline (the stretch v5
grafts below the seam zone; rec landmarks), union > 0.3 mm over the 10 mm bins WITHOUT a bright-neighbour contact
(< 5 % of the bin's ct_seg4 edge rays kind 3 'upper threshold' + kind 1 'dip before bright' -- the collect rule
relief_vs_contacts, whose contacts are bone in the foramina, enhanced veins / other arteries; class counts reported),
area-weighted, bins weighted by their overlap with the stretch. Coverage = the contact-free share of the stretch area;
the full stretch value (all bins) is reported beside it. Pair score = mean of the two sides (min also given).
"""
import json
import os

import numpy as np

RES = "/sp/v5_va/results"
ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
BARS = {"W1": 0.0966, "W2": 0.0204, "W3": 0.1416}
W7 = {"0-40": 0.0785, "40-80": 0.2183}
J = json.load(open(RES + "/pilot_proc_ia172.json"))


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


def rr(x, n=4):
    return None if x is None else round(float(x), n)


def wflags(u):
    if not u:
        return None
    a, b, c = u.get("0.3mm"), u.get("0.5mm"), u.get("rel0.15")
    w1 = a is not None and a >= BARS["W1"]
    return {"W1": w1, "W2": b is not None and b >= BARS["W2"], "W3_with_W1": bool(w1 and c is not None and c >= BARS["W3"]),
            "W3_alone": c is not None and c >= BARS["W3"]}


def contact_free(case, side, rng_):
    """area-weighted union > 0.3 mm over contact-free 10 mm bins overlapping [a, b]"""
    W = os.path.join(ROOT, case, "ct_seg/ct_seg4")
    M, E, PR = L(W + "/measure.json"), L(W + "/edge.json"), L(W + "/protrusion.json")
    sd = "right" if side == "R" else "left"
    b = g(M, "full", sd, "bins10") or []
    pb = g(E, "sides", side, "per_10mm") or []
    if not b or not pb or not rng_:
        return None
    a0, a1 = rng_
    cm = {round(x["from_mm"]): x for x in pb}
    A = Ac = D = Dc = 0.0
    D5 = DR = 0.0
    cls = {}
    nb = nbc = 0
    for x in b:
        lo, hi = x["from_mm"], x["to_mm"]
        ov = max(0.0, min(hi, a1) - max(lo, a0)) / max(hi - lo, 1e-9)
        if ov <= 0:
            continue
        e = cm.get(round(lo)) or {}
        k = (e.get("frac_rays_upper_threshold_kind3") or 0) + (e.get("frac_rays_dip_before_bright_kind1") or 0)
        con = k >= 0.05
        ar = (x.get("area_mm2") or 0.0) * ov
        A += ar
        D += ar * x["frac_0.3mm"]
        D5 += ar * x["frac_0.5mm"]
        DR += ar * (x.get("frac_rel_0.15") or 0.0)
        nb += 1
        if con:
            for c_, n_ in (e.get("contact_classes") or {}).items():
                cls[c_] = cls.get(c_, 0) + n_
        else:
            Ac += ar
            Dc += ar * x["frac_0.3mm"]
            nbc += 1
    pr = (PR or {}).get("rva" if side == "R" else "lva") or {}
    return {"range_mm": [a0, a1], "bins_overlapping": nb, "bins_contact_free": nbc,
            "union_0.3_all_bins": rr(D / A) if A > 0 else None, "union_0.5_all_bins": rr(D5 / A) if A > 0 else None,
            "union_rel0.15_all_bins": rr(DR / A) if A > 0 else None, "area_mm2_bins": rr(A, 1),
            "union_0.3_contact_free": rr(Dc / Ac) if Ac > 0 else None,
            "contact_free_area_share": rr(Ac / A, 3) if A > 0 else None,
            "contact_bin_ray_classes": cls,
            "protrusion_share_of_label": pr.get("protrusion_share"),
            "protrusion_lumen_bright_share": pr.get("protrusion_share_lumen_bright_h_ge_0.75"),
            "acquisition_voxel_mm": (M or {}).get("acquisition_voxel_mm"),
            "ownership": "k64" if str(g(M, "full", sd, "name") or "").endswith("(k64)") else g(M, "full", sd, "name")}


def backed_only(r):
    """the recalibrated donor verdict with ONLY the two-patient-backed changes kept (feasibility report, TaG-Net section:
    rule 6 -> review, a V3 floor below 15 mm, the C7 over-call fixed (Erec), calibre from the CT half-max); the provisional
    in-sample limits are reverted to the draft: any V1 bridge fails (not <= 3 mm), V2 band with the IQR term, the 25 mm
    total counted over ALL bridges, V3 on the study cord with d <= 5 (the draft's d) and the floor at 12 mm (any floor in
    (4.8, 14.3] gave the same calibration counts)."""
    if not r.get("trace_ok"):
        return {"passes": False, "failed": ["no trace"]}
    fails = [x for x in (r.get("gate_fail_reasons") or []) if x != "7_v3loop_robust"]
    br = r.get("bridges") or []
    if any(b.get("segment") == "V1" and (b.get("length_mm") or 0) > 0 for b in br) and "3a_no_V1_bridge" not in fails:
        fails.append("3a any V1 bridge (draft)")
    if not (r.get("v2_band") or {}).get("in_band"):
        fails.append("5a V2 band with IQR (draft)")
    if sum((b.get("length_mm") or 0) for b in br) > 25:
        fails.append("3d total over all bridges > 25 (draft)")
    v = r.get("v3_study") or [None, None]
    if not (v[0] is not None and v[0] >= 12 and v[1] is not None and v[1] <= 5):
        fails.append("V3 study cord exc >= 12, d <= 5 (%s/%s)" % (v[0], v[1]))
    return {"passes": not fails, "failed": fails}


rows = []
for r in J["records"]:
    case, side = r["case"], r["side"]
    v3s = r.get("v3_study") or [None, None]
    a1 = v3s[0] is not None and v3s[0] >= 15 and (v3s[1] if v3s[1] is not None else 99) <= 5
    a2 = "C1" in (r.get("levels") or [])
    rec_pass = bool(r.get("passes_as_v5_donor"))
    dr_pass = bool(r.get("passes_draft_with_additions"))
    why_draft = list(r.get("draft_failed") or []) + ([] if a1 else ["A1_v3loop_study_15_5 (%s/%s)" % tuple(v3s)]) + \
        ([] if a2 else ["A2_C1_threaded"])
    row = {"case": case, "side": side,
           "qc_recalibrated": {"status": r.get("status"), "passes": rec_pass, "failed": r.get("gate_fail_reasons"),
                               "review": r.get("review"), "distal_ok": r.get("distal_ok"),
                               "whole_va_pass": r.get("whole_va_pass")},
           "qc_draft_with_additions": {"passes": dr_pass, "failed": why_draft},
           "gates_differ": rec_pass != dr_pass,
           "qc_two_patient_backed_only": backed_only(r),
           "entry_Erec": r.get("entry_Erec"), "entry_ts_qc": r.get("entry_ts_qc"),
           "v3_robust": r.get("v3_robust"), "v3_study": v3s,
           "length_trace_mm": r.get("length_trace_mm"),
           "hu_native_ratio_CCA": g(r, "hu_native_origin_to_C1", "ratio_to_CCA"),
           "r_corrected_mm": g(r, "vmtk", "r_corrected_median_mm")}
    w = r.get("wall")
    if w:
        st = w.get("stretches") or {}
        don = g(st, "rec", "origin_to_C1_minus_seam") or {}
        seam = g(st, "rec", "seam_zone_C1m25_to_C1") or {}
        src = "rec stretch (exact)"
        if not don:
            # 140 L / 001 R: the wall was cut at the dura (distal course failed QC), so the landmarks have no dura and
            # va_metrics computed only the ts set's cervical stretches. Use the ts stretch when its C1 equals the rec C1;
            # else area-weighted 10 mm bins over origin -> C1rec - 25 (approximate, bins weighted by overlap).
            lr, lt = g(w, "landmarks", "rec") or {}, g(w, "landmarks", "ts") or {}
            if lr.get("C1") is not None and lr.get("C1") == lt.get("C1") and g(st, "ts", "origin_to_C1_minus_seam"):
                don, seam = g(st, "ts", "origin_to_C1_minus_seam"), g(st, "ts", "seam_zone_C1m25_to_C1") or {}
                src = "ts stretch (exact; ts C1 = rec C1)"
            elif lr.get("C1") is not None:
                c1 = float(lr["C1"])
                a_ = contact_free(case, side, [0.0, c1 - 25.0]) or {}
                b_ = contact_free(case, side, [c1 - 25.0, c1]) or {}
                don = {"0.3mm": a_.get("union_0.3_all_bins"), "0.5mm": a_.get("union_0.5_all_bins"),
                       "rel0.15": a_.get("union_rel0.15_all_bins"), "area_mm2": a_.get("area_mm2_bins"),
                       "range_mm": [0.0, c1 - 25.0]}
                seam = {"0.3mm": b_.get("union_0.3_all_bins"), "0.5mm": b_.get("union_0.5_all_bins"),
                        "rel0.15": b_.get("union_rel0.15_all_bins"), "range_mm": [c1 - 25.0, c1]}
                src = "10 mm bins over the rec range (approximate)"
        d04 = g(st, "ts", "distal_0_40", "0.3mm")
        d48 = g(st, "ts", "distal_40_80", "0.3mm")
        piece = g(w, "pieces", "origin_to_C1_rec") or {}
        cf = contact_free(case, side, don.get("range_mm"))
        row["wall"] = {
            "gates_V1_V6_whole": w.get("gates_pass"), "genus_whole": w.get("genus"),
            "genus_origin_to_C1_piece": piece.get("genus"),
            "centre_offset_mm": w.get("centre_offset_mm_median"),
            "r_misr_mm": w.get("r_misr_median_mm"), "edge_FWHM_mm": g(w, "edge", "FWHM_verifier_mm"),
            "noise_HU": g(w, "edge", "noise", "lumen_voxel_std_HU_median"),
            "whole": w.get("whole"), "whole_flags": wflags(w.get("whole")),
            "donor_stretch_origin_to_C1m25": {k: don.get(k) for k in ("0.3mm", "0.5mm", "rel0.15", "area_mm2", "range_mm")},
            "donor_stretch_flags": wflags(don), "donor_stretch_source": src,
            "seam_zone_C1m25_to_C1": {k: seam.get(k) for k in ("0.3mm", "0.5mm", "rel0.15", "range_mm")},
            "origin_to_C1_piece_rec": piece.get("union"),
            "C6_to_C1_ct": g(st, "ct", "C6_to_C1"), "C6_to_C1_rec": g(st, "rec", "C6_to_C1"),
            "W7_by_hand": {"0-40": d04, "40-80": d48,
                           "result": None if d04 is None or d48 is None else ("pass" if d04 >= W7["0-40"] and d48 >= W7["40-80"] else "FAIL"),
                           "wall_reaches_true_distal_end": bool(r.get("distal_ok")),
                           "note": "whole-VA donors only; where the distal course failed the wall stops at the dura cut"},
            "contact_free": cf,
            "plain_tube_bin_fraction": w.get("plain_tube_bin_fraction_lt_0.01"),
            "nofix_plain_ct_seg4": (r.get("wall_plain_ct_seg4_nofix") or {}).get("whole") if isinstance(r.get("wall_plain_ct_seg4_nofix"), dict) else None}
    rows.append(row)


def score(row):
    return g(row, "wall", "contact_free", "union_0.3_contact_free")


by_case = {}
for row in rows:
    by_case.setdefault(row["case"], {})[row["side"]] = row
pairs, pairs_draft, singles = [], [], []
for c, s in by_case.items():
    for sd, row in s.items():
        if row["qc_recalibrated"]["passes"] and row.get("wall"):
            singles.append({"case": c, "side": sd, "status": row["qc_recalibrated"]["status"],
                            "donor_stretch_0.3_contact_free": score(row),
                            "donor_stretch_0.3_all": g(row, "wall", "donor_stretch_origin_to_C1m25", "0.3mm"),
                            "contact_free_area_share": g(row, "wall", "contact_free", "contact_free_area_share"),
                            "whole_0.3": g(row, "wall", "whole", "0.3mm"),
                            "genus_origin_to_C1_piece": g(row, "wall", "genus_origin_to_C1_piece"),
                            "donor_stretch_source": g(row, "wall", "donor_stretch_source"),
                            "passes_draft_too": row["qc_draft_with_additions"]["passes"]})
    if len(s) == 2 and all(s[k]["qc_recalibrated"]["passes"] and s[k].get("wall") for k in ("R", "L")):
        sc = [score(s["R"]), score(s["L"])]
        al = [g(s[k], "wall", "donor_stretch_origin_to_C1m25", "0.3mm") for k in ("R", "L")]
        p = {"case": c, "score_mean_contact_free": rr(np.mean([x for x in sc if x is not None])) if any(x is not None for x in sc) else None,
             "score_min_contact_free": rr(min(x for x in sc if x is not None)) if any(x is not None for x in sc) else None,
             "R_L_contact_free": sc, "R_L_all_bins": al,
             "R_L_status": [s["R"]["qc_recalibrated"]["status"], s["L"]["qc_recalibrated"]["status"]],
             "review_flags": {k: s[k]["qc_recalibrated"]["review"] for k in ("R", "L")},
             "both_pass_draft_too": s["R"]["qc_draft_with_additions"]["passes"] and s["L"]["qc_draft_with_additions"]["passes"]}
        pairs.append(p)
        if p["both_pass_draft_too"]:
            pairs_draft.append(c)
pairs.sort(key=lambda p: -(p["score_mean_contact_free"] or -1))
singles.sort(key=lambda p: -(p["donor_stretch_0.3_contact_free"] or -1))
for i, p in enumerate(pairs):
    p["rank_within_ia172"] = i + 1
n_rec = sum(1 for x in rows if x["qc_recalibrated"]["passes"])
n_dr = sum(1 for x in rows if x["qc_draft_with_additions"]["passes"])
bk = [x for x in rows if x["qc_two_patient_backed_only"]["passes"]]
bk_cases = {}
for x in bk:
    bk_cases.setdefault(x["case"], []).append(x["side"])
out = {"about": __doc__.split("\n")[0], "bars_native": BARS, "W7_bars": W7, "rows": rows,
       "counts": {"sides": len(rows), "pass_recalibrated": n_rec, "pass_draft_with_additions": n_dr,
                  "sides_where_gates_differ": [x["case"] + " " + x["side"] for x in rows if x["gates_differ"]],
                  "pairs_recalibrated": [p["case"] for p in pairs], "pairs_draft_with_additions": pairs_draft,
                  "pass_two_patient_backed_only": len(bk),
                  "sides_two_patient_backed_only": [x["case"] + " " + x["side"] for x in bk],
                  "pairs_two_patient_backed_only": sorted(c for c, s in bk_cases.items() if len(s) == 2),
                  "note": "pairs_recalibrated needs both sides with a ct_seg4 wall"},
       "pair_ranking_within_ia172": pairs, "single_side_ranking_within_ia172": singles,
       "ranking_rule": "pair score = mean over R and L of the donor-stretch (origin -> C1f - 25 mm) union > 0.3 mm on "
                       "contact-free 10 mm bins (< 5 % bright-neighbour rays); recalibrated-gate donors with a ct_seg4 wall"}
json.dump(out, open(RES + "/pilot_proc_ia172_grade.json", "w"), indent=1, default=str)
# (02:10) also merge the grade into the main results file: each per-side record gets "grade_v5" (this row without the
# duplicated case / side), and the file gets top-level "grade_counts", "pair_ranking_within_ia172", "ranking_rule".
# Re-running pilot_proc_ia172_collect.py replaces a case's records (and so drops grade_v5); re-run this script after it.
gi_ = {(x["case"], x["side"]): x for x in rows}
for r in J["records"]:
    x = gi_.get((r["case"], r["side"]))
    if x:
        r["grade_v5"] = {k: v for k, v in x.items() if k not in ("case", "side")}
J["grade_counts"] = out["counts"]
J["pair_ranking_within_ia172"] = pairs
J["single_side_ranking_within_ia172"] = singles
J["ranking_rule"] = out["ranking_rule"]
J["grade_source"] = "pilot_proc_ia172_grade.py (results/pilot_proc_ia172_grade.json)"
json.dump(J, open(RES + "/pilot_proc_ia172.json", "w"), indent=1, default=str)
lines = ["COUNTS " + json.dumps(out["counts"]), "",
         "%-10s %-2s %-12s %-5s %-5s %-5s %-6s | %-22s | %-22s | %-15s | %-22s | %-11s | %s" % (
             "case", "sd", "recal", "draft", "bkd", "diff", "Erec", "whole .3/.5/rel W1W2W3", "donor .3/.5/rel W1W2W3",
             "seam .3/rel", "contact-free .3 (share)", "W7 0-40/40-80", "draft fails / review")]
for x in rows:
    w = x.get("wall") or {}
    wh, dn, sm, cf = w.get("whole") or {}, w.get("donor_stretch_origin_to_C1m25") or {}, w.get("seam_zone_C1m25_to_C1") or {}, w.get("contact_free") or {}
    fl = lambda f: "".join("Y" if (f or {}).get(k) else "n" for k in ("W1", "W2", "W3_with_W1")) if f else "--"  # noqa: E731
    w7 = w.get("W7_by_hand") or {}
    lines.append("%-10s %-2s %-12s %-5s %-5s %-5s %-6s | %-22s | %-22s | %-15s | %-22s | %-11s | %s" % (
        x["case"], x["side"], x["qc_recalibrated"]["status"], "pass" if x["qc_draft_with_additions"]["passes"] else "FAIL",
        "pass" if x["qc_two_patient_backed_only"]["passes"] else "FAIL", "DIFF" if x["gates_differ"] else "", x["entry_Erec"],
        "%s/%s/%s %s" % (wh.get("0.3mm"), wh.get("0.5mm"), wh.get("rel0.15"), fl(w.get("whole_flags"))) if wh else "-",
        "%s/%s/%s %s" % (dn.get("0.3mm"), dn.get("0.5mm"), dn.get("rel0.15"), fl(w.get("donor_stretch_flags"))) if dn else "-",
        "%s/%s" % (sm.get("0.3mm"), sm.get("rel0.15")) if sm else "-",
        "%s (%s)" % (cf.get("union_0.3_contact_free"), cf.get("contact_free_area_share")) if cf else "-",
        "%s/%s" % (w7.get("0-40"), w7.get("40-80")) if w7 else "-",
        "; ".join(x["qc_draft_with_additions"]["failed"]) + (" || recal: " + "; ".join(x["qc_recalibrated"]["failed"] or []) if x["qc_recalibrated"]["failed"] else "")))
lines += ["", "PAIR RANKING (recalibrated donors with a wall, within ia172):"]
for p in pairs:
    lines.append("#%d %s mean %s min %s  R/L contact-free %s  all-bins %s  %s  draft both %s" % (
        p["rank_within_ia172"], p["case"], p["score_mean_contact_free"], p["score_min_contact_free"], p["R_L_contact_free"],
        p["R_L_all_bins"], p["R_L_status"], p["both_pass_draft_too"]))
lines += ["", "SINGLE SIDES (recalibrated donors with a wall):"]
for s in singles:
    lines.append("%s %s %-12s contact-free %s (share %s) all %s whole %s draft %s piece genus %s [%s]" % (
        s["case"], s["side"], s["status"], s["donor_stretch_0.3_contact_free"], s["contact_free_area_share"],
        s["donor_stretch_0.3_all"], s["whole_0.3"], s["passes_draft_too"], s["genus_origin_to_C1_piece"],
        s["donor_stretch_source"]))
open(RES + "/pilot_proc_ia172_grade.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
