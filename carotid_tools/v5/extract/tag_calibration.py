#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_calibration.py (= /sp/v5_va/tag_calibration.py in the build containers)
"""tag_calibration.py -- assemble the recalibration of the v5 extraction and graft checks on the two ground-truth (GT)
patients (VMR 0248, right-side landmarks trusted; TaG-Net A; TaG-Net B path-only) and the 40 traced sides of the 21
TotalSegmentator (TS) Arterial runs (24 subjects tried).

    python3 tag_calibration.py      -> /sp/v5_va/results/tag_calibration.json        (neve-build-stagea)

Inputs (all produced by NEW tag_calibration_* scripts, or read-only study / tag_* results):
  results/tag_calibration_paths_{VMR,A,B,TS}.json  (tag_calibration_paths.py: ts_qc, V3 loop, entry rules, band, HU)
  results/tag_calibration_radius.json             (tag_calibration_radius.py)
  results/tag_calibration_gross.json              (tag_calibration_gross.py)
  results/tag_arterial_bench_<etag>.json          (tag_arterial: course correctness of A's traces vs the GT)
  results/arterial_metrics.json                   (study: TS per-side QC, usable flags)
  verify_arterial_yield_out/{qc,v3loop}.json       (verifier: axial threading test, V3 loop on the VMTK lines)
  derived/arterial/ts_*/va_extract.json, derived/tagnet/arterial/tagA_*/va_extract.json, derived/arterial/vmr_* (gaps,
  labels), results/vmr0248_metrics.json, results/tag_gt.json (GT landmarks).
Items 1-5 of the brief; definitions in notes/tag_calibration.md.
LICENCE: TaG-Net -> internal only; VMR research use; Arterial-derived -> non-commercial; TS CC BY 4.0.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import tag_calibration_paths as TP  # noqa: E402  (this agent's harness; path builders)
from tag_calibration_entry import entry as erec_entry  # noqa: E402  (this agent's corrected entry rule)

EREC = (0.5, 5.0, 2.0)   # d_thr, run_min, gap_min of the recommended entry rule (tag_calibration_entry.py grid)
RULES = TP.RULES + ("Erun", "Erec")
REC_V3 = ("robust", 12, 6)   # recommended V3-loop gate: robust-cord excursion >= 12 mm and d_C1post <= 6 mm

RES = "/sp/v5_va/results"
DER = "/repo/carotid_data/va_sources/derived"
WRONG6 = {"s0289 R", "s0333 L", "s0518 L", "s1107 R", "s1129 R", "s1154 R"}


def L(p):
    return json.load(open(p))


def rr(x, n=2):
    return None if x is None else round(float(x), n)


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def stats(a):
    a = [x for x in a if x is not None]
    if not a:
        return None
    return {"n": len(a), "min": rr(min(a), 1), "median": rr(np.median(a), 1), "max": rr(max(a), 1)}


# ----------------------------------------------------------------------------- correctness of the GT-patient traces
def a_correct(etag, side):
    """Course correctness of an A trace against the GT (tag_arterial bench): whole = coverage >= 0.95 and <= 5 % of
    the trace > 3 mm off; cervical (ostium -> GT dura) likewise."""
    p = RES + "/tag_arterial_bench_%s.json" % etag
    if not os.path.isfile(p):
        return None
    s = L(p)["sides"].get(side, {}).get("traced", {})
    o = {}
    for k in ("gt", "gt_cervical"):
        g = s.get(k) or {}
        if g.get("coverage_frac") is None:
            continue
        o["whole" if k == "gt" else "cervical"] = bool(g["coverage_frac"] >= 0.95 and (g.get("frac_cand_off_ref_gt3mm") or 0) <= 0.05)
        o["coverage_" + k] = g["coverage_frac"]
    return o


def augment(v):
    """Add the recommended entry rule 'Erec' to one harness record: entry level and C6 landmark from
    tag_calibration_entry.entry(d_thr 0.5, run_min 5, gap_min 2) on the stored profiles; level count = ts_qc's (its
    C6..C1 counts match the verifier's independent axial test on 39 of 40 TS sides); C2 = ts_qc's; C1f = the exit of
    the first C1 run after C2 (E1h runs) = VMR's C1 convention, else ts_qc's C1."""
    if not isinstance(v, dict) or "profile_1mm" not in v:
        return
    e = v["entry"]
    zr = {L: p["z_range_mm"] for L, p in e["per_level"].items()}
    lev, s0 = erec_entry(v["profile_1mm"], EREC[0], EREC[1], EREC[2], zr)
    q = v["ts_qc"]
    qlm = q.get("landmarks_mm") or {}
    # "Erun" = the pure run rule; "Erec" (recommended) = ts_qc's call with a C7 call confirmed only by Erun
    e["Erun_entry"] = lev
    e["Erun_n_C6_to_C1"] = q.get("n_C6_to_C1_passed")
    e["Erun_levels"] = q.get("foramen_levels_passed")
    e["Erun_landmarks_mm"] = {"entry": s0, "C2": qlm.get("C2"),
                              "C1f": (e.get("E1h_landmarks_mm") or {}).get("C1f") or qlm.get("C1")}
    e0 = q.get("foramen_entry_level")
    comb = "C6" if (e0 == "C7" and lev != "C7") else e0
    if comb is not None and comb == lev and s0 is not None:
        s_ent = s0
    elif comb is not None and (e["per_level"].get(comb) or {}).get("E1_runs_s_mm"):
        s_ent = e["per_level"][comb]["E1_runs_s_mm"][0][0]
    else:
        s_ent = qlm.get("C6")
    e["Erec_entry"] = comb
    e["Erec_n_C6_to_C1"] = q.get("n_C6_to_C1_passed")
    e["Erec_levels"] = q.get("foramen_levels_passed")
    e["Erec_landmarks_mm"] = {"entry": s_ent, "C2": qlm.get("C2"),
                              "C1f": (e.get("E1h_landmarks_mm") or {}).get("C1f") or qlm.get("C1")}


# ----------------------------------------------------------------------------- V3-loop item
def v3_rows(group_json, msname):
    out = {}
    for pid, row in group_json["paths"].items():
        v = row.get(msname)
        if not isinstance(v, dict) or "error" in v:
            continue
        out[pid] = {"kind": row.get("kind"), "quality": row.get("quality"), "etag": row.get("etag"), "side": row.get("side"),
                    "study": [v["v3loop_study"].get("posterior_excursion_mm"), v["v3loop_study"].get("d_C1_posterior_half_mm")],
                    "zcap": [v["v3loop_zcap"].get("posterior_excursion_mm"), v["v3loop_zcap"].get("d_C1_posterior_half_mm")],
                    "robust": [v["v3loop_robust"].get("posterior_excursion_mm"), v["v3loop_robust"].get("d_C1_posterior_half_mm")],
                    "cord_top": v.get("cord_top"),
                    "z_c1_top": v["z_c1_top_mm"], "pathonly": [row["pathonly"]["excursion_raw_mm"], row["pathonly"]["excursion_fit_mm"]]}
    return out


def gate_v3(ex_d, exc_min, d_max):
    e, d = ex_d
    return bool(e is not None and e >= exc_min and (d if d is not None else 99) <= d_max)


# ----------------------------------------------------------------------------- landmark projection on the GT path
def project(Pc, s_c, Pg):
    """s along the GT path Pg of the candidate path's point at arclength s_c."""
    if s_c is None:
        return None
    s = arclen(Pc)
    q = np.array([np.interp(s_c, s, Pc[:, i]) for i in range(3)])
    sg = arclen(Pg)
    return float(sg[cKDTree(Pg).query(q)[1]])


def main():
    out = {"about": __doc__.split("\n\n")[0], "licence": "TaG-Net: no data licence -> internal benchmarking only; VMR 0248 "
           "research use (Wilson et al. 2013); Arterial-derived traces non-commercial; TS v3 CC BY 4.0", "items": {}}
    PV, PA, PB, PT = (L(RES + "/tag_calibration_paths_%s.json" % g) for g in ("VMR", "A", "B", "TS"))
    for J in (PV, PA):
        for row in J["paths"].values():
            for ms in ("total_v3", "total"):
                augment(row.get(ms))
    for row in PT["paths"].values():
        for k in ("trace", "vmtk"):
            augment(row.get(k))
    VQ = L("/sp/v5_va/verify_arterial_yield_out/qc.json")
    VV = L("/sp/v5_va/verify_arterial_yield_out/v3loop.json")
    AM = {c["subject"]: c for c in L(RES + "/arterial_metrics.json")["totalsegmentator"]["cases"]}

    # ===================================================================== 1. V3 loop
    it = {"definitions": {
        "study": "verify_arterial_yield_v3loop / tag_totalseg_qc.v3loop (unchanged): stretch z >= C2 bottom and s <= ts_qc "
                 "dura proxy (first s above z_FM = C1 top + 6 mm after the C1 landmark); excursion = median dy(V2: C6 mid .. "
                 "C3 top) - min dy(stretch), dy = y - y_cord(z); d = min distance of the stretch to the posterior half of C1",
        "zcap": "same, stretch ends at the first point above C1 top + 10 mm instead of the dura proxy",
        "pathonly": "no masks: V2 = s in [0.20, 0.45] L, V3 window = s in [0.45, 0.90] L; raw = median y(V2) - min y(V3); "
                    "fit = max over V3 of (linear y(z) of V2 extrapolated) - y"},
        "mask_sets": {}}
    for ms in ("total_v3", "total"):
        rows = {}
        for grp, J in (("VMR", PV), ("A", PA)):
            for pid, r in v3_rows(J, ms).items():
                if grp == "A" and r["kind"] in ("trace", "vmtk"):
                    r["correct"] = a_correct(r["etag"], r["side"])
                elif grp == "VMR" and r["kind"] in ("trace", "vmtk"):
                    r["correct"] = {"whole": True, "cervical": True, "note": "VMR traces: coverage 0.94-1.00 (study bench)"}
                rows[pid] = r
        it["mask_sets"][ms] = rows
    # TS: trace (dataset masks, both definitions) + verifier VMTK numbers + study flags
    ts_rows = {}
    for key, r in PT["paths"].items():
        sid, sd = key.split()
        am = AM[sid]["vas"][sd]
        row = {"trace_ok": r.get("trace_ok", False), "study_usable_cervical": am.get("usable_cervical"),
               "study_usable_whole": am.get("usable_course"), "verifier_wrong_vessel": key in WRONG6,
               "verifier_vmtk": [VV.get(key, {}).get("posterior_excursion_mm"), VV.get(key, {}).get("d_C1_posterior_half_mm")]}
        if r.get("trace_ok"):
            t = r["trace"]
            row["trace_study"] = [t["v3loop_study"].get("posterior_excursion_mm"), t["v3loop_study"].get("d_C1_posterior_half_mm")]
            row["trace_zcap"] = [t["v3loop_zcap"].get("posterior_excursion_mm"), t["v3loop_zcap"].get("d_C1_posterior_half_mm")]
            row["trace_robust"] = [t["v3loop_robust"].get("posterior_excursion_mm"), t["v3loop_robust"].get("d_C1_posterior_half_mm")]
            row["cord_top"] = t.get("cord_top")
            row["pathonly"] = [r["pathonly"]["excursion_raw_mm"], r["pathonly"]["excursion_fit_mm"]]
            if "vmtk" in r:
                row["vmtk_zcap"] = [r["vmtk"]["v3loop_zcap"].get("posterior_excursion_mm"), r["vmtk"]["v3loop_zcap"].get("d_C1_posterior_half_mm")]
                row["vmtk_robust"] = [r["vmtk"]["v3loop_robust"].get("posterior_excursion_mm"), r["vmtk"]["v3loop_robust"].get("d_C1_posterior_half_mm")]
                row["vmtk_study"] = [r["vmtk"]["v3loop_study"].get("posterior_excursion_mm"), r["vmtk"]["v3loop_study"].get("d_C1_posterior_half_mm")]
        ts_rows[key] = row
    it["ts_sides"] = ts_rows
    # threshold sweep and counts
    cands = [(dfn, e, d) for dfn in ("study", "robust") for e in (15, 12, 10, 8) for d in (5, 6, 8)]
    sweep = {}
    for dfn, emin, dmax in cands:
        nm = "%s exc>=%d d<=%d" % (dfn, emin, dmax)
        o = {}
        for ms in ("total_v3", "total"):
            rows = it["mask_sets"][ms]
            gt = {k: gate_v3(v[dfn], emin, dmax) for k, v in rows.items() if v["kind"].startswith("gt")}
            corr = {k: gate_v3(v[dfn], emin, dmax) for k, v in rows.items() if v["kind"] in ("trace", "vmtk")
                    and (v.get("correct") or {}).get("cervical")}
            wrong = {k: gate_v3(v[dfn], emin, dmax) for k, v in rows.items() if v["kind"] in ("trace", "vmtk")
                     and (v.get("correct") or {}).get("cervical") is False}
            o[ms] = {"gt_pass": "%d/%d" % (sum(gt.values()), len(gt)), "gt_fail": [k for k, p in gt.items() if not p],
                     "correct_traces_pass": "%d/%d" % (sum(corr.values()), len(corr)),
                     "correct_traces_fail": [k for k, p in corr.items() if not p],
                     "wrong_traces_pass": "%d/%d" % (sum(wrong.values()), len(wrong)),
                     "wrong_traces_passing": [k for k, p in wrong.items() if p]}
        tsk = {"study": "trace_study", "zcap": "trace_zcap", "robust": "trace_robust"}[dfn]
        tp = {k: gate_v3(v[tsk], emin, dmax) for k, v in ts_rows.items() if v.get("trace_ok")}
        o["TS_traced_sides_pass"] = "%d/%d" % (sum(tp.values()), len(tp))
        o["TS_study_usable21_pass"] = "%d/21" % sum(p for k, p in tp.items() if ts_rows[k]["study_usable_cervical"])
        o["TS_wrong6_pass"] = [k for k in WRONG6 if tp.get(k)]
        o["TS_passing_not_study_usable"] = [k for k, p in tp.items() if p and not ts_rows[k]["study_usable_cervical"]]
        sweep[nm] = o
    it["threshold_sweep"] = sweep
    # separation margins on TS (trace, both defs)
    for dfn in ("trace_study", "trace_robust", "vmtk_robust", "verifier_vmtk"):
        real = [v[dfn][0] for k, v in ts_rows.items() if v.get(dfn) and v[dfn][0] is not None and v[dfn][1] is not None and v[dfn][1] <= 5]
        it.setdefault("ts_distributions", {})[dfn] = {
            "excursion_of_sides_with_d_le_5": sorted(real),
            "excursion_of_sides_with_d_gt_5": sorted(v[dfn][0] for k, v in ts_rows.items() if v.get(dfn) and v[dfn][0] is not None
                                                     and (v[dfn][1] is None or v[dfn][1] > 5)),
            "d_of_sides_with_exc_ge_15": sorted(v[dfn][1] for k, v in ts_rows.items() if v.get(dfn) and v[dfn][0] is not None
                                                and v[dfn][0] >= 15 and v[dfn][1] is not None)}
    # B path-only + calibration of the proxy
    po = {"B": {k: v["pathonly"] for k, v in PB["paths"].items()}}
    po["A_gt"] = {k: PA["paths"][k]["pathonly"] for k in ("A gt R", "A gt L")}
    po["VMR_gt"] = {k: PV["paths"][k]["pathonly"] for k in ("VMR expert R", "VMR expert L", "VMR ct_seg4 R", "VMR ct_seg4 L")}
    real_po = [v["pathonly"][0] for k, v in ts_rows.items() if v.get("trace_ok") and v.get("trace_zcap")
               and gate_v3(v["trace_zcap"], 12, 5)]
    wrong_po = [v["pathonly"][0] for k, v in ts_rows.items() if v.get("trace_ok") and v.get("trace_zcap")
                and not gate_v3(v["trace_zcap"], 12, 5)]
    po["TS_trace_raw_proxy_real_by_zcap_gate"] = sorted(real_po)
    po["TS_trace_raw_proxy_fail_by_zcap_gate"] = sorted(wrong_po)
    a_po = {}
    for k, v in it["mask_sets"]["total_v3"].items():
        if k.startswith("A ") and v["kind"] in ("trace", "vmtk"):
            a_po.setdefault("correct" if (v.get("correct") or {}).get("cervical") else "wrong", []).append(v["pathonly"][0])
    po["A_traces_raw_proxy"] = {k: sorted(x) for k, x in a_po.items()}
    it["pathonly_proxy"] = po
    out["items"]["1_v3_loop"] = it

    # ===================================================================== 2. entry level
    rules = RULES
    ent = {"rule_definitions": dict(TP.RULE_DEF, Erun="runs of path within 0.5 mm of the level mask, z inside its own range, >= 5 mm; lowest level C7..C3 with a run; a C7 run counts only if >= 2 mm of path separate it from the first C6 run (tag_calibration_entry.py grid)", Erec="RECOMMENDED: ts_qc's call (E0), except that a C7 call is kept only when Erun also calls C7, else C6; level count = ts_qc's; C6 landmark = start of the Erun run (else the first E1 run of the level)"), "truth": "C6 on all four GT VAs (VMR R/L, A R/L); VMR left landmark s disputed, level "
           "not", "rows": {}}
    gtlm = L(RES + "/tag_gt.json")["landmarks_A"]
    vmrlm = L(RES + "/vmr0248_metrics.json")
    # GT C6 points (for projection): VMR on its ct_seg4 VMTK line; A on its GT points
    gt_paths = {}
    for p in TP.paths_vmr():
        if p["id"] in ("VMR ct_seg4 R", "VMR ct_seg4 L"):
            gt_paths["VMR " + p["side"]] = p["P"]
    for p in TP.paths_A():
        if p["id"] in ("A gt R", "A gt L"):
            gt_paths["A " + p["side"]] = p["P"]
    gt_c6 = {"VMR R": vmrlm["landmarks_ct"]["right"]["C6"], "VMR L": 37.0,
             "A R": gtlm["right"]["landmarks_s_from_ostium_mm"]["C6"], "A L": gtlm["left"]["landmarks_s_from_ostium_mm"]["C6"]}
    gt_c2 = {"VMR R": vmrlm["landmarks_ct"]["right"]["C2"], "VMR L": vmrlm["landmarks_ct"]["left"]["C2"],
             "A R": gtlm["right"]["landmarks_s_from_ostium_mm"]["C2"], "A L": gtlm["left"]["landmarks_s_from_ostium_mm"]["C2"]}
    gt_c1 = {"VMR R": vmrlm["landmarks_ct"]["right"]["C1"], "VMR L": vmrlm["landmarks_ct"]["left"]["C1"],
             "A R": gtlm["right"]["landmarks_s_from_ostium_mm"]["C1"], "A L": 161.0}
    allpaths = {p["id"]: p["P"] for p in TP.paths_vmr() + TP.paths_A()}
    for grp, J in (("VMR", PV), ("A", PA)):
        for pid, row in J["paths"].items():
            for ms in ("total_v3", "total"):
                v = row.get(ms)
                if not isinstance(v, dict) or "error" in v:
                    continue
                e = v["entry"]
                key = grp + " " + row["side"]
                rec = {"group": grp, "side": row["side"], "kind": row["kind"], "quality": row.get("quality"), "mask_set": ms,
                       "calls": {r: e[r + "_entry"] for r in rules},
                       "n_C6_to_C1": {r: e[r + "_n_C6_to_C1"] for r in rules}}
                if grp == "A" and row["kind"] in ("trace", "vmtk"):
                    rec["correct"] = a_correct(row["etag"], row["side"])
                # landmark offsets projected on the GT path (+ = distal)
                Pc = allpaths.get(pid)
                if Pc is not None:
                    lmo = {}
                    for r in ("Erec", "Erun", "E1z", "E1h", "E2", "E0"):
                        lm = e.get(r + "_landmarks_mm") if r != "E0" else {"entry": v["ts_qc"]["landmarks_mm"].get("C6"),
                                                                            "C2": v["ts_qc"]["landmarks_mm"].get("C2"),
                                                                            "C1f": v["ts_qc"]["landmarks_mm"].get("C1")}
                        if not lm:
                            continue
                        o = {}
                        for nm, gtv in (("entry", gt_c6[key]), ("C2", gt_c2[key]), ("C1f", gt_c1[key])):
                            sp = project(Pc, lm.get(nm), gt_paths[key])
                            o[nm] = None if sp is None else rr(sp - gtv, 1)
                        lmo[r] = o
                    rec["landmark_offset_vs_gt_mm"] = lmo
                ent["rows"]["%s | %s" % (pid, ms)] = rec
    # scores
    score = {}
    for r in rules:
        for ms in ("total_v3", "total"):
            rows = [x for x in ent["rows"].values() if x["mask_set"] == ms]
            gt = [x for x in rows if x["kind"].startswith("gt")]
            tr = [x for x in rows if x["kind"] in ("trace", "vmtk")]
            score.setdefault(r, {})[ms] = {
                "gt_paths_C6": "%d/%d" % (sum(x["calls"][r] == "C6" for x in gt), len(gt)),
                "traces_C6": "%d/%d" % (sum(x["calls"][r] == "C6" for x in tr), len(tr)),
                "traces_C6_VMR": "%d/%d" % (sum(x["calls"][r] == "C6" for x in tr if x["group"] == "VMR"), sum(1 for x in tr if x["group"] == "VMR")),
                "traces_C6_A": "%d/%d" % (sum(x["calls"][r] == "C6" for x in tr if x["group"] == "A"), sum(1 for x in tr if x["group"] == "A")),
                "calls_other": sorted(set("%s=%s" % (k.split(" | ")[0], x["calls"][r]) for k, x in ent["rows"].items()
                                          if x["mask_set"] == ms and x["calls"][r] != "C6" and x["kind"].startswith("gt")))}
    ent["scores"] = score
    # landmark offsets summary (GT and correct traces)
    lmsum = {}
    for r in ("Erec", "Erun", "E1z", "E1h", "E2", "E0"):
        for nm in ("entry", "C2", "C1f"):
            for ms in ("total_v3",):
                vals = {}
                for k, x in ent["rows"].items():
                    if x["mask_set"] != ms or r not in (x.get("landmark_offset_vs_gt_mm") or {}):
                        continue
                    if x["kind"] in ("trace", "vmtk") and x["group"] == "A" and not (x.get("correct") or {}).get("cervical"):
                        if nm != "entry":
                            continue
                    vals.setdefault(x["group"] + " " + x["side"], []).append(x["landmark_offset_vs_gt_mm"][r].get(nm))
                lmsum.setdefault(r, {})[nm] = {k: stats(v) for k, v in vals.items()}
    ent["landmark_offsets_summary_total_v3"] = lmsum
    # TS: calls per rule vs the verifier's axial threading
    tse = {}
    for key, r in PT["paths"].items():
        if not r.get("trace_ok"):
            continue
        sid, sd = key.split()
        q = VQ.get("ts_" + sid, {}).get("sides", {}).get(sd, {})
        thr = q.get("threaded_C6_C1") or []
        vfirst = "C7" if q.get("threaded_C7") else (thr[0] if thr else None)
        e = r["trace"]["entry"]
        tse[key] = {"calls": {x: e[x + "_entry"] for x in rules}, "n_C6_to_C1": {x: e[x + "_n_C6_to_C1"] for x in rules},
                    "verifier_first_threaded": vfirst, "verifier_n_C6_C1": q.get("n_threaded_C6_C1"),
                    "stored_ts_qc_entry": (r.get("stored_ts_qc") or {}).get("foramen_entry_level"),
                    "stored_n_C6_to_C1": (r.get("stored_ts_qc") or {}).get("n_C6_to_C1_passed"),
                    "study_usable_cervical": AM[sid]["vas"][sd].get("usable_cervical"), "wrong6": key in WRONG6,
                    "E1z_landmarks": e.get("E1z_landmarks_mm"), "ts_qc_landmarks": r["trace"]["ts_qc"]["landmarks_mm"]}
    ent["ts_sides"] = tse
    cnt = {}
    for x in rules + ("verifier_first_threaded", "stored_ts_qc_entry"):
        c = {}
        for k, v in tse.items():
            lv = v["calls"][x] if x in rules else v[x]
            c[str(lv)] = c.get(str(lv), 0) + 1
        cnt[x] = c
        if x in rules:
            agree = sum(1 for v in tse.values() if v["calls"][x] == v["verifier_first_threaded"])
            cnt[x + "_agrees_with_verifier"] = "%d/%d" % (agree, len(tse))
            u = [v for v in tse.values() if v["study_usable_cervical"] and not v["wrong6"]]
            cu = {}
            for v in u:
                cu[str(v["calls"][x])] = cu.get(str(v["calls"][x]), 0) + 1
            cnt[x + "_on_15_usable"] = cu
            cnt[x + "_n_C6_to_C1_ge4"] = "%d/%d" % (sum(1 for v in tse.values() if v["n_C6_to_C1"][x] >= 4), len(tse))
    ent["ts_counts"] = cnt
    out["items"]["2_entry_level"] = ent

    # ===================================================================== 3. radius
    RAD = L(RES + "/tag_calibration_radius.json")
    canon = {"VMR": {"native": "vmr_native", "1p5": "vmr_1p5", "1p5_nclean": "vmr_1p5_nclean", "1p5_nsharp": "vmr_1p5_nsharp"},
             "A": {"native": "tagA_native_tsskull", "native (VMR skull, oracle end)": "tagA_native_oracle", "1p5": "tagA_1p5",
                   "1p5_nclean": "tagA_1p5_nclean", "1p5_nsharp": "tagA_1p5_nsharp", "1p5 (TS skull)": "tagA_1p5_tsskull"}}
    rad = {"reference": {p: RAD["patients"][p]["reference"] for p in ("VMR", "A")}, "table": {}}
    for pat, qs in canon.items():
        for q, et in qs.items():
            for sd in ("R", "L"):
                o = RAD["patients"][pat]["etags"].get(et, {}).get(sd, {})
                for zone in ("whole_clean", "origin_to_C1_clean", "cervical_clean"):
                    z = o.get(zone) or {}
                    rad["table"]["%s | %s | %s | %s" % (pat, q, sd, zone)] = {
                        "n": z.get("n"), "r_img": z.get("r_img_median"),
                        **{c: (z[c]["ratio_to_img_median"] if isinstance(z.get(c), dict) else None) for c in ("trace", "vmtk", "maskEDT", "maskArea")},
                        "vmtk_over_ct_misr": (z["vmtk"].get("ratio_to_ct_misr_median") if isinstance(z.get("vmtk"), dict) else None),
                        "ct_misr_over_img": z.get("ct_misr_over_img_median"),
                        "vmtk_mm": (z["vmtk"]["median_mm"] if isinstance(z.get("vmtk"), dict) else None)}
    # factor consistency (origin -> C1, clean: the stretch every trace keeps; native A = TS-skull run)
    cons = {}
    for q in ("native", "1p5", "1p5_nclean", "1p5_nsharp"):
        for c in ("vmtk", "maskArea", "maskEDT", "trace"):
            vals = {}
            for pat in ("VMR", "A"):
                for sd in ("R", "L"):
                    t = rad["table"].get("%s | %s | %s | origin_to_C1_clean" % (pat, q, sd))
                    if t and t.get(c):
                        vals["%s %s" % (pat, sd)] = rr(1.0 / t[c], 3)
            if len(vals) == 4:
                v = list(vals.values())
                vm_ = np.mean([vals["VMR R"], vals["VMR L"]]); va = np.mean([vals["A R"], vals["A L"]])
                cons["%s | %s" % (q, c)] = {"factor_img_over_cand": vals, "VMR_mean": rr(vm_, 3), "A_mean": rr(va, 3),
                                           "pooled": rr(np.mean(v), 3), "spread_max_over_min": rr(max(v) / min(v), 3),
                                           "between_patients_pct": rr(100 * abs(vm_ - va) / np.mean(v), 1),
                                           "within_patient_side_pct": [rr(100 * abs(vals["VMR R"] - vals["VMR L"]) / vm_, 1),
                                                                       rr(100 * abs(vals["A R"] - vals["A L"]) / va, 1)]}
    rad["factor_consistency_origin_to_C1"] = cons
    # additive offset model r_img - r_vmtk (mm)
    add = {}
    for q in ("native", "1p5", "1p5_nclean", "1p5_nsharp"):
        o = {}
        for pat in ("VMR", "A"):
            for sd in ("R", "L"):
                t = rad["table"].get("%s | %s | %s | origin_to_C1_clean" % (pat, q, sd))
                if t and t.get("vmtk_mm") and t.get("r_img"):
                    o["%s %s" % (pat, sd)] = rr(t["r_img"] - t["vmtk_mm"], 3)
        add[q] = o
    rad["offset_img_minus_vmtk_mm_origin_to_C1"] = add
    # what a correction would do to the TS radius gate (VMTK MISR median 0.9-2.6 mm; G1 1.09-2.25 mm)
    tsr = []
    for sid, c in AM.items():
        for sd in ("R", "L"):
            r_ = c["vas"][sd].get("r_median_mm")
            if r_ is not None:
                tsr.append(("%s %s" % (sid, sd), r_))
    effect = {}
    for nm, f in (("none", 1.0), ("x1.10", 1.10), ("x1.17", 1.17), ("x1.25", 1.25)):
        effect[nm] = {"band_0.9_2.6_pass": sum(1 for _, r_ in tsr if 0.9 <= r_ * f <= 2.6), "of": len(tsr),
                      "band_fail": ["%s %.2f" % (k, r_ * f) for k, r_ in tsr if not 0.9 <= r_ * f <= 2.6],
                      "G1_1.09_2.25_pass": sum(1 for _, r_ in tsr if 1.09 <= r_ * f <= 2.25)}
    rad["ts_radius_gate_effect"] = effect
    out["items"]["3_radius"] = rad
    out["_ts_radius_pairs"] = tsr

    # ===================================================================== 4. draft QC gate (donor_pool_plan Stage 2)
    out["items"]["4_draft_gate"] = item4(PV, PA, PT, AM)

    # ===================================================================== 5. gross reference ranges
    out["items"]["5_gross"] = L(RES + "/tag_calibration_gross.json")

    json.dump(out, open(RES + "/tag_calibration.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return out


# ----------------------------------------------------------------------------- item 4 helpers
def trace_meta(etag_dir, side):
    p = os.path.join(etag_dir, "va_extract.json")
    if not os.path.isfile(p):
        return {}
    ex = L(p)
    so = ex["sides"][side]
    tr = so["trace"]
    lab = so.get("labelled") or {}
    return {"gaps": list(zip(tr.get("gap_at_s_mm") or [], tr.get("gaps_mm") or [])),
            "labelled_present": bool(lab.get("present")), "labelled_within_2mm": tr.get("labelled_within_2mm_of_trace")}


def gate_info(v, meta, length):
    """Collect what the gates read from one path's harness record (v = one mask set) and trace metadata."""
    q = v["ts_qc"]
    e = v["entry"]
    lm = q.get("landmarks_mm") or {}
    l1 = e.get("E1z_landmarks_mm") or {}
    return {"origin_mm": q.get("junction_to_TS_SA_mask_mm"), "origin_ostium_mm": q.get("origin_to_TS_SA_mask_mm"),
            "n_C6_C1_E0": q.get("n_C6_to_C1_passed"), "n_C6_C1_E1z": e.get("E1z_n_C6_to_C1"),
            "levels_E0": q.get("foramen_levels_passed") or [], "levels_E1z": e.get("E1z_levels") or [],
            "entry_E0": q.get("foramen_entry_level"), "entry_E1z": e.get("E1z_entry"),
            "lm_E0": lm, "lm_E1z": l1, "lm_Erec": e.get("Erec_landmarks_mm") or {}, "entry_Erec": e.get("Erec_entry"),
            "landmarks_complete": q.get("landmarks_complete"),
            "origin_to_dura_mm": lm.get("dura"), "length_mm": length,
            "v2_band": v.get("v2_band") or {}, "hu": v.get("hu_ts_ct") or {}, "native_hu": v.get("native_ct_hu"),
            "cca": v.get("cca_med_HU"), "bcv_min": v.get("bcv_min_HU"),
            "v3_study": [v["v3loop_study"].get("posterior_excursion_mm"), v["v3loop_study"].get("d_C1_posterior_half_mm")],
            "v3_zcap": [v["v3loop_zcap"].get("posterior_excursion_mm"), v["v3loop_zcap"].get("d_C1_posterior_half_mm")],
            "v3_robust": [v["v3loop_robust"].get("posterior_excursion_mm"), v["v3loop_robust"].get("d_C1_posterior_half_mm")],
            "gaps": meta.get("gaps", []), "labelled_present": meta.get("labelled_present", False),
            "labelled_within_2mm": meta.get("labelled_within_2mm")}


def seg_of(s, lm):
    """V1 / V2 / V3low (C2 -> C1, below a C1 splice) / above, by the landmarks lm (entry, C2, C1)."""
    if lm.get("entry") is not None and s < lm["entry"]:
        return "V1"
    if lm.get("C2") is not None and s < lm["C2"]:
        return "V2"
    if lm.get("C1") is not None and s < lm["C1"]:
        return "V3low"
    return "above_splice"


def apply_gate(gi, variant):
    """variant 'draft' = donor_pool_plan Stage 2 as written (+ the report's additions separately);
    'recalibrated' = this agent's recommendation (notes)."""
    res, review = {}, []
    if variant == "draft":
        lm = {"entry": gi["lm_E0"].get("C6"), "C2": gi["lm_E0"].get("C2"), "C1": gi["lm_E0"].get("C1")}
        n_for = gi["n_C6_C1_E0"]
    else:
        lm = {"entry": gi["lm_Erec"].get("entry"), "C2": gi["lm_Erec"].get("C2"),
              "C1": gi["lm_Erec"].get("C1f") or gi["lm_E0"].get("C1")}
        n_for = gi["n_C6_C1_E0"]
    res["1_origin_le_2mm"] = gi["origin_mm"] is not None and gi["origin_mm"] <= 2.0
    res["2_foramina_ge_4"] = n_for is not None and n_for >= 4
    segs = {}
    for s, g in gi["gaps"]:
        segs.setdefault(seg_of(s, lm), []).append(g)
    tot = sum(g for _, g in gi["gaps"])
    v1 = segs.get("V1", [])
    res["3a_no_V1_bridge"] = len([g for g in v1 if g > (0.0 if variant == "draft" else 3.0)]) == 0
    res["3b_V2_bridge_le_16"] = all(g <= 16 for g in segs.get("V2", []))
    res["3c_V3low_bridge_le_10"] = all(g <= 10 for g in segs.get("V3low", []))
    tot_below = sum(g for k_, v_ in segs.items() if k_ != "above_splice" for g in v_)
    res["3d_total_le_25"] = (tot if variant == "draft" else tot_below) <= 25
    od = gi["origin_to_dura_mm"]
    res["4a_landmarks_complete"] = bool(gi["landmarks_complete"])
    if variant == "draft":
        res["4b_origin_to_dura_150_260"] = od is not None and 150 <= od <= 260
        if od is not None and od > 270:
            review.append("origin->dura > 270 mm")
    else:
        res["4b_origin_to_dura_150_260"] = od is not None and 150 <= od <= 260
        if od is not None and 260 < od <= 290:
            res["4b_origin_to_dura_150_260"] = True
            review.append("origin->dura 260-290 mm")
    b = gi["v2_band"]
    res["5a_V2_band"] = bool(b.get("in_band") if variant == "draft" else b.get("in_band_no_iqr"))
    hu = gi["hu"].get("cervical_over_cca")
    res["5b_HU_ge_0.6_CCA"] = hu is not None and hu >= 0.6
    bcv_ok = gi["bcv_min"] is not None and gi["bcv_min"] < 200
    if not bcv_ok:
        review.append("BCV min %s HU >= 200 (visual check)" % gi["bcv_min"])
    if variant == "draft":
        if gi["labelled_present"] and gi["labelled_within_2mm"] is not None:
            res["6_label_agreement_ge_0.9"] = gi["labelled_within_2mm"] >= 0.9
    else:
        if gi["labelled_present"] and gi["labelled_within_2mm"] is not None and gi["labelled_within_2mm"] < 0.9:
            review.append("Arterial VA label off the trace (%.2f within 2 mm)" % gi["labelled_within_2mm"])
    # additions
    add = {"A1_v3loop_study_15_5": gate_v3(gi["v3_study"], 15, 5), "A2_C1_threaded": "C1" in gi["levels_E0"],
           "A3_bridged_review": any(g > 2.0 for _, g in gi["gaps"])}
    if variant == "recalibrated":
        res["7_v3loop_robust"] = gate_v3(gi["v3_" + REC_V3[0]], REC_V3[1], REC_V3[2])
        res["8_C1_threaded"] = "C1" in gi["levels_E0"]
        if add["A3_bridged_review"]:
            review.append("bridged (max %.1f mm)" % max(g for _, g in gi["gaps"]))
    hard = all(res.values())
    return {"rules": res, "additions_to_draft": add, "pass_hard": hard,
            "pass_with_additions": hard and (variant != "draft" or (add["A1_v3loop_study_15_5"] and add["A2_C1_threaded"])),
            "review": review, "segment_bridges": {k: [rr(x, 1) for x in v] for k, v in segs.items()},
            "failed": [k for k, p in res.items() if not p]}


def item4(PV, PA, PT, AM):
    o = {"draft_rules_as_written": [
        "1 origin within 2 mm of the own TS subclavian mask (junction_to_TS_SA_mask_mm)",
        "2 n_C6_to_C1_passed >= 4 (ts_qc)", "3 bridges: none in V1 (origin -> ts_qc C6), any single V2 bridge <= 16 mm, any "
        "bridge in V3 below the splice (C2 -> ts_qc C1) <= 10 mm, total <= 25 mm",
        "4 landmarks_complete, origin -> dura 150-260 mm, > 270 mm review",
        "5 V2 foramen band (|dx| 12-22, dy 7-19, IQR <= 4 mm from the cord), cervical median HU >= 0.6 x CCA, contralateral "
        "BCV < 200 HU else visual check", "6 labelled_within_2mm_of_trace >= 0.9 where labelled", "7 visual check (not automatic)"],
        "report_additions": ["A1 V3-loop gate (excursion >= ~15, <= ~5 mm from the C1 posterior arch)", "A2 C1 threading",
                             "A3 every bridged side to review"],
        "recalibrated_rules": [
            "1 origin <= 2 mm (unchanged)",
            "2 n_C6_to_C1 >= 4 counted by ts_qc (unchanged: its counts match the verifier's axial test on 39 of 40 TS sides)",
            "3 unchanged limits; V1 / V2 / V3-low segments from the corrected entry landmark (Erec), ts_qc's C2 and the C1 "
            "foramen exit; V1 mask breaks <= 3 mm tolerated (two 1.5 mm voxels); the 25 mm total counts only gaps below the splice (the donor stretch)",
            "4 landmarks_complete; origin -> dura 150-260 pass, 260-290 review (the draft's 260-270 gap closed)",
            "5 V2 band without the IQR term; cervical HU >= 0.6 x CCA; BCV min >= 200 HU -> review (not fail)",
            "6 Arterial label agreement -> review only (labels are unreliable on both GT patients)",
            "7 V3 loop with a robust cord reference: excursion >= 12 mm and d_C1post <= 6 mm",
            "8 C1 threaded (ts_qc)", "bridged (> 2 mm) -> review"],
        "paths": {}}
    dirs = {"VMR": DER + "/arterial/", "A": DER + "/tagnet/arterial/"}
    for grp, J in (("VMR", PV), ("A", PA)):
        for pid, row in J["paths"].items():
            v = row.get("total_v3")
            if not isinstance(v, dict) or "error" in v:
                continue
            meta = trace_meta(dirs[grp] + row["etag"], row["side"]) if row.get("etag") and row["kind"] == "trace" else {}
            if row["kind"] == "vmtk":
                continue        # the gate reads the trace (va_extract), as art_collect does
            gi = gate_info(v, meta, row["length_mm"])
            if row["kind"].startswith("gt"):
                # GT paths have no trace junction: their first point is the ostium (or, for gt_full / the VMR expert
                # path, a point inside the subclavian lumen up to ~10 mm from the TS mask) -> gate the ostium
                gi["origin_mm"] = gi["origin_ostium_mm"]
            rec = {"group": grp, "side": row["side"], "kind": row["kind"], "quality": row.get("quality"),
                   "draft": apply_gate(gi, "draft"), "recalibrated": apply_gate(gi, "recalibrated"),
                   "values": {k: gi[k] for k in ("origin_mm", "n_C6_C1_E0", "entry_E0", "entry_Erec", "lm_Erec",
                                                 "origin_to_dura_mm", "landmarks_complete", "v3_study", "v3_robust", "bcv_min",
                                                 "cca", "labelled_within_2mm")},
                   "v2_band": gi["v2_band"], "hu_ratio_cervical_1p5ct": gi["hu"].get("cervical_over_cca"),
                   "hu_native_cervical": (gi["native_hu"] or {}).get("cervical_med")}
            if grp == "A" and row["kind"] == "trace":
                rec["correct"] = a_correct(row["etag"], row["side"])
            o["paths"][pid] = rec
    ts = {}
    for key, r in PT["paths"].items():
        if not r.get("trace_ok"):
            ts[key] = {"trace_ok": False}
            continue
        sid, sd = key.split()
        meta = trace_meta(DER + "/arterial/ts_" + sid, sd)
        gi = gate_info(r["trace"], meta, r["length_mm"])
        am = AM[sid]["vas"][sd]
        ts[key] = {"draft": apply_gate(gi, "draft"), "recalibrated": apply_gate(gi, "recalibrated"),
                   "study_usable_cervical": am.get("usable_cervical"), "verifier_wrong": key in WRONG6,
                   "study_corrected_usable": bool(am.get("usable_cervical") and key not in WRONG6),
                   "values": {k: gi[k] for k in ("origin_mm", "n_C6_C1_E0", "entry_E0", "entry_Erec", "lm_Erec",
                                                 "origin_to_dura_mm", "landmarks_complete", "v3_study", "v3_robust", "bcv_min",
                                                 "cca", "labelled_within_2mm")},
                   "v2_band": gi["v2_band"], "hu_ratio_cervical": gi["hu"].get("cervical_over_cca"),
                   "radius_vmtk_median_mm": am.get("r_median_mm"),
                   "study_radius_ok": am.get("r_median_mm") is not None and 0.9 <= am["r_median_mm"] <= 2.6}
    o["ts_sides"] = ts
    # counts
    cnt = {}
    for var in ("draft", "recalibrated"):
        rows = {k: v for k, v in ts.items() if v.get("trace_ok", True) and var in v}
        hard = [k for k, v in rows.items() if v[var]["pass_hard"]]
        plus = [k for k, v in rows.items() if v[var]["pass_with_additions"]]
        rev = [k for k in plus if rows[k][var]["review"]]
        fails = {}
        for k, v in rows.items():
            for f in v[var]["failed"]:
                fails[f] = fails.get(f, 0) + 1
        subj = sorted(set(k.split()[0] for k in plus))
        both = sorted(s for s in subj if all(("%s %s" % (s, x)) in plus for x in ("R", "L")))
        cnt[var] = {"traced_sides": len(rows), "pass_hard": len(hard), "pass_with_additions": len(plus),
                    "pass_with_additions_list": plus, "of_which_flagged_review": rev, "subjects": len(subj),
                    "subjects_both_sides": both, "rule_fail_counts": fails,
                    "agree_with_study_corrected_15": "%d of 15 passed; %d passed outside the 15" % (
                        sum(1 for k in plus if rows[k]["study_corrected_usable"]),
                        sum(1 for k in plus if not rows[k]["study_corrected_usable"])),
                    "wrong6_passing": [k for k in plus if rows[k]["verifier_wrong"]],
                    "with_study_radius_rule": len([k for k in plus if rows[k]["study_radius_ok"]])}
    o["ts_counts"] = cnt
    return o


if __name__ == "__main__":
    main()
    print("wrote", RES + "/tag_calibration.json")
