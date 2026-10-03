#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_tally.py (= /sp/v5_va/scaleup_tally.py in the build containers)
"""scaleup_tally.py -- the v5 VA donor TALLY and STOP RULE (scaleup_fixes, 2026-10-02). Run any time; read-only on every input
(it never writes a results file of a pilot or a lane). eve-training-fixed (python3 + numpy) -- it grades results files
itself through scaleup_v2_grade.py (subprocess, SCALEUP_GRADE_MERGE=0, output under results/scaleup_tally_work/).

Inputs (per case and side, later sources override earlier ones for the SAME source + case):
  1. pilot           results/pilot_proc_{ia172,song}_grade.json (the pilot's own grade rows; walls ct_seg4)
  2. pilot under v2  results/scaleup_fixes_cases_{ia172,song}.json (pilot cases re-run with pipeline v2 because a fix changed
                     them: V1 rescue, pinch fix, trace selection; walls ct_seg/ct_seg4v2) -> graded here
  3. scale-up lanes  results/scaleup_{ia172,song}.json (new cases; walls ct_seg4) -> graded here
  4. VMR 0248        results/pilot_donor_ranking.json -> vmr0248_pair_reference (eligible; same harness)
  5. by-eye verdicts results/scaleup_eye_rejections.json (optional; written by the lanes): {"rejections": [{"source": "song",
                     "case": "AHMU1218xxx", "side": "R", "why": "...", "by": "<agent prefix>", "when": "..."}]} -- a listed
                     side counts as failing QC ("rejected by eye: <why>"); for flagged sides (review / rescue) seen and
                     kept, add {"..., "verdict": "kept"} entries (recorded, not used to change eligibility).
ELIGIBILITY (the build's rule, V5_BUILD_LOG 03:35): a donor PAIR (both VAs of one case) is eligible when BOTH sides pass the
  recalibrated QC gate (qc_recalibrated.passes) AND both have a valid ct_seg4 wall (gates V1-V6 pass, acquisition voxel:
  wall.gates_V1_V6_whole, and the wall was measured: whole-VA area > 0 -- 08:50, Song 138 R had gates "pass" on a wall whose VMTK centreline was 0.7 mm, so measure saw nothing). A side that passes only through the provisional in-sample limits (V1 bridge <= 3 mm, the dropped
  V2-band IQR term = recalibrated pass but two-patient-backed-only fail) is flagged "review: provisional".
COARSENESS: score = mean over R / L of the contact-free donor-stretch union > 0.3 mm (pilot_decide's ranking key; min side
  reported); all-bins donor-stretch union > 0.3 mm per side (wall.donor_stretch_origin_to_C1m25 0.3mm, else the 10 mm-bin
  value over the stretch); dense host RVA mesh = 0.027.
STOP RULE: eligible pairs >= 30 AND >= 25 eligible pairs with all-bins donor-stretch union > 0.027 on BOTH sides; batch 1 =
  18 + 18 new cases (selection order of pilot_next_cases.json), then re-tally every 4 cases, cap 25 new cases per lane.
Writes results/scaleup_tally.json (+ a console table).
"""
import json
import os
import subprocess
import sys
import time

SP = "/sp/v5_va"
RES = SP + "/results"
WORK = RES + "/scaleup_tally_work"
DER = "/repo/carotid_data/va_sources/derived/thin_slice"
HOST_DENSE = 0.027
NEED, NEED_SPARES = 25, 30
STOP_ELIGIBLE, STOP_COARSE = 30, 25
BATCH1, CAP = 18, 25
os.makedirs(WORK, exist_ok=True)


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


def mtime(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))) if os.path.isfile(p) else None


def grade_rows(src, resf, tag, label):
    """grade a results file with scaleup_v2_grade.py into WORK (never merged back); returns its rows."""
    out = os.path.join(WORK, "%s_grade.json" % label)
    env = dict(os.environ, PILOT_SOURCE=src, SCALEUP_RESULTS=resf, PILOT_SEG_TAG=tag, SCALEUP_GRADE_OUT=out,
               SCALEUP_GRADE_MERGE="0")
    rc = subprocess.call([sys.executable, SP + "/scaleup_v2_grade.py"], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)
    G = L(out) if rc == 0 else None
    return (G or {}).get("rows") or [], rc, out


inputs = []
side_rows = {}      # (source, case) -> {"R": row, "L": row, "origin": ...}
case_origin = {}
for src in ("ia172", "song"):
    p = RES + "/pilot_proc_%s_grade.json" % src
    G = L(p) or {}
    inputs.append({"what": "pilot grade", "source": src, "file": p, "mtime": mtime(p), "rows": len(G.get("rows") or [])})
    for r in G.get("rows") or []:
        side_rows.setdefault((src, r["case"]), {})[r["side"]] = dict(r, _origin="pilot", _tag="ct_seg4")
        case_origin[(src, r["case"])] = "pilot"
for src in ("ia172", "song"):
    for label, resf, tag, origin in (("scaleup_fixes_cases_%s" % src, RES + "/scaleup_fixes_cases_%s.json" % src, "ct_seg4v2",
                                      "pilot case re-run under pipeline v2"),
                                     ("scaleup_%s" % src, RES + "/scaleup_%s.json" % src, "ct_seg4", "scale-up lane (new case)")):
        if not os.path.isfile(resf):
            inputs.append({"what": origin, "source": src, "file": resf, "present": False})
            continue
        rows, rc, gout = grade_rows(src, resf, tag, label)
        J = L(resf) or {}
        inputs.append({"what": origin, "source": src, "file": resf, "mtime": mtime(resf), "graded_to": gout, "grade_rc": rc,
                       "rows": len(rows), "cases": sorted((J.get("cases") or {}).keys())})
        by = {}
        for r in rows:
            by.setdefault(r["case"], {})[r["side"]] = dict(r, _origin=origin, _tag=tag)
        for c, sides in by.items():
            prev = case_origin.get((src, c))
            if origin.startswith("scale-up") and prev and prev != origin:
                # a lane case that is also a pilot case would be a duplicate patient: keep the pilot / v2 record, flag it
                inputs[-1].setdefault("ignored_duplicates", []).append(c)
                continue
            side_rows[(src, c)] = sides
            case_origin[(src, c)] = origin


EYE_F = RES + "/scaleup_eye_rejections.json"
EYE = {}
EYE_KEPT = {}
for e in (L(EYE_F) or {}).get("rejections") or []:
    try:
        key = (e["source"], e["case"], e["side"])
    except (KeyError, TypeError):
        continue
    if str(e.get("verdict", "reject")).lower() == "kept":
        EYE_KEPT[key] = e
    else:
        EYE[key] = e
inputs.append({"what": "by-eye verdicts", "file": EYE_F, "present": os.path.isfile(EYE_F), "mtime": mtime(EYE_F),
               "rejected_sides": ["%s/%s/%s" % k for k in sorted(EYE)], "kept_sides": ["%s/%s/%s" % k for k in sorted(EYE_KEPT)]})


def side_view(r, key=None):
    q = r.get("qc_recalibrated") or {}
    w = r.get("wall") or {}
    don = g(w, "donor_stretch_origin_to_C1m25", "0.3mm")
    allb = don if don is not None else g(w, "contact_free", "union_0.3_all_bins")
    prov = bool(q.get("passes")) and not bool(g(r, "qc_two_patient_backed_only", "passes"))
    tr = g(w, "label_truncation", "truncated")
    if key in EYE:
        q = dict(q, passes=False, status="rejected_by_eye",
                 failed=list(q.get("failed") or []) + ["rejected by eye: %s" % EYE[key].get("why", "")])
    return {"status": q.get("status"), "qc_pass": bool(q.get("passes")), "fails": q.get("failed"), "review": q.get("review") or [],
            "eye_kept": key in EYE_KEPT,
            "provisional_only": prov, "backed_only_fails": g(r, "qc_two_patient_backed_only", "failed"),
            "has_wall": bool(w), "wall_valid": bool(w.get("gates_V1_V6_whole") and (g(w, "whole", "area_mm2") or 0) > 0) if w else False,
            "wall_measured": bool((g(w, "whole", "area_mm2") or 0) > 0) if w else False,
            "genus_whole": w.get("genus_whole") if w else None,
            "donor_u03_all_bins": rr(allb), "donor_u03_all_bins_source": "donor stretch (exact)" if don is not None else "10 mm bins",
            "donor_u03_contact_free": rr(g(w, "contact_free", "union_0.3_contact_free")),
            "contact_free_share": g(w, "contact_free", "contact_free_area_share"),
            "partial_truncated_wall": g(w, "contact_free", "partial_truncated_wall"), "wall_truncated": bool(tr),
            "whole_u03": g(w, "whole", "0.3mm"), "trace_set": r.get("trace_set") or "tsba",
            "v1_rescue_used": bool(r.get("v1_rescue_used")),
            "pinch_fix_triggered": bool(g(w, "pinch_fix", "triggered")), "origin": r.get("_origin"), "wall_tag": r.get("_tag")}


pairs, not_eligible = [], []
for (src, c), sides in sorted(side_rows.items()):
    if set(sides) != {"R", "L"}:
        not_eligible.append({"source": src, "case": c, "why": "only side(s) %s graded" % sorted(sides)})
        continue
    V = {s: side_view(sides[s], (src, c, s)) for s in ("R", "L")}
    why = []
    for s in ("R", "L"):
        v = V[s]
        if not v["qc_pass"]:
            why.append("%s fails QC (%s)" % (s, "; ".join(v["fails"] or [])))
        elif not v["has_wall"]:
            why.append("%s has no ct_seg4 wall" % s)
        elif not v["wall_measured"]:
            why.append("%s wall not measured (whole-VA area 0: VMTK centreline degenerate) -> run scaleup_ctseg_fallback.sh" % s)
        elif not v["wall_valid"]:
            why.append("%s wall gates V1-V6 fail (genus %s)" % (s, v["genus_whole"]))
    rec = {"source": src, "case": c, "origin": case_origin.get((src, c)), "R": V["R"], "L": V["L"]}
    if why:
        rec["why"] = why
        rec["single_side_passing"] = [s for s in ("R", "L") if V[s]["qc_pass"] and V[s]["wall_valid"]]
        not_eligible.append(rec)
        continue
    cf = [V["R"]["donor_u03_contact_free"], V["L"]["donor_u03_contact_free"]]
    ab = [V["R"]["donor_u03_all_bins"], V["L"]["donor_u03_all_bins"]]
    flags = []
    for s in ("R", "L"):
        v = V[s]
        if v["provisional_only"]:
            flags.append("%s review: passes only via provisional limits (%s)" % (s, "; ".join(v["backed_only_fails"] or [])))
        if v["status"] == "donor_review":
            flags.append("%s review: %s" % (s, "; ".join(v["review"])))
        if v["wall_truncated"] or v["partial_truncated_wall"]:
            flags.append("%s wall truncated (partial score)" % s)
        if v["contact_free_share"] is not None and v["contact_free_share"] < 0.2:
            flags.append("%s contact-free score rests on %.0f %% of the stretch" % (s, 100 * v["contact_free_share"]))
        if v["v1_rescue_used"]:
            flags.append("%s V1 mask-break rescue (trace %s)" % (s, v["trace_set"]))
        if v["trace_set"] not in ("tsba", None):
            flags.append("%s trace set %s" % (s, v["trace_set"]))
    eye_needed = sorted({f.split(" ")[0] for f in flags if " review" in f or "rescue" in f})
    rec["eye_check_needed_sides"] = eye_needed
    rec["eye_kept_sides"] = [s for s in ("R", "L") if V[s]["eye_kept"]]
    rec["eye_check_pending_sides"] = [s for s in eye_needed if not V[s]["eye_kept"]]
    rec.update({"score_mean_contact_free": rr(sum(cf) / 2) if None not in cf else None,
                "score_min_side_contact_free": rr(min(cf)) if None not in cf else None,
                "R_L_contact_free": cf, "R_L_all_bins": ab,
                "all_bins_gt_dense_host_both": bool(None not in ab and min(ab) > HOST_DENSE),
                "contact_free_ge_dense_host_both": bool(None not in cf and min(cf) >= HOST_DENSE),
                "flags": flags, "eligible": True})
    pairs.append(rec)

# VMR 0248 (eligible reference donor; same harness)
rk = L(RES + "/pilot_donor_ranking.json") or {}
vmr = rk.get("vmr0248_pair_reference") or {}
inputs.append({"what": "VMR 0248 reference", "file": RES + "/pilot_donor_ranking.json", "mtime": mtime(RES + "/pilot_donor_ranking.json")})
if vmr:
    cf = vmr.get("R_L_contact_free_donor_u03") or [None, None]
    ab = vmr.get("R_L_donor_u03_all_bins") or [None, None]
    pairs.append({"source": "vmr0248", "case": "VMR0248", "origin": "pilot_decide reference (eligible)",
                  "score_mean_contact_free": vmr.get("score_mean_contact_free_donor_u03"),
                  "score_min_side_contact_free": vmr.get("score_min_side"), "R_L_contact_free": cf, "R_L_all_bins": ab,
                  "all_bins_gt_dense_host_both": bool(None not in ab and min(ab) > HOST_DENSE),
                  "contact_free_ge_dense_host_both": bool(None not in cf and min(cf) >= HOST_DENSE),
                  "flags": ["left V1 (first 40 mm) vein contact; research use, cite Wilson 2013"], "eligible": True})
pairs.sort(key=lambda p: -(p["score_mean_contact_free"] if p["score_mean_contact_free"] is not None else -1))
for i, p in enumerate(pairs):
    p["rank"] = i + 1

n_el = len(pairs)
n_coarse_all = sum(p["all_bins_gt_dense_host_both"] for p in pairs)
n_coarse_cf = sum(p["contact_free_ge_dense_host_both"] for p in pairs)
lanes = {}
for src in ("ia172", "song"):
    J = L(RES + "/scaleup_%s.json" % src) or {}
    cs = sorted((J.get("cases") or {}).keys())
    lanes[src] = {"new_cases_in_results": len(cs), "cases": cs, "batch1_done": len(cs) >= BATCH1, "cap_reached": len(cs) >= CAP}
met = n_el >= STOP_ELIGIBLE and n_coarse_all >= STOP_COARSE
by_src = {}
for p in pairs:
    by_src[p["source"]] = by_src.get(p["source"], 0) + 1
out = {"what": __doc__.split("\n\n")[0], "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": inputs,
       "rules": {"eligibility": "both sides pass the recalibrated QC gate (provisional-limit passes flagged review) AND both "
                                "have a valid ct_seg4 wall (gates V1-V6, acquisition voxel; measured: whole-VA area > 0)",
                 "coarseness": "score = mean contact-free donor-stretch union > 0.3 mm (min side reported); all-bins donor-stretch "
                               "union > 0.3 mm per side; dense host RVA mesh 0.027",
                 "stop": "eligible >= %d AND >= %d eligible pairs with all-bins donor-stretch union > %.3f on BOTH sides; batch 1 = "
                         "%d + %d new cases, then re-tally every 4, cap %d per lane" % (STOP_ELIGIBLE, STOP_COARSE, HOST_DENSE,
                                                                                         BATCH1, BATCH1, CAP)},
       "counts": {"eligible_pairs": n_el, "eligible_by_source": by_src,
                  "eligible_all_bins_gt_dense_host_both_sides": n_coarse_all,
                  "eligible_contact_free_ge_dense_host_both_sides": n_coarse_cf,
                  "eligible_with_review_flags": sum(bool(p["flags"]) for p in pairs),
                  "eligible_with_provisional_side": sum(any("provisional" in f for f in p["flags"]) for p in pairs),
                  "eligible_with_eye_check_pending": sum(bool(p.get("eye_check_pending_sides")) for p in pairs),
                  "sides_rejected_by_eye": len(EYE),
                  "need_pairs": NEED, "need_with_spares": NEED_SPARES,
                  "cases_graded": len(side_rows), "cases_not_eligible": len(not_eligible)},
       "stop_rule": {"met": met, "eligible_ge_%d" % STOP_ELIGIBLE: n_el >= STOP_ELIGIBLE,
                     "coarse_all_bins_both_ge_%d" % STOP_COARSE: n_coarse_all >= STOP_COARSE,
                     "short_by": {"eligible": max(0, STOP_ELIGIBLE - n_el), "coarse_all_bins_both": max(0, STOP_COARSE - n_coarse_all)},
                     "lanes": lanes,
                     "next": ("STOP: rule met" if met else
                              "continue: both lanes at the cap (25) -- stop and report the shortfall" if all(lanes[s]["cap_reached"] for s in lanes)
                              else "continue: finish batch 1 (18 per lane)" if not all(lanes[s]["batch1_done"] for s in lanes)
                              else "continue in order, re-tally after every 4 cases (cap 25 per lane)")},
       "eligible_pairs": pairs, "not_eligible": not_eligible}
json.dump(out, open(RES + "/scaleup_tally.json", "w"), indent=1, default=str)
print("ELIGIBLE %d (by source %s); all-bins > %.3f both sides %d; contact-free >= %.3f both %d; STOP %s %s" % (
    n_el, by_src, HOST_DENSE, n_coarse_all, HOST_DENSE, n_coarse_cf, met, out["stop_rule"]["next"]))
for p in pairs:
    print("#%-2d %-6s %-12s %-30s cf %s (min %s) all %s %s | %s" % (p["rank"], p["source"], p["case"], p.get("origin", "")[:30],
          p["score_mean_contact_free"], p["score_min_side_contact_free"], p["R_L_all_bins"],
          "COARSE" if p["all_bins_gt_dense_host_both"] else "smooth", "; ".join(p["flags"])[:160]))
print("not eligible:")
for r in not_eligible:
    print("  %s %s [%s]: %s" % (r["source"], r["case"], r.get("origin"), "; ".join(r.get("why") if isinstance(r.get("why"), list) else [r.get("why", "")])[:200]))
for i in inputs:
    print("input", json.dumps({k: v for k, v in i.items() if k != "cases"}))
