#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_qc2.py (= /sp/v5_va/pilot_proc_ia172_qc2.py in the build containers)
"""pilot_proc_ia172_qc2.py <case> -- the RECALIBRATED Stage-2 QC gate (notes/tag_calibration.md) plus the brief's checks,
for both VA sides of one 172-series pilot case. Replaces the first instance's pilot_proc_ia172_qc.py (draft thresholds,
written before tag_calibration.md existed; never run). neve-build-stagea. Writes <der>/qc/qc.json (env PILOT_TRACE_DIR=<dir>:
grade arterial/<dir> instead of arterial/tsba, -> qc/qc_<dir>.json; used for the no-label rescue retrace tsba_nl).

Harness = the calibration's own code, imported UNCHANGED:
  tag_calibration_paths.one(...)   ts_qc (art_extract), per-level profiles, entry rules, V3 loop (study / zcap / ROBUST
                                   cord), V2 band, HU on the TS 1.5 mm CT copy, CCA / BCV HU (ts_screen.hu_stats rule)
  tag_calibration.augment(...)     the recommended entry rule "Erec" (C6 landmark; C1f = exit of the first C1 run)
  tag_calibration.gate_info / apply_gate(gi, "recalibrated")   the recommended gate; "draft" also reported:
     1 origin <= 2 mm | 2 foramina (C6..C1, ts_qc) >= 4 | 3a V1 bridges <= 3 mm (V1 to the Erec C6) | 3b V2 bridge <= 16 |
     3c V3-low (C2 -> C1f) bridge <= 10 | 3d total <= 25 mm counted below the splice | 4a landmarks complete |
     4b origin -> dura 150-260 (260-290 review) | 5a V2 band |dx| 12-22, dy 7-19 (no IQR term) | 5b cervical HU >= 0.6 CCA |
     BCV >= 200 -> review | 6 Arterial label agreement -> review | 7 V3 loop, robust cord, excursion >= 12, d_C1post <= 6 |
     8 C1 threaded | bridged > 2 mm -> review
  The input gate_info reads: the trace (va_extract tsba = art_extract + the basilar fix), its junction, its bridges.
Bridges: art_extract stores the bridge START positions on the final (trimmed, smoothed) path (gap_at_s_mm) and the bridge
  LENGTHS on the raw untrimmed path (gaps_mm); when a raw gap lies inside the trimmed subclavian part there is one more
  length than start, so the leading extra lengths are dropped (the first instance's rule); calibration's trace_meta zips
  them unaligned. Both raw lists are kept in the record.
Added for the brief (reported; the ones marked HARD enter the donor verdict):
  * HU on the NATIVE CT (trilinear) along origin -> C1f: median / CCA median (native, TS CCA masks eroded), fraction of
    points >= 700 HU (bone / dense venous contrast; > 2 % -> review), contralateral BCV on the native CT (>= 200 -> review);
  * V3 loop also with the study's 15 / 5 (the brief's "~15 / ~5"), reported only;
  * VMTK (art_vmtk_one on the cut-out stage-A surface): ok, MISR median, x 1.10 (tag_calibration item 3, native factor),
    HARD: VMTK ok and corrected radius 0.9-2.6 mm;
  * distal (above C1): art_collect-style plausibility (V4 length, lateral excursion, end above z_FM, distal bridges, the
    traces join) -> distal_ok; reported separately;
  * the unchanged art_extract trace (arterial/ts) compared with tsba.
DONOR VERDICT (the brief): the cervical stretch origin -> at least the C2-C1 loop is clean = recalibrated rules 1, 2, 3a-3d,
  5a, 5b, 7, 8 pass + VMTK HARD check + origin -> C1f length 100-230 mm. Rule 4 (dura reached, origin -> dura) is reported
  with the distal grade, since strategy B discards the trace above the V3 splice.
"""
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import art_extract as AE  # noqa: E402  (unchanged)
import tag_calibration_paths as TP  # noqa: E402  (unchanged: the calibration harness)
import tag_calibration as TC  # noqa: E402  (unchanged: augment, gate_info, apply_gate)

case = sys.argv[1]
SRC = "/repo/carotid_data/va_sources/thin_slice/ia172/%s" % case
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s" % case
TS15 = os.path.join(DER, "totalseg_1p5")
OUTD = os.path.join(DER, "qc")
os.makedirs(OUTD, exist_ok=True)
MISR_FACTOR = 1.10
TRD = os.environ.get("PILOT_TRACE_DIR", "tsba")   # rescue retraces (e.g. tsba_nl) are graded the same way -> qc/qc_<dir>.json


def rr(x, n=2):
    return None if x is None else round(float(x), n)


def arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


t0 = time.time()
ct = nib.load(os.path.join(SRC, "ct.nii.gz"))
CTA = ct.affine
CTV = np.asanyarray(ct.dataobj)
CTAi = np.linalg.inv(CTA)


def hu_at(P, order=1):
    I = (CTAi @ np.c_[P, np.ones(len(P))].T)[:3]
    return map_coordinates(CTV, I, order=order, mode="nearest").astype(float)


def mask_hu_native(nm):
    p = os.path.join(TS15, "segmentations", nm + ".nii.gz")
    if not os.path.isfile(p):
        return None
    im = nib.load(p)
    m = np.asanyarray(im.dataobj) > 0
    if m.sum() < 5:
        return None
    e = ndimage.binary_erosion(m)
    use = e if e.sum() >= 20 else m
    W = AE.i2w(im.affine, np.argwhere(use).astype(float))
    h = hu_at(W, order=0)
    return {"median": rr(np.median(h), 1), "p10": rr(np.percentile(h, 10), 1), "p90": rr(np.percentile(h, 90), 1),
            "n": int(len(h)), "eroded": bool(use is e)}


vessels = {nm: mask_hu_native(nm) for nm in (
    "common_carotid_artery_right", "common_carotid_artery_left", "subclavian_artery_right", "subclavian_artery_left",
    "brachiocephalic_vein_right", "brachiocephalic_vein_left", "superior_vena_cava", "aorta", "brachiocephalic_trunk")}
cca = [vessels[k]["median"] for k in ("common_carotid_artery_right", "common_carotid_artery_left") if vessels.get(k)]
cca_nat = float(np.median(cca)) if cca else None
bR, bL = vessels.get("brachiocephalic_vein_right"), vessels.get("brachiocephalic_vein_left")
inj, contra = None, None
if bR and bL:
    inj = "right" if bR["median"] >= bL["median"] else "left"
    contra = bL["median"] if inj == "right" else bR["median"]

ex = json.load(open(os.path.join(DER, "arterial", TRD, "va_extract.json")))
exr = None
if os.path.isfile(os.path.join(DER, "arterial/ts/va_extract.json")):
    exr = json.load(open(os.path.join(DER, "arterial/ts/va_extract.json")))
meta = json.load(open(os.path.join(SRC, "meta.json")))
conf = ex.get("confluence") or {}
out = {"case": case, "what": __doc__, "thresholds": "tag_calibration.md RECOMMENDED (apply_gate 'recalibrated', imported "
       "unchanged); the draft gate is reported beside it", "misr_factor": MISR_FACTOR,
       "vessel_HU_native_at_TS_total_v3_masks": vessels, "cca_median_HU_native": rr(cca_nat, 1),
       "injection_side_by_bcv": inj, "contralateral_bcv_median_HU_native": contra,
       "acquisition_voxel_mm": meta["acquisition_voxel_mm"],
       "extract": {k: ex.get(k) for k in ("x_mid_mm", "x_mid_skull_mm", "z_fm_mm", "z_fm_from", "midline_from",
                                          "cord_x_range_mm", "basilar_on_mask", "ba_target", "arterial_labels",
                                          "mask_components", "class_lengths_mm", "arterial_times_s", "arterial_peak_ram_gb",
                                          "grid_mm", "mask_shape", "seconds")},
       "confluence": {k: v for k, v in conf.items() if k != "basilar_points"}, "sides": {}}
if exr is not None:
    out["extract_unchanged_tool"] = {"basilar_on_mask": exr.get("basilar_on_mask"), "ba_target": exr.get("ba_target"),
                                     "confluence_method": (exr.get("confluence") or {}).get("method")}

for side in ("R", "L"):
    so = ex["sides"][side]
    tr = so.get("trace", {})
    rec = {"trace_ok": bool(tr.get("ok")), "trace_why": tr.get("why"), "start_kind": tr.get("start_kind")}
    out["sides"][side] = rec
    if not tr.get("ok"):
        rec.update({"passes_as_v5_donor": False, "status": "fail", "gate_fail_reasons": ["no trace: %s" % tr.get("why")],
                    "distal_ok": False})
        continue
    P = np.asarray(tr["points"], float)
    s = arclen(P)
    L = float(s[-1])
    # --- calibration harness record (unchanged code)
    v = TP.one("ia172", P, tr["start_on_sa_world"], side, "total_v3", TS15)
    TC.augment(v)
    st, ln = list(tr.get("gap_at_s_mm") or []), list(tr.get("gaps_mm") or [])
    ln_al = ln[len(ln) - len(st):] if len(ln) > len(st) else ln
    gaps = list(zip(st, ln_al))
    lab = so.get("labelled") or {}
    tmeta = {"gaps": gaps, "labelled_present": bool(lab.get("present")),
             "labelled_within_2mm": tr.get("labelled_within_2mm_of_trace")}
    gi = TC.gate_info(v, tmeta, L)
    grec = TC.apply_gate(gi, "recalibrated")
    gdraft = TC.apply_gate(gi, "draft")
    q = v["ts_qc"]
    lmE = v["entry"].get("Erec_landmarks_mm") or {}
    lm0 = q.get("landmarks_mm") or {}
    lm = {"C6": lmE.get("entry"), "C2": lmE.get("C2"), "C1": lmE.get("C1f") or lm0.get("C1"), "dura": lm0.get("dura"),
          "end": rr(L, 1)}
    rec.update({"length_trace_mm": rr(L, 1), "junction_world_mm": tr.get("start_on_sa_world"),
                "s_leave_sa_mm": tr.get("s_leave_sa_mm"), "r_edt_median_mm": tr.get("r_edt_median_mm"),
                "landmarks_used_mm": lm,
                "landmarks_note": "C6 = Erec entry landmark; C2 = ts_qc; C1 = C1f (exit of the first C1 run after C2, VMR's "
                                  "convention) else ts_qc; dura = ts_qc (first s above z_FM = C1 top + 6 mm after C1)",
                "landmarks_ts_qc_raw_mm": lm0, "entry_level_ts_qc": q.get("foramen_entry_level"),
                "entry_level_Erec": v["entry"].get("Erec_entry"), "foramen_levels_passed": q.get("foramen_levels_passed"),
                "n_C6_to_C1_passed": q.get("n_C6_to_C1_passed"), "landmarks_complete": q.get("landmarks_complete"),
                "junction_to_TS_SA_mm": q.get("junction_to_TS_SA_mask_mm"), "origin_to_TS_SA_mm": q.get("origin_to_TS_SA_mask_mm"),
                "v3loop_study": v["v3loop_study"], "v3loop_zcap": v["v3loop_zcap"], "v3loop_robust": v["v3loop_robust"],
                "cord_top": v["cord_top"], "v2_band": v["v2_band"], "hu_ts_1p5_ct": v["hu_ts_ct"],
                "cca_med_HU_ts_1p5": v["cca_med_HU"], "bcv_min_HU_ts_1p5": v["bcv_min_HU"],
                "bridges_raw": {"gap_at_s_mm": st, "gaps_mm": ln, "aligned_pairs": [[rr(a, 1), rr(b, 2)] for a, b in gaps]},
                "labelled_VA_present": bool(lab.get("present")), "labelled_length_mm": lab.get("length_mm"),
                "labelled_within_2mm_of_trace": tr.get("labelled_within_2mm_of_trace"),
                "gap_vessel_like_HU_frac": tr.get("gap_vessel_like_HU_frac"), "surface": so.get("surface"),
                "gate_recalibrated": grec, "gate_draft": gdraft})
    # --- native-CT phase / contamination, origin -> C1f
    c1 = lm.get("C1")
    m1 = s <= (c1 if c1 is not None else L)
    h = hu_at(P[m1])
    rec["trace_HU_native_origin_to_C1"] = {"median": rr(np.median(h), 0), "p10": rr(np.percentile(h, 10), 0),
                                           "p95": rr(np.percentile(h, 95), 0), "frac_ge700": rr(np.mean(h >= 700), 4),
                                           "ratio_to_CCA": rr(np.median(h) / cca_nat, 3) if cca_nat else None}
    hall = hu_at(P)
    rec["trace_HU_native_whole"] = {"median": rr(np.median(hall), 0), "frac_ge700": rr(np.mean(hall >= 700), 4)}
    # --- bridges below / above the splice with their native HU
    rows = []
    for a, b in gaps:
        sel = (s >= a) & (s <= a + b)
        hv = hu_at(P[sel]) if sel.any() else np.array([np.nan])
        rows.append({"s_mm": rr(a, 1), "length_mm": rr(b, 1), "segment": TC.seg_of(a, {"entry": lm["C6"], "C2": lm["C2"],
                                                                                      "C1": lm["C1"]}),
                     "z_mm": rr(P[int(np.argmin(np.abs(s - a)))][2], 1), "HU_median_native": rr(np.nanmedian(hv), 0)})
    rec["bridges"] = rows
    # --- VMTK line of the cut-out surface
    vp = os.path.join(DER, "arterial", TRD, "%sva_stageA_vmtk.json" % side.lower())
    vm_ok, r_med = False, None
    if os.path.isfile(vp):
        vj = json.load(open(vp))
        Q = np.asarray(vj.get("points") or [], float)
        if len(Q) > 5:
            if np.linalg.norm(Q[-1] - P[0]) < np.linalg.norm(Q[0] - P[0]):
                Q = Q[::-1]
            vm_ok = bool(vj.get("ok"))
            r_med = float(np.median(vj["radii"]))
            q2 = AE.ts_qc({"P": Q, "start_on_sa_world": Q[0]}, TP._TS[TS15][0], side, TP._TS[TS15][0]["z_c1_top"] + 6.0)
            rec["vmtk"] = {"ok": vm_ok, "length_mm": rr(vj.get("length_mm"), 1), "r_misr_median_mm": rr(r_med, 3),
                           "r_corrected_median_mm": rr(MISR_FACTOR * r_med, 3), "note": vj.get("note"),
                           "landmarks_ts_qc_mm": q2.get("landmarks_mm", {}),
                           "v3loop_robust": TP.v3_zcap(Q, TP._TS[TS15][0], TP._TS[TS15][1], robust=True),
                           "trace_to_vmtk_median_mm": rr(np.median(cKDTree(Q).query(P)[0]), 2),
                           "trace_to_vmtk_p95_mm": rr(np.percentile(cKDTree(Q).query(P)[0], 95), 2)}
    # --- distal (above C1)
    du = lm.get("dura")
    rec["V4_length_trace_mm"] = rr(L - du, 1) if du is not None else None
    z_fm = v["z_fm_mm"]
    if du is not None:
        mv4 = s >= du + 10
        rec["V4_max_lateral_mm"] = rr(np.abs(P[mv4, 0] - ex.get("x_mid_mm", 0)).max(), 1) if mv4.any() else 0.0
    rec["end_above_zfm_mm"] = rr(P[-1, 2] - z_fm, 1)
    if exr is not None and exr["sides"][side].get("trace", {}).get("ok"):
        Pref = np.asarray(exr["sides"][side]["trace"]["points"], float)
        d1 = cKDTree(Pref).query(P)[0]
        rec["unchanged_tool_trace"] = {"length_mm": rr(arclen(Pref)[-1], 1), "tsba_points_within_1mm": rr(np.mean(d1 <= 1.0), 3),
                                       "first_divergence_s_mm": rr(s[np.argmax(d1 > 2.0)], 1) if (d1 > 2.0).any() else None}

    # ------------------------------------------------------------------ verdicts
    R = grec["rules"]
    cerv_rules = ["1_origin_le_2mm", "2_foramina_ge_4", "3a_no_V1_bridge", "3b_V2_bridge_le_16", "3c_V3low_bridge_le_10",
                  "3d_total_le_25", "5a_V2_band", "5b_HU_ge_0.6_CCA", "7_v3loop_robust", "8_C1_threaded"]
    why = [k for k in cerv_rules if not R.get(k)]
    review = list(grec["review"])
    vm_hard = vm_ok and r_med is not None and 0.9 <= MISR_FACTOR * r_med <= 2.6
    if not vm_hard:
        why.append("VMTK ok %s, corrected MISR %s" % (vm_ok, rr(None if r_med is None else MISR_FACTOR * r_med, 2)))
    c1len_ok = c1 is not None and 100.0 <= c1 <= 230.0
    if not c1len_ok:
        why.append("origin -> C1 %s mm (100-230)" % rr(c1, 1))
    hn = rec["trace_HU_native_origin_to_C1"]
    if hn["frac_ge700"] > 0.02:
        review.append("%.1f %% of origin->C1 points >= 700 HU (native)" % (100 * hn["frac_ge700"]))
    if hn["ratio_to_CCA"] is not None and hn["ratio_to_CCA"] < 0.6:
        review.append("native origin->C1 HU %.2f x CCA" % hn["ratio_to_CCA"])
    if contra is not None and contra >= 200:
        review.append("contralateral BCV %.0f HU on the native CT" % contra)
    big = [g for g in rows if g["length_mm"] > 2.0 and g["segment"] != "above_splice"]
    if big:
        review.append("bridge(s) > 2 mm below the splice: " + ", ".join("%.1f mm at s %.0f (%s, %s HU)" % (
            g["length_mm"], g["s_mm"], g["segment"], g["HU_median_native"]) for g in big))
    rec["v3loop_study_15_5_pass"] = TC.gate_v3(gi["v3_study"], 15, 5)
    donor = len(why) == 0
    dwhy = []
    if not R.get("4a_landmarks_complete"):
        dwhy.append("dura landmark not reached")
    if not R.get("4b_origin_to_dura_150_260"):
        dwhy.append("origin -> dura %s mm" % lm.get("dura"))
    if not R.get("7_v3loop_robust"):
        dwhy.append("V3 loop failed")
    v4 = rec.get("V4_length_trace_mm")
    if v4 is None or v4 > 60:
        dwhy.append("V4 %s mm" % v4)
    if (rec.get("V4_max_lateral_mm") or 0) > 25:
        dwhy.append("V4 runs %.0f mm off the midline" % rec["V4_max_lateral_mm"])
    e = rec["end_above_zfm_mm"]
    if e is None or not (0 < e <= 45):
        dwhy.append("end %s mm above z_FM" % e)
    if any(g["length_mm"] > 12 for g in rows if g["segment"] == "above_splice"):
        dwhy.append("bridge > 12 mm above the splice")
    if conf.get("method") != "two traces join":
        dwhy.append("traces do not join (%s)" % conf.get("method"))
    rec["distal_ok"] = len(dwhy) == 0
    rec["distal_why_not"] = dwhy
    rec["whole_va_pass"] = bool(donor and grec["pass_hard"] and rec["distal_ok"])
    rec["gate_fail_reasons"] = why
    rec["review"] = review
    rec["passes_as_v5_donor"] = bool(donor)
    rec["passes_draft_with_additions"] = bool(gdraft["pass_with_additions"])
    rec["status"] = "donor" if donor and not review else "donor_review" if donor else "fail"
    print("%s %s: L %.1f lm %s entry %s/%s n6 %s junc %s v3r %s/%s v3s %s/%s HU %s/%s donor %s distal %s | why %s | review %s"
          % (case, side, L, json.dumps(lm), q.get("foramen_entry_level"), v["entry"].get("Erec_entry"),
             q.get("n_C6_to_C1_passed"), q.get("junction_to_TS_SA_mask_mm"), gi["v3_robust"][0], gi["v3_robust"][1],
             gi["v3_study"][0], gi["v3_study"][1], (v["hu_ts_ct"] or {}).get("cervical_over_cca"), hn["ratio_to_CCA"],
             donor, rec["distal_ok"], why, review), flush=True)

out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(OUTD, "qc.json" if TRD == "tsba" else "qc_%s.json" % TRD), "w"), indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
print("done %.0f s" % out["seconds"])
