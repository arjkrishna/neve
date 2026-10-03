#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_ctwall_assemble.py (= /sp/v5_va/tag_ctwall_assemble.py in the build containers)
"""tag_ctwall_assemble.py -- assemble results/tag_ctwall.json (the tag_ctwall deliverable) and the figures, from
  results/tag_ctwall_measure.json   (va_metrics unchanged; exact ownership; TaG-Net A and VMR 0248 in one harness)
  results/tag_ctwall_landmarks.json (art_extract.ts_qc on total_v3 masks; CT-method landmarks)
  results/tag_ctwall_edge.json      (edge FWHM, image-domain relief, contacts)
  results/tag_ctwall_fine.json      (2 mm union bins with curvature / MISR)
  results/tag_ctwall_noise.json     (noise and CNR)
  derived/tagnet/ct_seg/*/seg_qc.json, seeds_fix.json
Adds: native-threshold set rules (W1-W3 native AND all-piece bars; VMR re-graded at native thresholds, not with
the wrong-signed 0.75 mm factors), W7 by hand, stretch tables, the relief attribution (2 mm bins assigned to
causes by the intervals below, chosen on the QC cross-sections), take-off variants (tag_labels_gross.takeoffs),
bends (vmr_compare.bends). eve-training-fixed.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402
from tag_ctwall_common import LICENCE, QC_DIR, RES, SEG_ROOT, jdump, load_gt  # noqa: E402

NATIVE = {"W1": 0.0966, "W2": 0.0204, "W3": 0.1416}
ALLPIECE = {"W1": 0.1074, "W2": 0.0218, "W3": 0.1519}
W4 = {"sections": 0.1045, "fitted": 0.0627}
W5 = {"0.75-1.5": 0.051, "1.5-3": 0.0451}
W7 = {"0-40": 0.0785, "40-80": 0.2183}
TOPBRAIN = {"gate_valid_use_weighted": {"u03": 0.102, "u05": 0.0215, "rel015": 0.149},
            "pieces_alone_median_of_50": {"u03": 0.121, "u05": 0.026, "rel015": 0.172},
            "v4_donor_stretch_in_situ": {"u03": 0.125, "u05": 0.035, "rel015": 0.176},
            "distal_bins_u03": {"0-40": 0.083, "40-80": 0.230}, "source": "report tables (Wall references)"}
VMR_REPORT = {"edge_FWHM_verifier_mm": [1.39, 1.44], "edge_FWHM_vmr_agent_mm": [1.36, 1.39],
              "whole_u03": [0.045, 0.138], "C6_C1_u03": [0.076, 0.126], "C6_C1_set_W1_W3": [0.101, 0.155],
              "distal_0_40_u03": [0.000, 0.007]}
# relief attribution intervals (s on the ct_seg4 stage-A VMTK centerline), chosen on qc/xsec_*.png
ATTR = {
    "right": [("origin flare (ostium)", 0, 10), ("C6 foramen entry: bone contact", 34, 42),
              ("lumen-bright vessel attached at the C4/C5 foramen (branch stub or abutting vein)", 62, 76),
              ("V4 branch origin (PICA?) + calibre step to r 0.7 mm", 198, 204)],
    "left": [("origin flare (ostium)", 0, 12), ("C6 entry / low V2: bone contact", 36, 51),
             ("lumen-bright vessel attached at the C5 foramen (branch stub or abutting vein)", 70, 80),
             ("C4/C3 foramina: bone contacts", 80, 100),
             ("C2 loop (bend r 2.45 mm) + C2 foramen bone", 106, 126), ("C1 foramen: bone contact", 150, 160),
             ("V3 on the posterior arch of C1 (bends r ~8 mm)", 170, 188),
             ("V4 elliptical lumen at bends (no contact)", 214, 238)]}
CONTACT_KEYS = ("bone", "lumen-bright", "branch", "foramen")


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def set_rules(R, L):
    out = {}
    for k, path in (("W1", ("wall_union", "core", "frac_0.3mm")), ("W2", ("wall_union", "core", "frac_0.5mm")),
                    ("W3", ("wall_union", "core", "frac_rel_0.15"))):
        v = [g(R, *path), g(L, *path)]
        med = float(np.median(v))
        out[k] = {"R": v[0], "L": v[1], "set_median": round(med, 4), "native_min": NATIVE[k],
                  "native": "pass" if med >= NATIVE[k] else "FAIL", "all_piece_min": ALLPIECE[k],
                  "all_piece": "pass" if med >= ALLPIECE[k] else "FAIL"}
    sec = float(np.median([g(R, "wall_sections", "in_out_0.3_mean"), g(L, "wall_sections", "in_out_0.3_mean")]))
    fit = float(np.median([g(R, "wall_native", "fitted_frac_abs_0.3"), g(L, "wall_native", "fitted_frac_abs_0.3")]))
    out["W4_sections"] = {"set_median": round(sec, 4), "min": W4["sections"], "result": "pass" if sec >= W4["sections"] else "FAIL"}
    out["W4_fitted"] = {"set_median": round(fit, 4), "min": W4["fitted"], "result": "pass" if fit >= W4["fitted"] else "FAIL"}
    for b in ("0.75-1.5", "1.5-3"):
        m = float(np.median([g(R, "wall_native", "roughness_bands", b, "rms_mm"), g(L, "wall_native", "roughness_bands", b, "rms_mm")]))
        out["W5_" + b] = {"set_median": round(m, 4), "min": W5[b], "result": "pass" if m >= W5[b] else "FAIL"}
    for side, r in (("R", R), ("L", L)):
        bins = [b["frac_0.3mm"] for b in r["wall_union"]["bins_from_origin"] + r["wall_union"]["bins_from_distal"]]
        out["W6_" + side] = {"min_40mm_bin": round(min(bins), 4), "result": "pass" if min(bins) >= 0.01 else "FAIL"}
        bd = r["wall_union"]["bins_from_distal"]
        out["W7_" + side] = {"0-40": bd[0]["frac_0.3mm"], "40-80": bd[1]["frac_0.3mm"],
                             "result": "pass" if (bd[0]["frac_0.3mm"] >= W7["0-40"] and bd[1]["frac_0.3mm"] >= W7["40-80"]) else "FAIL",
                             "min": W7, "note": "computed by hand (not in the grader): wall_union.bins_from_distal core"}
    return out


def stretch_set(R, L, key):
    u = [g(R, "stretches", "union", key, "core"), g(L, "stretches", "union", key, "core")]
    if None in u:
        return None
    out = {"R": {k: u[0][k] for k in ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15", "area_mm2")},
           "L": {k: u[1][k] for k in ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15", "area_mm2")},
           "range_R": g(R, "stretches", "ranges_mm", key), "range_L": g(L, "stretches", "ranges_mm", key)}
    for k, f in (("W1", "frac_0.3mm"), ("W2", "frac_0.5mm"), ("W3", "frac_rel_0.15")):
        med = float(np.median([u[0][f], u[1][f]]))
        out[k] = {"set": round(med, 4), "native": "pass" if med >= NATIVE[k] else "FAIL",
                  "all_piece": "pass" if med >= ALLPIECE[k] else "FAIL"}
    return out


def attribution(rows, side, L, ranges):
    a = np.array([r["u03"] * r["area"] for r in rows])
    s0 = np.array([r["from"] for r in rows])
    tot = float(a.sum())
    cls = np.full(len(rows), "unattributed (smooth wall / scattered)", dtype=object)
    for nm, lo, hi in ATTR[side]:
        cls[(s0 >= lo) & (s0 < hi)] = nm
    out = {"detail_area_mm2_total": round(tot, 2), "by_cause": {}}
    for c in list(dict.fromkeys(list(cls))):
        m = cls == c
        out["by_cause"][c] = {"share_of_detail_area": round(float(a[m].sum() / max(tot, 1e-9)), 3),
                              "union_0.3_inside": round(float(a[m].sum() / max(sum(r["area"] for r, k in zip(rows, m) if k), 1e-9)), 3)}
    # C6 -> C1 with and without the attributed contact / branch zones
    for lab, (c6, c1) in ranges.items():
        m = (s0 >= c6) & (s0 < c1)
        area = np.array([r["area"] for r in rows])
        cont = np.array([any(k in str(c) for k in CONTACT_KEYS) for c in cls])
        attd = np.array([not str(c).startswith("unattributed") for c in cls])
        out["C6_to_C1_" + lab] = {
            "range": [c6, c1], "u03_all_bins": round(float(a[m].sum() / area[m].sum()), 4),
            "u03_outside_contact_and_branch_zones": round(float(a[m & ~cont].sum() / area[m & ~cont].sum()), 4),
            "u03_outside_all_attributed_zones": round(float(a[m & ~attd].sum() / area[m & ~attd].sum()), 4),
            "share_of_C6_C1_detail_in_contact_and_branch_zones": round(float(a[m & cont].sum() / max(a[m].sum(), 1e-9)), 3),
            "share_of_C6_C1_wall_area_in_contact_and_branch_zones": round(float(area[m & cont].sum() / max(area[m].sum(), 1e-9)), 3),
            "contact_and_branch_zone_length_mm": round(float(2.0 * (m & cont).sum()), 1), "C6_C1_length_mm": round(float(c1 - c6), 1),
            "note": "2 mm bins, 'all' cells (no 2 mm end trim), area-weighted; contact/branch zones = intervals naming "
                    "bone, foramen, lumen-bright vessel or branch (the left C2-loop zone is combined with C2 foramen bone)"}
    return out


def bend_in(P, lo, hi, sigma=3.0):
    Pr, _, s = vm.resample(P, None)
    _, k, _, _ = vm.frenet(Pr, sigma)
    m = (s >= max(lo, vm.END_EXCL_MM)) & (s <= min(hi, s[-1] - vm.END_EXCL_MM))
    i = np.nonzero(m)[0][np.argmax(k[m])]
    return {"min_bend_radius_mm": round(float(1 / k[i]), 2), "at_s": round(float(s[i]), 1), "z": round(float(Pr[i, 2]), 1)}


def main():
    M = json.load(open(os.path.join(RES, "tag_ctwall_measure.json")))
    LM = json.load(open(os.path.join(RES, "tag_ctwall_landmarks.json")))
    E = json.load(open(os.path.join(RES, "tag_ctwall_edge.json")))
    F = json.load(open(os.path.join(RES, "tag_ctwall_fine.json")))
    N = json.load(open(os.path.join(RES, "tag_ctwall_noise.json")))
    from tag_labels_gross import takeoffs
    from vmr_compare import bends
    res = {"agent": "tag_ctwall", "date": "2026-10-01", "licence": LICENCE,
           "source": "TaG-Net SampleData 002 CTA (patient A), 0.405 x 0.405 x 0.4 mm acquisition voxel; GT VA paths "
                     "from the tag_labels agent; TotalSegmentator total_v3 masks from tag_totalseg",
           "method": {"segmentation": "tag_ct_seg.py = vmr_ct_seg3.py with VMR_OPEN_BRIGHT=1 (ct_seg4), algorithm and "
                                      "parameters unchanged; loader adapter tag_ctwall_common.py; VMTK vmr_vmtk.py unchanged",
                      "metrics": "va_metrics.py unchanged; exact ownership by run-time KTree64 (k 24 -> 64); "
                                 "vmr_measure.union_segments for stretches; va_standard.grade_set unchanged",
                      "thresholds": {"native": NATIVE, "all_piece": ALLPIECE, "W4": W4, "W5": W5, "W7": W7},
                      "voxel_for_V5": [0.405, 0.405, 0.4]},
           "references": {"TopBrain": TOPBRAIN, "VMR_report": VMR_REPORT}}
    # ---------------- segmentation QC
    sq = {}
    for tag in ("ct_seg4", "ct_seg3", "ct_seg4_se082", "ct_seg4_r117"):
        q = json.load(open(os.path.join(SEG_ROOT, tag, "seg_qc.json")))
        sf = os.path.join(SEG_ROOT, tag, "seeds_fix.json")
        sq[tag] = {"vessels": {v: {k: q["vessels"][v].get(k) for k in (
            "pass1_shift_median_mm", "pass2_shift_median_mm", "r_eq_rays_median", "r_contour_median", "I_core_median",
            "I_bg_median", "T_loc_median", "T_loc_p10_p90", "ray_rule_frac", "T_hi_rays_frac", "voxels",
            "r_volume_equiv_median", "frac_boundary_at_cap", "frac_vox_touching_bone700")} for v in ("rva", "lva", "ba")},
            "open_bright": q.get("open_bright"), "structuring_element": q.get("structuring_element"),
            "tag_adapter": {k: v for k, v in (q.get("tag_adapter") or {}).items() if k != "licence"},
            "surfaces_open_edges": {v: {k: q["surfaces"][v][k]["open_edges"] for k in ("stageA", "iso")} for v in q["surfaces"]},
            "seeds_fix": json.load(open(sf)) if os.path.exists(sf) else None, "seconds": q.get("seconds")}
    res["segmentation_qc"] = sq
    # ---------------- per representation: summary, set rules (native), stretches
    reps = {}
    for src, a in M["full"].items():
        for rep, b in a.items():
            for km, c in b.items():
                if "right" not in c or "left" not in c:
                    continue
                R, L = c["right"], c["left"]
                key = "%s/%s/%s" % (src, rep, km)
                summ = {}
                for side, r in (("right", R), ("left", L)):
                    gr, ca, wu, ws, wn = r["gross"], r["calibre"], r["wall_union"], r["wall_sections"], r["wall_native"]
                    summ[side] = {"length_mm": gr["length_mm"], "DM": gr["DM"], "kappa_p50_s3": gr["shape@3.0"]["kappa_p50"],
                                  "min_bend_radius_s3_mm": gr["shape@3.0"]["min_bend_radius_mm"],
                                  "loops_360_60mm": gr["shape@3.0"]["n_loops_360_60mm"], "uturns": gr["shape@3.0"]["n_uturns_150_30mm"],
                                  "takeoff_va_metrics": gr.get("takeoff"), "segments": gr.get("segments_landmarks"),
                                  "r_misr_median": ca["r_median_mm"], "r_p10_p90": [ca["r_p10_mm"], ca["r_p90_mm"]],
                                  "q_min_W10": ca["q_min_W10"], "union_core": wu["core"], "union_genus": wu["genus"],
                                  "union_bins_from_origin_0.3": [x["frac_0.3mm"] for x in wu["bins_from_origin"]],
                                  "union_bins_from_distal_0.3": [x["frac_0.3mm"] for x in wu["bins_from_distal"]],
                                  "sections_in_out_0.3_mean": ws.get("in_out_0.3_mean"), "sections_circ_median": ws.get("circ_median"),
                                  "centre_offset_mm": ws.get("centre_offset_mm_median"), "sections_missed": ws.get("n_missed"),
                                  "outside_frac": r["centerline_qc"]["outside_frac"], "misr_over_declared": r["centerline_qc"]["misr_over_declared"],
                                  "native_fitted_abs_0.3": wn.get("fitted_frac_abs_0.3"),
                                  "roughness_bands_rms_mm": {k: v["rms_mm"] for k, v in wn["roughness_bands"].items()},
                                  "lowpass_fitted_abs_0.3": {k: v["fitted_frac_abs_0.3"] for k, v in wn["lowpass"].items()}}
                gate = M["grades"].get(key)
                gk = ("V1_medial_centerline", "V2_centerline_inside_wall", "V3_closed_sections", "V4_union_topology")
                reps[key] = {"summary": summ, "set_rules_native": set_rules(R, L),
                             "gates_V1_V4_ok": {s: (None if not gate else not any(str(gate["per_va"][j].get(k, {}).get("result", "")).startswith("FAIL") for k in gk))
                                                for j, s in enumerate(("right", "left"))},
                             "set_rules_note": "native set rules use BOTH VAs; where gates_V1_V4_ok is false for a side the grader would drop it",
                             "grader_per_va": [{k: (v.get("value"), v.get("result")) for k, v in pv.items()} for pv in gate["per_va"]] if gate else None,
                             "grader_set_rules_as_implemented": gate["set_rules"] if gate else None,
                             "stretches": {k: stretch_set(R, L, k) for k in R["stretches"]["ranges_mm"] if k in L["stretches"]["ranges_mm"]},
                             "bins10": {"right": R["bins10"], "left": L["bins10"]}}
    res["representations"] = reps
    res["pieces_alone"] = {}
    for src, a in M["pieces"].items():
        for rep, b in a.items():
            for label, c in b.items():
                gk = "piece/%s/%s/%s" % (src, rep, label)
                row = {s: {"range": c[s]["provenance"].get("piece_mm_on_full_centerline"), "union_core": c[s]["wall_union"]["core"],
                           "genus": c[s]["wall_union"]["genus"], "sections_in_out_0.3_mean": c[s]["wall_sections"].get("in_out_0.3_mean")}
                       for s in c}
                if "right" in c and "left" in c:
                    row["set_rules_native"] = {k: v for k, v in set_rules(c["right"], c["left"]).items() if k[:2] in ("W1", "W2", "W3", "W4", "W5")}
                    row["grader_set_rules_as_implemented"] = M["grades"][gk]["set_rules"]
                res["pieces_alone"][gk] = row
    # ---------------- landmarks
    lmk = {}
    for side in ("right", "left"):
        rec = LM["tags"]["ct_seg4"]["stageA"][side]
        lmk[side] = {"ts_total_v3_used": rec["total_v3_1p5"]["landmarks_used"],
                     "ts_qc_raw_landmarks": rec["total_v3_1p5"]["ts_qc_landmarks_mm"],
                     "ts_qc_entry_level": rec["total_v3_1p5"]["ts_qc_entry_level"],
                     "enclosed_runs_per_level_mm": rec["total_v3_1p5"]["enclosed_runs_per_level_mm"],
                     "landmark_z_mm": rec["total_v3_1p5"]["landmark_z_mm"], "v3loop": rec["total_v3_1p5"]["v3loop"],
                     "origin_to_TS_SA_mask_mm": rec["total_v3_1p5"]["origin_to_TS_SA_mask_mm"],
                     "ts_total_used": rec["total_1p5"]["landmarks_used"], "ct_method": rec["ct_method"]}
        for nm, lmx in (("ts", lmk[side]["ts_total_v3_used"]), ("ct", lmk[side]["ct_method"])):
            e = lmx["end"]
            lmk[side]["V1_V4_mm_" + nm] = [round(lmx["C6"], 1), round(lmx["C2"] - lmx["C6"], 1),
                                           round(lmx["dura"] - lmx["C2"], 1), round(e - lmx["dura"], 1)]
    res["landmarks"] = lmk
    # ---------------- gross: take-off variants and bends on the primary centerline
    gross = {}
    for side, v in (("right", "rva"), ("left", "lva")):
        d = json.load(open(os.path.join(SEG_ROOT, "ct_seg4", "%s_stageA_vmtk.json" % v)))
        P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
        sca = np.asarray(load_gt(side)["subclavian"]["points"], float)
        tk, _ = takeoffs(P, sca)
        st = json.load(open(os.path.join(SEG_ROOT, "ct_seg4", "stations_rays.json")))
        sca_ct = np.asarray(st["rsub" if side == "right" else "lsub"]["P"])
        if np.linalg.norm(sca_ct[0] - sca[0]) > np.linalg.norm(sca_ct[-1] - sca[0]):
            sca_ct = sca_ct[::-1]
        tk_ct, _ = takeoffs(P, sca_ct)
        lmx = lmk[side]["ts_total_v3_used"]
        lmc = lmk[side]["ct_method"]
        gross[side] = {"takeoff_study_definition_gt_sca": tk, "takeoff_with_ct_recentred_sca": tk_ct,
                       "bends_s3": bends(P, R, 3.0), "bends_s1.5": bends(P, R, 1.5),
                       "C2_region_min_bend_s3": dict(bend_in(P, lmc["C2"] - 30.0, lmc["C1"]),
                                                     window="CT-method C2 - 30 mm .. CT-method C1 (the C2 loop lies "
                                                            "inside / at the exit of the C2 foramen)"),
                       "V3_C1_to_dura_min_bend_s3": bend_in(P, lmx["C1"], lmx["dura"]),
                       "summary": reps["TaG-A/ct_seg4_stageA/k64"]["summary"][side]}
    res["gross"] = gross
    # ---------------- edge, image relief, contacts, noise
    ed = {}
    for name in ("TaG-A", "VMR"):
        ed[name] = {}
        for side in ("right", "left"):
            r = E[name]["sides"][side]
            ed[name][side] = {"FWHM_vmr_agent_method_default": r["edge_vmr_method_default"],
                              "FWHM_vmr_agent_method_dense": r["edge_vmr_method_dense"],
                              "FWHM_verifier_method": r["edge_verifier_method"],
                              "image_edge_dev": r["image_edge_dev"], "contacts_total": r["contacts_total"],
                              "per_10mm": r["per_10mm"]}
    res["edge"] = ed
    res["erf_fit_bias_on_blurred_discs"] = E["erf_fit_bias_on_blurred_discs"]
    res["noise"] = {k: N[k] for k in ("TaG-A", "VMR")}
    # ---------------- relief attribution (TaG-Net A, primary)
    att = {}
    for side in ("right", "left"):
        rows = F["TaG-A"][side]["bins2"]
        ranges = {"ts": (lmk[side]["ts_total_v3_used"]["C6"], lmk[side]["ts_total_v3_used"]["C1"]),
                  "ct": (lmk[side]["ct_method"]["C6"], lmk[side]["ct_method"]["C1"])}
        att[side] = attribution(rows, side, F["TaG-A"][side]["length_mm"], ranges)
        att[side]["intervals"] = ATTR[side]
    res["relief_attribution"] = att
    res["fine_bins_2mm"] = F
    jdump(res, os.path.join(RES, "tag_ctwall.json"))
    print("wrote tag_ctwall.json")
    # ---------------- compact printout for the notes
    p = reps["TaG-A/ct_seg4_stageA/k64"]
    print("SET", json.dumps(p["set_rules_native"]))
    for k in ("C6_to_C1", "ct:C6_to_C1", "origin_to_C1", "ct:origin_to_C1", "origin_to_C1_minus_seam", "ct:origin_to_C1_minus_seam",
              "seam_zone_C1m25_to_C1", "ct:seam_zone_C1m25_to_C1", "V1", "V2", "V3_C2_to_C1", "V3_C1_to_dura", "V4",
              "ct:V1", "ct:V2", "ct:V3_C2_to_C1", "ct:V3_C1_to_dura", "ct:V4", "distal_0_40", "distal_40_80"):
        print("STRETCH", k, json.dumps(p["stretches"].get(k)))
    v = reps["VMR/ct_seg4_stageA/k64"]
    print("VMRSET", json.dumps(v["set_rules_native"]))
    for k in ("C6_to_C1", "verifierC6_37:C6_to_C1", "s40:C6_to_C1", "origin_to_C1_minus_seam", "seam_zone_C1m25_to_C1"):
        print("VMRSTRETCH", k, json.dumps(v["stretches"].get(k)))
    print("ATTR", json.dumps(att))
    print("GROSS", json.dumps({s: {k: gross[s][k] for k in ("takeoff_study_definition_gt_sca", "takeoff_with_ct_recentred_sca",
                                                             "C2_region_min_bend_s3", "V3_C1_to_dura_min_bend_s3")} for s in gross}))
    print("BENDS", json.dumps({s: gross[s]["bends_s3"] for s in gross}))
    print("LMK", json.dumps({s: {k: lmk[s][k] for k in ("V1_V4_mm_ts", "V1_V4_mm_ct", "ts_qc_entry_level", "landmark_z_mm")} for s in lmk}))
    for k in ("TaG-A/ct_seg4_iso/k64", "TaG-A/ct_seg3_stageA/k64", "TaG-A/ct_seg3_iso/k64", "TaG-A/ct_seg4_se082_stageA/k64",
              "TaG-A/ct_seg4_r117_stageA/k64", "VMR/ct_seg4_iso/k64", "VMR/ct_seg3_stageA/k64", "TaG-A/ct_seg4_stageA/k24",
              "VMR/ct_seg4_stageA/k24"):
        q = reps[k]
        print("REP", k, json.dumps({kk: (vv.get("set_median"), vv.get("native", vv.get("result"))) for kk, vv in q["set_rules_native"].items() if kk[:2] in ("W1", "W2", "W3")}),
              "genus", [q["summary"][s]["union_genus"] for s in ("right", "left")],
              "C6C1 ct", None if not q["stretches"].get("ct:C6_to_C1") else (q["stretches"]["ct:C6_to_C1"]["W1"]["set"], q["stretches"]["ct:C6_to_C1"]["W3"]["set"]),
              "C6C1 ts", None if not q["stretches"].get("C6_to_C1") else (q["stretches"]["C6_to_C1"]["W1"]["set"], q["stretches"]["C6_to_C1"]["W3"]["set"]))
    figures(res)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    reps = res["representations"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharey=True, facecolor="#fcfcfb")
    for ax, side in zip(axs, ("right", "left")):
        ax.set_facecolor("#fcfcfb")
        tg = reps["TaG-A/ct_seg4_stageA/k64"]["bins10"][side]
        vr = reps["VMR/ct_seg4_stageA/k64"]["bins10"][side]
        lm = res["landmarks"][side]["ts_total_v3_used"]
        lc = res["landmarks"][side]["ct_method"]
        L = lm["end"]
        for nm, lo, hi in ATTR[side]:
            ax.axvspan(lo, hi, color="#e8e7e2", zorder=0)
        for k in ("C6", "C2", "C1", "dura"):
            ax.axvline(lm[k], color="#c3c2b7", lw=1, zorder=1)
            ax.text(lm[k] + 0.8, 0.735, k + " (TS)", color="#52514e", fontsize=8.5)
            ax.axvline(lc[k], color="#c3c2b7", lw=1, ls=":", zorder=1)
        ax.axhline(0.102, color="#8a8984", lw=1, ls="--", zorder=1)
        ax.text(L - 1, 0.112, "TopBrain VA union, use-weighted median 0.102", color="#52514e", fontsize=9, ha="right")
        ax.hlines([0.083, 0.230], [L - 40, L - 80], [L, L - 40], color="#52514e", lw=1.5, ls=":", zorder=1)
        for rows, col, lab in ((tg, C1, "TaG-Net A, CT segmentation (ct_seg4, stage-A)"),
                               (vr, "#8a8984", "VMR 0248, CT segmentation (ct_seg4), same harness")):
            x = [0.5 * (r["from_mm"] + r["to_mm"]) for r in rows]
            y = [r["frac_0.3mm"] for r in rows]
            ax.plot(x, y, "-o", color=col, lw=2 if col == C1 else 1.2, ms=4.5 if col == C1 else 3, label=lab, zorder=3)
        ax.set_ylim(0, 0.79)
        ax.set_xlim(0, max(L, 225))
        ax.set_ylabel("union wall > 0.3 mm off the tube\n(area fraction per 10 mm)")
        ax.set_title("TaG-Net A %s vertebral artery (grey bands: relief attributed on the cross-sections; solid lines TS "
                     "landmarks, dotted CT-method)" % side, loc="left", color="#0b0b0b", fontsize=10)
        ax.grid(axis="y", color="#e8e7e2", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[1].set_xlabel("arclength from the VA ostium (mm, each VA's own CT centerline)")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.45, 0.93), frameon=False, fontsize=9)
    plt.tight_layout()
    os.makedirs(QC_DIR, exist_ok=True)
    plt.savefig(os.path.join(QC_DIR, "detail_along_va.png"), dpi=100, facecolor="#fcfcfb")
    # FWHM by ray inclination
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="#fcfcfb")
    for ax, meth, title in ((axs[0], "FWHM_vmr_agent_method_dense", "vmr agent's erf fit (segmentation rays)"),
                            (axs[1], "FWHM_verifier_method", "verifier's erf fit (VMTK centerline)")):
        ax.set_facecolor("#fcfcfb")
        bins = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.01"]
        xs = np.arange(len(bins))
        for j, (name, side, col, ls) in enumerate((("TaG-A", "right", C1, "-"), ("TaG-A", "left", C1, "--"),
                                                    ("VMR", "right", C2, "-"), ("VMR", "left", C2, "--"))):
            y = [res["edge"][name][side][meth]["by_ray_inclination_absdz"][b].get("FWHM_mm") for b in bins]
            ax.plot(xs, y, ls, marker="o", color=col, lw=2, ms=6, label="%s %s" % ("TaG-Net A" if name == "TaG-A" else "VMR 0248", side))
        ax.set_xticks(xs)
        ax.set_xticklabels(["0-0.25\n(axial-plane rays)", "0.25-0.5", "0.5-0.75", "0.75-1\n(along z)"])
        ax.set_xlabel("|z component| of the ray direction")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e8e7e2", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("lumen edge FWHM (mm)")
    axs[0].set_ylim(0.9, 1.6)
    axs[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("VA lumen edge width by ray direction: the 0.4 mm slices sharpen only the z direction", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(QC_DIR, "edge_fwhm_by_direction.png"), dpi=100, facecolor="#fcfcfb")
    print("figures written")


if __name__ == "__main__":
    main()
