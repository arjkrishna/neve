#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_collect.py (= /sp/v5_va/scaleup_song_collect.py in the build containers)
# scaleup_song_collect.py (scaleup_lane_song, 2026-10-02) = pilot_proc_song_collect.py with TWO changes: RES = results/scaleup_song.json
#   (pilot: results/pilot_proc_song.json) and the timing log _logs_scaleup/timing.jsonl (pilot: _logs_proc/). Same record fields.
# SONG COPY (pilot_proc_song, re-copied 2026-10-01 21:55): pilot_proc_ia172_collect.py with the song paths, the Song meta keys and W7 by hand (marked 'song').
# 10-02 01:25 (2nd instance): + ia172's later relief_vs_contacts block (verbatim) + wall.protrusion (song protrusion.json) + wall.label_truncation.
"""pilot_proc_song_collect.py <case> [<case> ...] -- gather one 172-series pilot case's outputs into ONE record per
case and side, and merge them into /sp/v5_va/results/pilot_proc_song.json (re-run safe: a case's records are replaced,
the other cases' kept). eve-training-fixed (json + numpy only).

Per side: trace / QC verdict (recalibrated gate rules, draft verdict, review flags, distal grade), landmarks, entry level,
V3 loop, phase, bridges, gross shape (qc/gross.json), and -- for sides with a ct_seg4 wall -- validity gates V1-V6,
union detail > 0.3 / > 0.5 mm / > 0.15 r for the whole VA and for every stretch on three landmark sets ("ts" = ts_qc
C2/C1/dura + the C6 level's own run, tag_ctwall's TS set; "ct" = VMR's CT bony-ring method; "rec" = tag_calibration's
Erec C6, ts_qc C2, C1f, ts_qc dura), pieces alone (C6 -> C1, origin -> C1), plain-tube 10 mm bins (< 0.01), where the
relief sits (top bins), edge FWHM, image relief, noise and contacts (edge.json).
Native thresholds only: W1 0.0966, W2 0.0204, W3 0.1416 (set medians; quoted per VA as the bar), acquisition voxel.
"""
import json
import os
import sys
import time

import numpy as np

RES = "/sp/v5_va/results/scaleup_song.json"
ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/song"
SRCR = "/repo/carotid_data/va_sources/thin_slice/song"
BARS = {"W1": 0.0966, "W2": 0.0204, "W3": 0.1416}
STRETCHES = ("V1", "V2", "V3_C2_to_C1", "V3_C1_to_dura", "V4", "C6_to_C1", "origin_to_C1", "origin_to_C1_minus_seam",
             "seam_zone_C1m25_to_C1", "C6_to_C1_minus_seam", "distal_0_40", "distal_40_80")


def L(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def rr(x, n=4):
    return None if x is None else round(float(x), n)


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def u3(u):
    if not isinstance(u, dict):
        return None
    c = u.get("core", u)
    return {"0.3mm": rr(c.get("frac_0.3mm")), "0.5mm": rr(c.get("frac_0.5mm")), "rel0.15": rr(c.get("frac_rel_0.15")),
            "area_mm2": rr(c.get("area_mm2"), 1)}


def wall_record(case, sd, M, LM, E):
    r = g(M, "full", sd)
    if not r:
        return None
    out = {"whole": u3(r["wall_union"]), "genus": g(r, "wall_union", "genus"),
           "centre_offset_mm_median": rr(g(r, "wall_sections", "centre_offset_mm_median")),
           "sections_in_out_0.3_mean": rr(g(r, "wall_sections", "in_out_0.3_mean")),
           "fitted_abs_0.3": rr(g(r, "wall_native", "fitted_frac_abs_0.3")),
           "r_misr_median_mm": rr(g(r, "calibre", "r_median_mm"), 3), "q_min_W10": rr(g(r, "calibre", "q_min_W10"), 3),
           "length_mm": rr(g(r, "gross", "length_mm"), 1), "DM": rr(g(r, "gross", "DM"), 3),
           "min_bend_radius_s3_mm": g(r, "gross", "shape@3.0", "min_bend_radius_mm"),
           "grade": g(M, "grades", sd),
           "fitted_abs_0.3_lowpass_0.75_1.5": [rr(g(r, "wall_native", "fitted_frac_abs_0.3")),
                                               rr(g(r, "wall_native", "lowpass", "sigma_0.75", "fitted_frac_abs_0.3")),
                                               rr(g(r, "wall_native", "lowpass", "sigma_1.50", "fitted_frac_abs_0.3"))],
           "roughness_rms_mm": {k: (v or {}).get("rms_mm") for k, v in (g(r, "wall_native", "roughness_bands") or {}).items()},
           "circ_mean": rr(g(r, "wall_sections", "circ_mean"), 3)}
    gr = out["grade"] or {}
    out["gates_V1_V6"] = {k: (gr.get(k) or {}).get("result") for k in ("V1_medial_centerline", "V2_centerline_inside_wall",
                                                                         "V3_closed_sections", "V4_union_topology",
                                                                         "V5_resolution", "V6_radius_protocol")}
    out["gates_pass"] = not any(str(v).startswith("FAIL") for v in out["gates_V1_V6"].values() if v is not None)
    w = out["whole"] or {}
    out["whole_vs_bars"] = {"W1": w.get("0.3mm") is not None and w["0.3mm"] >= BARS["W1"],
                            "W2": w.get("0.5mm") is not None and w["0.5mm"] >= BARS["W2"],
                            "W3": w.get("rel0.15") is not None and w["rel0.15"] >= BARS["W3"]}
    st = g(r, "stretches", "union") or {}
    rng = g(r, "stretches", "ranges_mm") or {}
    out["stretches"] = {}
    for pre in ("", "ct:", "rec:"):
        nm = {"": "ts", "ct:": "ct", "rec:": "rec"}[pre]
        out["stretches"][nm] = {k: dict(u3(st.get(pre + k)) or {}, range_mm=rng.get(pre + k))
                                for k in STRETCHES if (pre + k) in st or (pre == "" and k in st and k.startswith("distal"))}
    # song addition: W7 by hand (tag_ctwall_assemble's rule: union > 0.3 mm over the distal 0-40 / 40-80 mm >= 0.0785 /
    # 0.2183; not in the grader). Only meaningful for a whole-VA donor; strategy B discards the donor above the V3 splice.
    d04 = g(out, "stretches", "ts", "distal_0_40", "0.3mm")
    d48 = g(out, "stretches", "ts", "distal_40_80", "0.3mm")
    out["W7_by_hand"] = {"0-40": d04, "40-80": d48, "min": {"0-40": 0.0785, "40-80": 0.2183},
                         "result": None if d04 is None or d48 is None else (
                             "pass" if d04 >= 0.0785 and d48 >= 0.2183 else "FAIL"),
                         "note": "whole-VA donor only; a side cut at the dura has no true distal end (then this is the "
                                 "last 0-80 mm below the cut)"}
    pcs = {}
    for lab, d in (M.get("pieces") or {}).items():
        if sd in d:
            pr = d[sd]
            pcs[lab] = {"union": u3(pr["wall_union"]), "genus": g(pr, "wall_union", "genus"),
                        "range_mm": g(pr, "provenance", "extra", "piece_mm_on_full_centerline")
                        or g(pr, "provenance", "piece_mm_on_full_centerline")}
    out["pieces"] = pcs
    b = r.get("bins10") or []
    if b:
        f = np.array([x["frac_0.3mm"] for x in b], float)
        out["bins10"] = [[x["from_mm"], x["to_mm"], rr(x["frac_0.3mm"]), rr(x["frac_rel_0.15"])] for x in b]
        out["plain_tube_bin_fraction_lt_0.01"] = rr(np.mean(f < 0.01), 3)
        out["n_bins10"] = int(len(f))
        top = np.argsort(-f)[:4]
        out["top_relief_bins"] = [{"from_mm": b[i]["from_mm"], "to_mm": b[i]["to_mm"], "frac_0.3mm": rr(f[i]),
                                   "share_of_detail_area": None} for i in top]
        A = np.array([x.get("area_mm2") or 0.0 for x in b], float)
        da = f * A
        if da.sum() > 0:
            for t, i in zip(out["top_relief_bins"], top):
                t["share_of_detail_area"] = rr(da[i] / da.sum(), 3)
    lmr = g(LM, "sides", sd) or {}
    out["landmarks"] = {"ts": lmr.get("landmarks_used"), "ct": lmr.get("landmarks_ct"), "rec": lmr.get("landmarks_rec"),
                        "entry_ts_qc": lmr.get("ts_qc_entry_level"), "entry_rec": lmr.get("entry_level_rec"),
                        "entry_ct": g(lmr, "ct_method", "entry_level"), "v3loop_robust": lmr.get("v3loop_robust")}
    side = "R" if sd == "right" else "L"
    e = g(E, "sides", side)
    if e:
        out["edge"] = {"FWHM_verifier_mm": g(e, "edge_verifier_method", "all", "FWHM_mm"),
                       "FWHM_verifier_by_stretch": {k: (v or {}).get("FWHM_mm") for k, v in
                                                    (g(e, "edge_verifier_method", "by_stretch") or {}).items()},
                       "FWHM_verifier_axial_rays": g(e, "edge_verifier_method", "by_ray_inclination_absdz", "0.00-0.25", "FWHM_mm"),
                       "FWHM_verifier_z_rays": g(e, "edge_verifier_method", "by_ray_inclination_absdz", "0.75-1.01", "FWHM_mm"),
                       "FWHM_verifier_like_for_like": g(e, "edge_verifier_method", "r1.0-1.6_and_absdz<0.5", "FWHM_mm"),
                       "FWHM_vmr_method_default": g(e, "edge_vmr_method_default", "FWHM_mm"),
                       "image_edge_dev_0.3": {k: (v or {}).get("frac_rays_dev_gt_0.3mm") for k, v in (e.get("image_edge_dev") or {}).items()},
                       "noise": e.get("noise"), "contacts_total": e.get("contacts_total")}
        # song 10-02 01:25: ported from pilot_proc_ia172_collect.py (added there after the 21:55 copy), unchanged:
        # where the relief sits: 10 mm bins with >= 5 % of the ct_seg4 rays meeting a brighter neighbour (kind 3, upper
        # threshold, or kind 1, dip before bright: bone in the foramina, enhanced veins) vs the rest (same s bins: both
        # start at the ct_seg4 centreline origin)
        pb = e.get("per_10mm") or []
        if b and pb:
            cmap_ = {round(x["from_mm"]): (x.get("frac_rays_upper_threshold_kind3") or 0) + (x.get("frac_rays_dip_before_bright_kind1") or 0)
                     for x in pb}
            con = np.array([cmap_.get(round(x["from_mm"]), 0.0) >= 0.05 for x in b])
            A_ = np.array([x.get("area_mm2") or 0.0 for x in b], float)
            f_ = np.array([x["frac_0.3mm"] for x in b], float)
            da_ = f_ * A_
            out["relief_vs_contacts"] = {
                "contact_bins": int(con.sum()), "n_bins": int(len(con)),
                "share_of_detail_area_in_contact_bins": rr(da_[con].sum() / da_.sum(), 3) if da_.sum() > 0 else None,
                "union_0.3_contact_bins": rr(da_[con].sum() / A_[con].sum(), 4) if A_[con].sum() > 0 else None,
                "union_0.3_other_bins": rr(da_[~con].sum() / A_[~con].sum(), 4) if A_[~con].sum() > 0 else None,
                "rule": ">= 5 % of the bin's rays kind 3 (upper threshold) or kind 1 (dip before bright)"}
            # song addition (01:30): the LOWER VA (what v5 grafts: origin -> V3 splice ~ C1) -- 10 mm bins whose midpoint
            # lies below the rec C1 landmark (else ct, else ts), all bins vs non-contact bins only (contamination excluded)
            rv = out["relief_vs_contacts"]
            rv["contact_bins_from_mm"] = [b[i]["from_mm"] for i in range(len(b)) if con[i]]
            c1 = None
            for lset in ("rec", "ct", "ts"):
                c1 = g(out, "landmarks", lset, "C1")
                if c1 is not None:
                    rv["lower_VA_C1_landmark"] = lset
                    break
            if c1 is not None:
                mid = np.array([0.5 * (x["from_mm"] + x["to_mm"]) for x in b], float)
                lo_ = mid < float(c1)
                rl_ = np.array([x.get("frac_rel_0.15") or 0.0 for x in b], float)
                for nm_, msk in (("all_bins", lo_), ("non_contact_bins", lo_ & ~con)):
                    a_ = A_[msk].sum()
                    rv["lower_VA_" + nm_] = {"n_bins": int(msk.sum()), "union_0.3": rr(da_[msk].sum() / a_, 4) if a_ > 0 else None,
                                             "union_rel0.15": rr((rl_ * A_)[msk].sum() / a_, 4) if a_ > 0 else None}
                rv["lower_VA_C1_s_mm"] = rr(float(c1), 1)
    return out


def collect(case):
    D = os.path.join(ROOT, case)
    meta = L(os.path.join(SRCR, case, "meta.json")) or {}
    qc = L(os.path.join(D, "qc/qc.json"))
    gross = L(os.path.join(D, "qc/gross.json"))
    W = os.path.join(D, "ct_seg/ct_seg4")
    M, LM, E = L(os.path.join(W, "measure.json")), L(os.path.join(W, "landmarks.json")), L(os.path.join(W, "edge.json"))
    seg_qc = L(os.path.join(W, "seg_qc.json"))
    Mnf = L(os.path.join(D, "ct_seg/ct_seg4_nofix/measure.json"))     # plain ct_seg4 (no diff 8), kept where fix 8 acted
    wd = L(os.path.join(D, "arterial/runs/native/watchdog.json"))
    args = L(os.path.join(D, "arterial/args.json")) or {}
    rep = L(os.path.join(D, "arterial/runs/native/run_report.json")) or {}
    times = []
    tj = os.path.join(ROOT, "_logs_scaleup/timing.jsonl")
    if os.path.isfile(tj):
        for line in open(tj):
            try:
                t = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if t.get("case") == case and not str(t.get("stage", "")).startswith("stage_"):
                times.append({k: t.get(k) for k in ("stage", "seconds", "rc", "end")})
    crec = {"case": case, "sex": meta.get("sex"), "age": meta.get("age"), "status": meta.get("group"),   # song meta keys
            "acquisition_voxel_mm": g(meta, "image", "acquisition_voxel_mm"), "grid_spacing_mm": g(meta, "image", "spacing_mm"),
            "scanner": g(meta, "scanner", "family"),
            "arterial": {"args_mode": args.get("mode"), "run_args": args.get("run_args"), "tool_ok": g(args, "tool", "ok"),
                         "tool_why": g(args, "tool", "why"), "retry_args": args.get("retry_args"),
                         "segmentation": os.path.isfile(os.path.join(D, "arterial/runs/native/extracranial_vessels/segmentation.nii.gz")),
                         "labelled_centerlines": os.path.isfile(os.path.join(D, "arterial/runs/native/labelled_centerlines.json")),
                         "errors": rep.get("errors"), "times_s": rep.get("times_s"), "peak_ram_gb": rep.get("peak_ram_gb_sampled"),
                         "attempt1_failed_dir": os.path.isdir(os.path.join(D, "arterial/runs/native_attempt1_failed")),
                         "centreline_stage_stopped": wd},
            "ct_seg_seeds_fix": g(seg_qc, "seeds_fix"),
            "ct_seg_open_bright": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "flagged_s_ranges_mm"}
                                   for k, v in (g(seg_qc, "open_bright") or {}).items()},
            "qc_present": qc is not None, "cca_median_HU_native": g(qc, "cca_median_HU_native"),
            "contralateral_bcv_HU_native": g(qc, "contralateral_bcv_median_HU_native"),
            "confluence": g(qc, "confluence", "method"), "basilar_on_mask": g(qc, "extract", "basilar_on_mask"),
            "L_over_R_calibre": g(gross, "L_over_R_calibre"), "dominance": g(gross, "dominance"),
            "ct_seg_present": seg_qc is not None, "timing": times}
    recs = []
    sel = L(os.path.join(D, "qc/select.json")) or {}
    qts = L(os.path.join(D, "qc/qc_ts.json"))
    for side, sd in (("R", "right"), ("L", "left")):
        # song (23:38): the trace set selected per side (qc/select.json); tsba = protocol trace, ts = unchanged tool
        ext = g(sel, "sides", side, "ext") or "tsba"
        q = (g(qc, "sides", side) if ext == "tsba" else g(qts, "sides", side)) or {}
        qa = g(qc, "sides", side) or {}
        gr = g(gross, "sides", side) or {}
        R = g(q, "gate_recalibrated", "rules") or {}
        rec = {"case": case, "side": side, "trace_set": ext, "route_rescue": ext != "tsba",
               "tsba_protocol": {"status": qa.get("status"), "passes_as_v5_donor": qa.get("passes_as_v5_donor"),
                                 "whole_va_pass": qa.get("whole_va_pass"), "gate_fail_reasons": qa.get("gate_fail_reasons"),
                                 "distal_why_not": qa.get("distal_why_not")},
               "ts_unchanged_tool": {k: g(qts, "sides", side, k) for k in ("status", "passes_as_v5_donor", "whole_va_pass",
                                                                          "gate_fail_reasons", "distal_why_not")} if qts else None,
               "trace_ok": q.get("trace_ok"), "status": q.get("status"),
               "passes_as_v5_donor": q.get("passes_as_v5_donor"), "gate_fail_reasons": q.get("gate_fail_reasons"),
               "review": q.get("review"), "distal_ok": q.get("distal_ok"), "distal_why_not": q.get("distal_why_not"),
               "whole_va_pass": q.get("whole_va_pass"), "passes_draft_with_additions": q.get("passes_draft_with_additions"),
               "recalibrated_rules": R, "recalibrated_failed": g(q, "gate_recalibrated", "failed"),
               "draft_failed": g(q, "gate_draft", "failed"),
               "length_trace_mm": q.get("length_trace_mm"), "landmarks_trace_mm": q.get("landmarks_used_mm"),
               "entry_Erec": q.get("entry_level_Erec"), "entry_ts_qc": q.get("entry_level_ts_qc"),
               "n_C6_to_C1": q.get("n_C6_to_C1_passed"), "levels": q.get("foramen_levels_passed"),
               "junction_to_TS_SA_mm": q.get("junction_to_TS_SA_mm"),
               "v3_robust": [g(q, "v3loop_robust", "posterior_excursion_mm"), g(q, "v3loop_robust", "d_C1_posterior_half_mm")],
               "v3_study": [g(q, "v3loop_study", "posterior_excursion_mm"), g(q, "v3loop_study", "d_C1_posterior_half_mm")],
               "v2_band": q.get("v2_band"), "hu_ratio_1p5": g(q, "hu_ts_1p5_ct", "cervical_over_cca"),
               "hu_native_origin_to_C1": q.get("trace_HU_native_origin_to_C1"), "bridges": q.get("bridges"),
               "labelled_within_2mm": q.get("labelled_within_2mm_of_trace"), "vmtk": q.get("vmtk"),
               "unchanged_tool_trace": q.get("unchanged_tool_trace"),
               "gross": {k: gr.get(k) for k in ("extent", "length_mm", "DM", "min_bend_radius_s3_mm", "loops_360_60mm",
                                                "takeoff_10mm_fwd5_deg", "subclavian_elevation_fwd5_deg", "V1_V4_mm",
                                                "entry_level_Erec", "r_corrected_x1.10_mm", "sca_source")} if gr else None,
               "gross_c2_loop_bend_mm": g(gr, "c2_loop", "landmark", "bend_radius_mm"),
               "wall": wall_record(case, sd, M, LM, E) if M else None,
               "wall_plain_ct_seg4_nofix": ({"whole": u3(g(Mnf, "full", sd, "wall_union")),
                                             "length_mm": rr(g(Mnf, "full", sd, "gross", "length_mm"), 1)}
                                            if g(Mnf, "full", sd) else None)}
        # song 10-02 01:25: protrusion.json (pilot_proc_song_protrusion.py = ia172's + LAS read + contamination classes)
        # and the ct_seg4 label truncation (seeds_fix: the label stops short of the station line -- the tight-bend pinch)
        PR = L(os.path.join(W, "protrusion.json")) or {}
        vv = "rva" if side == "R" else "lva"
        if rec["wall"] is not None:
            if PR.get(vv):
                rec["wall"]["protrusion"] = {k: v for k, v in PR[vv].items() if not k.startswith("per_10mm")}
            sf = g(seg_qc, "seeds_fix", vv) or {}
            if isinstance(sf, dict) and sf.get("label_s_range_mm") is not None:
                miss = (sf.get("missing_at_start_mm") or 0.0) + (sf.get("missing_at_end_mm") or 0.0)
                rec["wall"]["label_truncation"] = {"label_s_range_mm": sf.get("label_s_range_mm"),
                                                   "station_length_mm": sf.get("station_length_mm"),
                                                   "missing_mm": rr(miss, 2), "truncated": bool(miss > 5.0)}
        recs.append(rec)
    return crec, recs


def main():
    t0 = time.time()
    J = L(RES) or {"about": __doc__.split("\n\n")[0], "cases": {}, "records": []}
    for case in sys.argv[1:]:
        crec, recs = collect(case)
        J["cases"][case] = crec
        J["records"] = [r for r in J["records"] if r["case"] != case] + recs
        print(case, json.dumps([(r["side"], r["status"], r["passes_as_v5_donor"],
                                 (r["wall"] or {}).get("whole")) for r in recs]))
    J["records"].sort(key=lambda r: (r["case"], r["side"]))
    J["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    J["bars_native"] = BARS
    tmp = RES + ".tmp"
    json.dump(J, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, RES)
    print("wrote", RES, "%.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
