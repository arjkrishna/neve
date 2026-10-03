#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_calibration_gross.py (= /sp/v5_va/tag_calibration_gross.py in the build containers)
"""tag_calibration_gross.py -- gross reference ranges with TaG-Net A and B beside VMR 0248, the host and SPL.

    python3 tag_calibration_gross.py      -> /sp/v5_va/results/tag_calibration_gross.json     (eve-training-fixed)

New measurements here (everything else is read from the study / tag_* results, cited per field):
  * C2-loop bend radius: va_metrics.frenet (UNCHANGED, sigma 3 mm, the G2 definition) maximum curvature inside the
    window [C2 exit - 20 mm, C2 exit + 15 mm] (landmark window) and, path-only, inside s in [0.40, 0.65] x length
    (the proxy used for TaG-Net B, which has no vertebrae; checked against the landmark window on VMR / A / SPL);
  * SPL subclavian elevation at the VA origin and SPL take-off with the study's fwd5 definition
    (tag_labels_gross.takeoffs, imported unchanged) on the SPL stage-A VMTK centrelines.
LICENCE: TaG-Net -> internal only; VMR research use; SPL 3D Slicer licence B.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402  (unchanged)
from tag_labels_gross import takeoffs  # noqa: E402  (unchanged)

R = "/repo/carotid_data/va_sources/derived"
RES = "/sp/v5_va/results"


def rr(x, n=2):
    return None if x is None else round(float(x), n)


def bend_window(P, lo, hi, sigma=3.0):
    Pr, _, s = vm.resample(np.asarray(P, float), None)
    _, kap, _, _ = vm.frenet(Pr, sigma)
    core = (s >= vm.END_EXCL_MM) & (s <= s[-1] - vm.END_EXCL_MM)
    m = core & (s >= lo) & (s <= hi)
    if not m.any():
        return None
    i = np.nonzero(m)[0][np.argmax(kap[m])]
    j = np.nonzero(core)[0][np.argmax(kap[core])]
    return {"window_mm": [rr(lo, 1), rr(hi, 1)], "bend_radius_mm": rr(1.0 / kap[i]), "at_s_mm": rr(s[i], 1),
            "at_z_mm": rr(Pr[i, 2], 1), "global_min_bend_radius_mm": rr(1.0 / kap[j]), "global_at_s_mm": rr(s[j], 1),
            "length_mm": rr(s[-1], 1)}


def c2_bends(P, c2_s):
    L = vm.arclen(np.asarray(P, float))[-1]
    o = {"pathonly": bend_window(P, 0.40 * L, 0.65 * L)}
    if c2_s is not None:
        o["landmark"] = bend_window(P, c2_s - 20.0, c2_s + 15.0)
    return o


def main():
    out = {"about": __doc__.split("\n\n")[0], "vas": {}}
    # ---- TaG-Net A / B (GT points, tag_labels)
    gg = json.load(open(RES + "/tag_labels_gross.json"))["vas"]
    lmA = json.load(open(RES + "/tag_gt.json"))["landmarks_A"]
    for pt in ("A", "B"):
        for side, sd in (("R", "right"), ("L", "left")):
            g = json.load(open(R + "/tagnet/gt/%s_%s.json" % (pt, sd)))
            P = np.asarray(g["vertebral"]["points"], float)
            sm = gg["%s_%s" % (pt, sd)]["summary"]
            tk = gg["%s_%s" % (pt, sd)]["takeoff_variants"]
            lm = lmA[sd]["landmarks_s_from_ostium_mm"] if pt == "A" else None
            seg = lmA[sd]["segments_mm"] if pt == "A" else None
            out["vas"]["TaG-Net %s %s" % (pt, side)] = {
                "length_mm": rr(sm["length_mm"], 1), "DM": rr(sm["DM"], 3), "takeoff_10mm_fwd5_deg": sm["takeoff_10mm_fwd5_deg"],
                "takeoff_10mm_central7_deg": tk["takeoff_10mm_central7"],
                "min_bend_radius_s3_mm": sm["min_bend_radius_s3_mm"], "loops_360_60mm": sm["loops_360_60mm"],
                "c2_loop": c2_bends(P, lm["C2"] if lm else None),
                "V1_V4_mm": None if seg is None else [seg["V1"], seg["V2"], seg["V3"], seg["V4"]],
                "landmarks_s_mm": lm, "subclavian_elevation_fwd5_deg": tk["sca_fwd5_elevation_deg"],
                "va_10mm_elevation_deg": tk["va_10mm_chord_elevation_deg"],
                "r_median_mm": sm.get("r_median_mm"),
                "source": "tag_labels GT points (ostium -> merge), va_metrics sigma 3; landmarks semi-manual +-3 mm (A only)"}
    # ---- VMR 0248 (ct_seg4 VMTK; study values)
    vmr = json.load(open(RES + "/vmr0248_metrics.json"))
    tkm = json.load(open("/sp/v5_va/graft/takeoff_measure.json"))["donor"]
    for side, sd, ct in (("R", "right", "rva"), ("L", "left", "lva")):
        sfv = vmr["summary_full_va"]["ct_seg4_stageA"][sd]
        lm = vmr["landmarks_ct"][sd]
        c = json.load(open(R + "/vmr0248/ct_seg4/%s_stageA_vmtk.json" % ct))
        P = np.asarray(c["points"], float)
        V = [lm["C6"], lm["C2"] - lm["C6"], lm["dura"] - lm["C2"], sfv["length_mm"] - lm["dura"]]
        out["vas"]["VMR 0248 %s" % side] = {
            "length_mm": rr(sfv["length_mm"], 1), "DM": rr(sfv["DM"], 3),
            "takeoff_10mm_fwd5_deg": tkm[sd]["takeoff_10mm_fwd5"], "takeoff_10mm_central_deg": tkm[sd]["takeoff_10mm_central"],
            "takeoff_10mm_va_metrics_deg": (sfv.get("takeoff") or {}).get("takeoff_10mm_deg"),
            "min_bend_radius_s3_mm": sfv["min_bend_radius_s3_mm"], "loops_360_60mm": sfv["loops_360_60mm"],
            "c2_loop": c2_bends(P, lm["C2"]), "V1_V4_mm": [rr(x, 1) for x in V], "landmarks_s_mm": {k: lm[k] for k in ("C6", "C2", "C1", "dura")},
            "subclavian_elevation_fwd5_deg": tkm[sd]["subclavian_fwd5_elevation_deg"], "va_10mm_elevation_deg": tkm[sd]["va_10mm_elevation_deg"],
            "r_median_mm": sfv["r_median_mm"],
            "source": "vmr0248_metrics.json ct_seg4_stageA (CT segmentation VMTK), graft/takeoff_measure.json; left C6 disputed (verifier s ~37)"}
    # ---- SPL (label stage-A VMTK; spl_metrics_measure.json) + new: C2-loop window, elevation, fwd5 take-off
    spl = json.load(open(RES + "/spl_metrics_measure.json"))["vessels"]
    for side, sd in (("R", "right"), ("L", "left")):
        v = spl["%s_stageA" % sd]
        lm = v["spl_extras"]["landmarks_s_mm"]
        cl = json.load(open(R + "/spl/%s_va_stageA_cl.json" % sd))
        P = np.asarray(cl["points"], float)
        sa = np.asarray(json.load(open(R + "/spl/%s_subclavian_cl.json" % sd))["points"], float)
        # orient the subclavian proximal (medial) -> distal (lateral): the proximal end is nearer the midline
        o = {}
        try:
            xm = 0.5 * (np.asarray(json.load(open(R + "/spl/right_va_stageA_cl.json"))["points"])[0][0] +
                        np.asarray(json.load(open(R + "/spl/left_va_stageA_cl.json"))["points"])[0][0])
            if abs(sa[0, 0] - xm) > abs(sa[-1, 0] - xm):
                sa = sa[::-1]
            tk, _ = takeoffs(P, sa)
            o = tk
        except Exception as e:  # noqa: BLE001
            o = {"error": repr(e)}
        out["vas"]["SPL %s" % side] = {
            "length_mm_sigma3_report": 221.1 if side == "R" else 193.2, "DM_sigma3_report": 1.59 if side == "R" else 1.41,
            "length_raw_vmtk_mm": v["gross"]["length_mm"], "takeoff_10mm_report": "113-134 (origin hook)" if side == "R" else "40-50",
            "takeoff_fwd5_this_agent": o.get("takeoff_10mm_fwd5"), "min_bend_radius_s3_mm": v["gross"]["shape@3.0"]["min_bend_radius_mm"],
            "c2_loop": c2_bends(P, lm.get("C2")), "V1_V4_mm_report": [44.5, 109.8, 53.0, 39.5] if side == "R" else [34.8, 84.0, 64.5, 27.5],
            "subclavian_elevation_fwd5_deg": o.get("sca_fwd5_elevation_deg"), "va_10mm_elevation_deg": o.get("va_10mm_chord_elevation_deg"),
            "ostium_to_sca_axis_mm": o.get("ostium_to_sca_axis_mm"), "landmarks_s_mm_raw_vmtk": lm,
            "source": "SPL label stage-A VMTK (raw arclength; report lengths are sigma 3)"}
    # ---- host (report / ref / graft notes; synthetic)
    out["vas"]["host R"] = {"length_mm": 210.0, "DM": 1.39, "takeoff_10mm": "81-93 (ref 85, graft fwd5 92-93)", "min_bend_radius_s3_mm": 4.24,
                            "c2_loop": "none (synthetic; V3-like loop 80-200 mm, min bend 4.24 mm)", "V1_V4_mm": "synthetic",
                            "subclavian_elevation_fwd5_deg": 26.5, "va_10mm_elevation_deg": 61.2, "r_median_mm": 1.67,
                            "source": "report gross table; ref.md; verify_graft_wall (26.5 / 13.2 deg with a true 5 mm chord)"}
    out["vas"]["host L"] = {"length_mm": 209.0, "DM": 1.42, "takeoff_10mm": "76-81 (ref 81, graft fwd5 76-78)", "min_bend_radius_s3_mm": 5.39,
                            "c2_loop": "none (synthetic)", "V1_V4_mm": "synthetic", "subclavian_elevation_fwd5_deg": 13.2,
                            "va_10mm_elevation_deg": 80.2, "r_median_mm": 1.75, "source": "as host R"}
    # ---- TotalSegmentator whole VAs (study, corrected n = 11) and neck VAs (n = 15)
    am = json.load(open(RES + "/arterial_metrics.json"))["totalsegmentator"]["cases"]
    wrong = {("s0289", "R"), ("s0333", "L"), ("s0518", "L"), ("s1107", "R"), ("s1129", "R"), ("s1154", "R")}
    whole, neck = [], []
    for c in am:
        for sd in ("R", "L"):
            r_ = c["vas"][sd]
            if (c["subject"], sd) in wrong:
                continue
            if r_.get("usable_course"):
                whole.append(r_)
            if r_.get("usable_cervical"):
                neck.append(r_)

    def st(rows, k):
        a = [x.get(k) for x in rows if x.get(k) is not None]
        return None if not a else {"n": len(a), "median": rr(np.median(a)), "min": rr(min(a)), "max": rr(max(a)),
                                   "p10": rr(np.percentile(a, 10)), "p90": rr(np.percentile(a, 90))}
    out["vas"]["TS whole VAs (corrected)"] = {k: st(whole, k) for k in ("vmtk_length_mm", "DM", "takeoff_10mm_deg", "min_bend_radius_s3_mm", "r_median_mm")}
    out["vas"]["TS neck VAs (corrected)"] = {k: st(neck, k) for k in ("vmtk_length_mm", "DM", "takeoff_10mm_deg", "min_bend_radius_s3_mm", "r_median_mm")}
    out["ts_lists"] = {"whole": ["%s %s" % x for x in [(c["subject"], sd) for c in am for sd in ("R", "L")
                                                     if c["vas"][sd].get("usable_course") and (c["subject"], sd) not in wrong]],
                       "neck": ["%s %s" % x for x in [(c["subject"], sd) for c in am for sd in ("R", "L")
                                                    if c["vas"][sd].get("usable_cervical") and (c["subject"], sd) not in wrong]]}
    json.dump(out, open(RES + "/tag_calibration_gross.json", "w"), indent=1)
    for k, v in out["vas"].items():
        c2 = v.get("c2_loop")
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("source", "c2_loop", "landmarks_s_mm", "landmarks_s_mm_raw_vmtk")}),
              "| C2:", json.dumps(c2) if isinstance(c2, dict) else c2, flush=True)
    print("wrote", RES + "/tag_calibration_gross.json")


if __name__ == "__main__":
    main()
