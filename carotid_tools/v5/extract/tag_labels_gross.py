#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_labels_gross.py (= /sp/v5_va/tag_labels_gross.py in the build containers)
"""tag_labels_gross.py -- gross shape of the TaG-Net GT VAs with /sp/v5_va/va_metrics.py UNCHANGED (centreline mode).

    bash /sp/v5_va/tag_labels_run.sh eve-training-fixed tag_labels_gross.py

va_metrics.measure_va(points = GT vertebral.points (ostium -> merge, de-staircased sigma 0.75 mm), radii, surface=None,
do_union=False, starts_at_origin=True, parent_dir = the study's take-off tangent, up = +z): length, DM, sigma-3 shape
(curvature, min bend radius, turning, loops, U-turns, inflections, ICM, SOAM), take-off at 5 / 10 mm, elevation,
bins from origin / distal end. Patient A radii = image half-max r_eq; patient B has none -> a constant 1.5 mm is
passed only because measure_va requires radii (its calibre block is then dropped).

Take-off (same definition as the study: graft/takeoff_measure.py, graft_v5_va.takeoff_of): angle between the VA
chord ostium -> first point >= d mm along the VA and the parent subclavian's distal tangent at the branch point, the
tangent measured as
   fwd5      5 mm forward chord along the SCA from O_v = the foot of the ostium on the SCA centreline (graft_v5_va's
             forward_chord, copied below) -- the PRIMARY definition (VMR 53.5 / 58.7 deg);
   central7  chord SCA(s_foot - 7 mm) -> SCA(s_foot + 7 mm)  (~ VMR's +-5 expert path points, 1.3-1.4 mm apart);
   ref3_4    chord SCA(s_foot - 4.2 mm) -> SCA(s_foot + 4.2 mm) (~ ref_vmr0248's +-3 points);
and the fwd5 angle with the chord taken from O_v instead of the ostium ("from_branch_point").
Writes results/tag_labels_gross.json.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402

GT = "/repo/carotid_data/va_sources/derived/tagnet/gt"
RES = "/sp/v5_va/results"
OUT = os.path.join(RES, "tag_labels_gross.json")
UP = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1, 1))))


def foot_on(sp, X):
    """(foot point, its arclength) of X on the polyline sp (graft_v5_va.load_donor / forward_chord logic)."""
    s = vm.arclen(sp)
    k = int(np.argmin(np.linalg.norm(sp - X, axis=1)))
    best, q0, s0 = np.inf, sp[k], s[k]
    for a, b in ((max(k - 1, 0), k), (k, min(k + 1, len(sp) - 1))):
        if a == b:
            continue
        ab = sp[b] - sp[a]
        t = float(np.clip((X - sp[a]) @ ab / (ab @ ab), 0.0, 1.0))
        q = sp[a] + t * ab
        d = float(np.linalg.norm(q - X))
        if d < best:
            best, q0, s0 = d, q, s[a] + t * (s[b] - s[a])
    return q0, s0, best


def at_s(sp, s_q):
    s = vm.arclen(sp)
    return np.array([np.interp(s_q, s, sp[:, i]) for i in range(3)])


def d_chord(p, d):
    s = vm.arclen(p)
    return p[min(int(np.searchsorted(s, d)), len(p) - 1)] - p[0]


def takeoffs(P, sca):
    O, s0, dist = foot_on(sca, P[0])
    t_fwd5 = unit(at_s(sca, s0 + 5.0) - O)
    t_c7 = unit(at_s(sca, s0 + 7.0) - at_s(sca, s0 - 7.0))
    t_c4 = unit(at_s(sca, s0 + 4.2) - at_s(sca, s0 - 4.2))
    out = {"ostium_to_sca_axis_mm": round(dist, 2), "foot_arc_on_sca_mm": round(float(s0), 2),
           "sca_fwd5_elevation_deg": round(90.0 - angle(t_fwd5, UP), 1),
           "va_10mm_chord_elevation_deg": round(90.0 - angle(d_chord(P, 10.0), UP), 1)}
    s = vm.arclen(P)
    for d in (5.0, 10.0):
        for tag, t in (("fwd5", t_fwd5), ("central7", t_c7), ("ref3_4", t_c4)):
            out["takeoff_%dmm_%s" % (d, tag)] = round(angle(d_chord(P, d), t), 1)
        i = min(int(np.searchsorted(s, d)), len(P) - 1)
        out["takeoff_%dmm_fwd5_from_branch_point" % d] = round(angle(P[i] - O, t_fwd5), 1)
    return out, t_fwd5


def main():
    out = {"version": vm.VERSION, "definition": __doc__.split("\n\n")[1], "vas": {}, "vmr0248_side_by_side": {}}
    for pt in ("A", "B"):
        for sd in ("right", "left"):
            g = json.load(open(os.path.join(GT, "%s_%s.json" % (pt, sd))))
            v = g["vertebral"]
            P = np.asarray(v["points"], float)
            R = None if v["radius_mm"] is None else np.asarray(v["radius_mm"], float)
            sca = np.asarray(g["subclavian"]["points"], float)
            tk, t_fwd5 = takeoffs(P, sca)
            vox = [0.405, 0.405, 0.4] if pt == "A" else [0.366, 0.366, 0.4]
            m = vm.measure_va(P, R if R is not None else np.full(len(P), 1.5), surface=None, voxel_mm=vox,
                              source_type="TaG-Net ground-truth centreline label (1-voxel), de-staircased",
                              radius_kind=("image half-maximum equal-area radius (vmr_ct_seg3 rays)" if R is not None
                                           else "NONE (constant 1.5 mm placeholder; calibre not reported)"),
                              starts_at_origin=True, parent_dir=t_fwd5, up=UP, name="TaG-Net %s %s VA" % (pt, sd),
                              do_union=False)
            if R is None:
                m["calibre"] = None
            # full path (branch point -> confluence) length and DM for the record
            F = np.asarray(v["full_points"], float)
            sh = m["gross"]["shape@3.0"]
            rec = {"measure_va": m, "takeoff_variants": tk,
                   "summary": {"length_mm": m["gross"]["length_mm"], "chord_mm": m["gross"]["chord_mm"], "DM": m["gross"]["DM"],
                               "full_path_length_mm": round(float(vm.arclen(F)[-1]), 2),
                               "kappa_p50_s3": sh["kappa_p50"], "min_bend_radius_s3_mm": sh["min_bend_radius_mm"],
                               "turning_total_s3_deg": sh["turning_total_deg"], "SOAM_s3": sh["SOAM_deg_per_cm"],
                               "ICM_s3": sh["ICM"], "inflections_s3": sh["inflections"],
                               "loops_360_60mm": sh["n_loops_360_60mm"], "uturns_150_30mm": sh["n_uturns_150_30mm"],
                               "turn_max_60mm_deg": sh["turn_max_60mm_deg"], "reversal_max_30mm_deg": sh["reversal_max_30mm_deg"],
                               "min_bend_radius_s1.5_mm": m["gross"]["shape@1.5"]["min_bend_radius_mm"],
                               "takeoff_10mm_fwd5_deg": tk["takeoff_10mm_fwd5"],
                               "takeoff_va_metrics": m["gross"].get("takeoff"),
                               "r_median_mm": None if R is None else m["calibre"]["r_median_mm"],
                               "r_p10_p90_mm": None if R is None else [m["calibre"]["r_p10_mm"], m["calibre"]["r_p90_mm"]]}}
            # where is the tightest bend (sigma 3)?
            Pr, Rr, s = vm.resample(P, None)
            _, kap, _, _ = vm.frenet(Pr, 3.0)
            core = (s >= 2) & (s <= s[-1] - 2)
            i = int(np.nonzero(core)[0][np.argmax(kap[core])])
            rec["summary"]["tightest_bend_s3"] = {"s_mm": round(float(s[i]), 2), "z_mm": round(float(Pr[i, 2]), 1),
                                                  "radius_mm": round(1.0 / kap[i], 2)}
            out["vas"]["%s_%s" % (pt, sd)] = rec
            print(pt, sd, json.dumps(rec["summary"]), flush=True)
    # VMR 0248 side by side (study results, read-only)
    vmr = json.load(open(os.path.join(RES, "vmr0248_metrics.json")))["summary_full_va"]["ct_seg4_stageA"]
    ref = json.load(open(os.path.join(RES, "ref_e_vmr0248.json")))["vessels"]
    tkm = json.load(open("/sp/v5_va/graft/takeoff_measure.json"))["donor"]
    for sd in ("right", "left"):
        e = ref[sd + "_vertebral"]["declared_contour_req"]
        out["vmr0248_side_by_side"][sd] = {
            "ct_seg4_vmtk": {k: vmr[sd][k] for k in ("length_mm", "DM", "kappa_p50_s3", "min_bend_radius_s3_mm",
                                                     "loops_360_60mm", "takeoff", "r_median_mm")},
            "expert_path": {"length_mm": e["gross"]["length_mm"], "DM": e["gross"]["DM"],
                            "min_bend_radius_s3_mm": e["gross"]["shape@3.0"]["min_bend_radius_mm"],
                            "loops_360_60mm": e["gross"]["shape@3.0"]["n_loops_360_60mm"],
                            "takeoff": e["gross"].get("takeoff"), "r_median_contour_req_mm": e["calibre"]["r_median_mm"]},
            "graft_takeoff_measure": {k: tkm[sd][k] for k in ("takeoff_10mm_fwd5", "takeoff_10mm_central", "takeoff_10mm_ref3",
                                                              "takeoff_10mm_fwd5_from_branch_point",
                                                              "subclavian_fwd5_elevation_deg", "va_10mm_elevation_deg")}}
    for pt in ("A", "B"):
        ra = out["vas"]["%s_right" % pt]["summary"]["r_median_mm"]
        la = out["vas"]["%s_left" % pt]["summary"]["r_median_mm"]
        out["dominance_%s" % pt] = vm.dominance(la, ra) if ra and la else None
    json.dump(out, open(OUT, "w"), indent=1)
    print("dominance A", out["dominance_A"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
