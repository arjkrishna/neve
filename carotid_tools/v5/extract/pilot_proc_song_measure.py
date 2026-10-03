#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_measure.py (= /sp/v5_va/pilot_proc_song_measure.py in the build containers)
# SONG COPY (pilot_proc_song, 2026-10-01 21:20): pilot_proc_ia172_measure.py with the song paths (thin_slice/song), pilot_proc_song_ imports and the Song meta keys; diffs beyond that are stated in the docstring / marked 'song'.
"""pilot_proc_song_measure.py -- wall detail, validity gates and gross shape of one 172-series pilot donor's
CT-segmented (ct_seg4) VAs (PILOT_CASE), in the SAME harness as tag_ctwall_measure.py (TaG-Net A and VMR 0248):

  * va_metrics.measure_va UNCHANGED (VMTK points + MISR radii of the ct_seg4 stage-A surface, the surface itself,
    voxel_mm = the ACQUISITION voxel from meta.json (SONG: image.acquisition_voxel_mm = the native grid; ia172: in-plane x 0.5 mm, never the 0.25-0.4 mm
    slice interval and never Arterial's grid), landmarks = the "ts" set of pilot_proc_song_landmarks.py,
    starts_at_origin=True, parent_dir = the SCA fwd5 tangent at the ostium foot (the ct_seg4 recentred SCA stations),
    up = +z);
  * vmr_measure.union_segments (unchanged, imported) for the landmark stretches, the strategy-B seam split and 10 mm bins
    (tag_ctwall_measure.full_job, copied verbatim apart from the arguments);
  * EXACT ownership: va_metrics.cKDTree swapped at run time for tag_ctwall_common.KTree64 (k 24 -> 64), the file unedited;
  * va_metrics.section_detail per stretch;
  * pieces alone (tag_ctwall_measure.piece_job, copied): C6 -> C1 (ts and ct landmarks) and origin -> C1;
  * va_standard.grade per VA (validity gates V1-V6, W rules at native thresholds: the acquisition voxel <= 0.65 mm);
  * gross extras: take-off (tag_labels_gross.takeoffs, imported: fwd5 / central7 / ref3_4, SCA fwd5 elevation, VA 10 mm
    elevation), bends (vmr_compare.bends, sigma 3 and 1.5), C2-region minimum bend (tag_ctwall_assemble.bend_in's
    definition: C2 - 30 mm .. C1, sigma 3; on both landmark sets), V1-V4 lengths.
  * third instance (2026-10-01 20:40): the "rec" landmark set of landmarks.json (tag_calibration: Erec C6, ts_qc C2, C1f,
    ts_qc dura) added as a third stretch set ("rec:" prefix), a C6 -> C1 piece on it, and V1-V4 on it.
Writes <der>/ct_seg/<tag>/measure.json.  eve-training-fixed.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import va_metrics as vm  # noqa: E402
import va_standard as vs  # noqa: E402
from tag_ctwall_common import exact_ownership, jdump  # noqa: E402  (KTree64 swap; unchanged)

CASE = os.environ["PILOT_CASE"]
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % CASE
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % CASE
TAGS = [t for t in os.environ.get("PILOT_SEG_TAGS", "ct_seg4").split(",") if t]
UP = np.array([0.0, 0.0, 1.0])
BIN = 10.0
SEAM_MM = 25.0
RADIUS_KIND = "VMTK MISR (vmtkCenterlines pointlist, AppendEndPoints=1; neve-build-vmtk)"


def stretches(lm, L, prefix=""):
    """tag_ctwall_measure.stretches, verbatim."""
    out = {}
    g = lm.get
    for nm, a, b in (("V1", 0.0, g("C6")), ("V2", g("C6"), g("C2")), ("V3", g("C2"), g("dura")), ("V4", g("dura"), L),
                     ("V3_C2_to_C1", g("C2"), g("C1")), ("V3_C1_to_dura", g("C1"), g("dura")),
                     ("C6_to_C1", g("C6"), g("C1")), ("origin_to_C1", 0.0, g("C1"))):
        if a is not None and b is not None and b - a > 4.0:
            out[prefix + nm] = (float(a), float(b))
    if g("C1") is not None and g("C1") > SEAM_MM + 5:
        out[prefix + "origin_to_C1_minus_seam"] = (0.0, float(g("C1") - SEAM_MM))
        out[prefix + "seam_zone_C1m25_to_C1"] = (float(g("C1") - SEAM_MM), float(g("C1")))
        if g("C6") is not None:
            out[prefix + "C6_to_C1_minus_seam"] = (float(g("C6")), float(g("C1") - SEAM_MM))
    return out


def full_job(args):
    """tag_ctwall_measure.full_job, verbatim apart from the source label."""
    (src, rep, side, cl_path, surf_path, voxel, lm_primary, lm_extra, kmode, parent_dir) = args
    import pyvista as pv
    from vmr_measure import union_segments
    exact_ownership(vm, kmode == "k64")
    t0 = time.time()
    d = json.load(open(cl_path))
    P, R = np.asarray(d["points"], float), np.asarray(d["radii"], float)
    surf = pv.read(surf_path)
    res = vm.measure_va(P, R, surf, voxel_mm=voxel, source_type="%s CT segmentation %s" % (src, rep),
                        radius_kind=RADIUS_KIND, landmarks={k: lm_primary[k] for k in ("C6", "C2", "C1", "dura")},
                        starts_at_origin=True, parent_dir=parent_dir, up=UP, name="%s %s VA %s (%s)" % (src, side, rep, kmode),
                        extra={"centerline": cl_path, "surface": surf_path, "ownership": kmode,
                               "landmarks_primary": lm_primary})
    Pr, Rr, sr = vm.resample(P, R)
    L = float(sr[-1])
    ranges = stretches(lm_primary, L)
    for pre, lme in (lm_extra or {}).items():
        ranges.update(stretches(lme, L, prefix=pre + ":"))
    ranges["distal_0_40"] = (L - 40.0, L)
    ranges["distal_40_80"] = (L - 80.0, L - 40.0)
    edges = np.arange(0.0, L, BIN)
    bins = {"bin10:%d" % k: (float(a), float(min(a + BIN, L))) for k, a in enumerate(edges) if min(a + BIN, L) - a >= 5}
    allr = dict(ranges)
    allr.update(bins)
    us = union_segments(P, R, surf, allr)
    rows, nb = [], len(bins)
    for k, (key, (a, b)) in enumerate(bins.items()):
        tg = "core" if k in (0, nb - 1) else "all"
        u = us[key][tg]
        rows.append({"from_mm": round(a, 1), "to_mm": round(b, 1), "frac_0.3mm": u["frac_0.3mm"], "frac_0.5mm": u["frac_0.5mm"],
                     "frac_rel_0.15": u["frac_rel_0.15"], "area_mm2": u["area_mm2"], "measure": tg})
    secs = {}
    for nm, (a, b) in ranges.items():
        m = (sr >= a) & (sr <= b)
        if m.sum() < 40:
            continue
        secs[nm] = vm.section_detail(Pr[m], Rr[m], sr[m] - sr[m][0], surf)
        secs[nm]["r_misr_median"] = round(float(np.median(Rr[m])), 3)
    res["stretches"] = {"ranges_mm": {k: [round(a, 2), round(b, 2)] for k, (a, b) in ranges.items()},
                        "union": {k: us[k] for k in ranges}, "sections": secs, "whole_adapter": us["whole"],
                        "adapter_check": {"measure_va_core_0.3": res["wall_union"]["core"]["frac_0.3mm"],
                                          "adapter_core_0.3": us["whole"]["core"]["frac_0.3mm"]}}
    res["bins10"] = rows
    res["radius_profile"] = {"s": np.round(sr[::4], 2).tolist(), "r": np.round(Rr[::4], 3).tolist(),
                             "P": np.round(Pr[::4], 2).tolist()}
    res["seconds"] = round(time.time() - t0, 1)
    wu = res["wall_union"]["core"]
    print("%-5s %-16s %-5s %s L=%.1f r=%.2f DM=%.2f union %.4f/%.4f rel %.4f genus %s ecc %.3f sec %.3f (%.0fs)" % (
        src, rep, side, kmode, L, res["calibre"]["r_median_mm"], res["gross"]["DM"], wu["frac_0.3mm"], wu["frac_0.5mm"],
        wu["frac_rel_0.15"], res["wall_union"]["genus"], res["wall_sections"].get("centre_offset_mm_median", -1),
        res["wall_sections"].get("in_out_0.3_mean", -1), res["seconds"]), flush=True)
    return ("full", src, rep, side, kmode), res


def piece_job(args):
    """tag_ctwall_measure.piece_job, verbatim."""
    (src, rep, side, cl_path, surf_path, voxel, lo, hi, label, kmode) = args
    import pyvista as pv
    exact_ownership(vm, kmode == "k64")
    d = json.load(open(cl_path))
    P, R, s = vm.resample(np.asarray(d["points"]), np.asarray(d["radii"]))
    m = (s >= lo) & (s <= hi)
    surf = pv.read(surf_path)
    res = vm.measure_va(P[m], R[m], surf, voxel_mm=voxel, source_type="%s CT segmentation %s piece %s" % (src, rep, label),
                        radius_kind=RADIUS_KIND, starts_at_origin=False,
                        name="%s %s VA %s %s (%s)" % (src, side, label, rep, kmode),
                        extra={"piece_mm_on_full_centerline": [round(lo, 2), round(hi, 2)], "ownership": kmode})
    print("piece %s %s %s %s [%.1f, %.1f] union %.4f rel %.4f genus %s" % (src, rep, side, label, lo, hi,
          res["wall_union"]["core"]["frac_0.3mm"], res["wall_union"]["core"]["frac_rel_0.15"], res["wall_union"]["genus"]),
          flush=True)
    return ("piece", src, rep, side, label, kmode), res


def sca_lateral(st, side):
    P = np.asarray(st["rsub" if side == "right" else "lsub"]["P"], float)
    sg = 1.0 if side == "right" else -1.0
    if sg * (P[-1, 0] - P[0, 0]) < 0:
        P = P[::-1]
    return P


def main():
    t0 = time.time()
    meta = json.load(open(os.path.join(SRC, "meta.json")))
    voxel = tuple(float(v) for v in meta["image"]["acquisition_voxel_mm"])   # song meta key (= the native grid; no SliceThickness in the Song sidecars)
    from tag_labels_gross import takeoffs, at_s, foot_on, unit
    from vmr_compare import bends
    for tag in TAGS:
        D = os.path.join(DER, "ct_seg", tag)
        LM = json.load(open(os.path.join(D, "landmarks.json")))
        st = json.load(open(os.path.join(D, "stations_rays.json")))
        jobs, pjobs, gross = [], [], {}
        for side, v in (("right", "rva"), ("left", "lva")):
            cl = os.path.join(D, "%s_stageA_vmtk.json" % v)
            sf = os.path.join(D, "%s_stageA.vtp" % v)
            if not (os.path.isfile(cl) and side in LM["sides"]):
                continue
            rec = LM["sides"][side]
            lm_ts = rec["landmarks_used"]
            lm_ct = rec["landmarks_ct"]
            lm_rec = rec.get("landmarks_rec") or {}
            P = np.asarray(json.load(open(cl))["points"], float)
            sca = sca_lateral(st, side)
            O, s0, _ = foot_on(sca, P[0])
            pdir = unit(at_s(sca, s0 + 5.0) - O)
            extra = {}
            if all(lm_ct.get(k) is not None for k in ("C6", "C2", "C1", "dura")):
                extra["ct"] = lm_ct
            if all(lm_rec.get(k) is not None for k in ("C6", "C2", "C1", "dura")):
                extra["rec"] = lm_rec
            if all(lm_ts.get(k) is not None for k in ("C6", "C2", "C1", "dura")):
                jobs.append((CASE, "%s_stageA" % tag, side, cl, sf, voxel, lm_ts, extra, "k64", pdir))
            else:
                lm_fill = dict(lm_ts)
                for k in ("C6", "C2", "C1", "dura"):
                    if lm_fill.get(k) is None:
                        lm_fill[k] = lm_ct.get(k)
                jobs.append((CASE, "%s_stageA" % tag, side, cl, sf, voxel, lm_fill, extra, "k64", pdir))
            for lab, a, b in (("C6_to_C1_ts", lm_ts.get("C6"), lm_ts.get("C1")), ("C6_to_C1_ct", lm_ct.get("C6"), lm_ct.get("C1")),
                              ("origin_to_C1_ts", 0.0, lm_ts.get("C1")),
                              ("C6_to_C1_rec", lm_rec.get("C6"), lm_rec.get("C1")), ("origin_to_C1_rec", 0.0, lm_rec.get("C1"))):
                if a is not None and b is not None and b - a > 20:
                    pjobs.append((CASE, "%s_stageA" % tag, side, cl, sf, voxel, float(a), float(b), lab, "k64"))
            R = np.asarray(json.load(open(cl))["radii"], float)
            tk, _ = takeoffs(P, sca)
            from tag_ctwall_assemble import bend_in
            g = {"takeoff_study_definition_ct_recentred_sca": tk, "bends_s3": bends(P, R, 3.0), "bends_s1.5": bends(P, R, 1.5)}
            for nm, lmx in (("ts", lm_ts), ("ct", lm_ct), ("rec", lm_rec)):
                if lmx.get("C2") is not None and lmx.get("C1") is not None:
                    try:
                        g["C2_region_min_bend_s3_" + nm] = bend_in(P, lmx["C2"] - 30.0, lmx["C1"])
                    except Exception as e:  # noqa: BLE001
                        g["C2_region_min_bend_s3_" + nm] = "failed: %s" % e
                if all(lmx.get(k) is not None for k in ("C6", "C2", "dura")):
                    g["V1_V4_mm_" + nm] = [round(lmx["C6"], 1), round(lmx["C2"] - lmx["C6"], 1),
                                           round(lmx["dura"] - lmx["C2"], 1), round(lmx["end"] - lmx["dura"], 1)]
            gross[side] = g
        print("jobs", len(jobs), "pieces", len(pjobs), flush=True)
        with Pool(int(os.environ.get("PILOT_POOL", "4"))) as pool:
            out = pool.map(full_job, jobs, chunksize=1) + pool.map(piece_job, pjobs, chunksize=1)
        res = {"what": __doc__, "version": vm.VERSION, "case": CASE, "acquisition_voxel_mm": voxel, "full": {}, "pieces": {},
               "gross_extra": gross, "grades": {}}
        for key, r in out:
            if key[0] == "full":
                _, src, rep, side, km = key
                res["full"][side] = r
                res["grades"][side] = vs.grade(r)
            else:
                _, src, rep, side, label, km = key
                res["pieces"].setdefault(label, {})[side] = r
                res["grades"]["piece_%s_%s" % (label, side)] = vs.grade(r)
        res["seconds"] = round(time.time() - t0, 1)
        jdump(res, os.path.join(D, "measure.json"))
        print("done %.0f s" % res["seconds"], flush=True)


if __name__ == "__main__":
    main()
