#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_calibration_paths.py (= /sp/v5_va/tag_calibration_paths.py in the build containers)
"""tag_calibration_paths.py -- ONE QC harness for every vertebral-artery (VA) path the calibration needs: the ground
truth (GT) of VMR 0248 and TaG-Net patient A (and B, path-only), their CT-segmentation centrelines, every Arterial
mask trace / VMTK line of both GT patients at every benchmark quality, and the 40 traced sides of the 21
TotalSegmentator (TS) Arterial runs (24 subjects tried).

    python3 tag_calibration_paths.py <group> [<group> ...]       group = VMR | A | B | TS
    -> /sp/v5_va/results/tag_calibration_paths_<group>.json         (neve-build-stagea)

Per path and mask set:
  * art_extract.ts_qc (UNCHANGED study tool, imported): origin distance to the own TS subclavian mask, vertebra
    enclosure (16 rays, 8 mm), foramen levels, entry level, C6 / C2 / C1 / dura landmarks;
  * tag_totalseg_qc.v3loop (UNCHANGED; = verify_arterial_yield_v3loop's definitions): posterior excursion of V3 vs V2
    relative to the TS cord, distance to the posterior half of the C1 mask, V3 stretch = z >= C2 bottom, s <= dura;
  * NEW, this agent (definitions in the notes, notes/tag_calibration.md):
      - per vertebra level C7..C1: distance of the path to the level mask (TS vertebra masks fill the transverse
        foramina, so a VA in a foramen is INSIDE the mask), the ts_qc enclosure profile recomputed with 8 / 5 / 4 mm
        rays (the 8 mm loop is a verbatim copy of ts_qc's), the longest run of path inside / within 1 mm of the mask
        with z inside the level's OWN z range;
      - entry-level rules E0 (= ts_qc), E1 ("inside run": lowest level with >= 3 mm of path within 1.0 mm of its own
        mask, z inside its own range), E2 (5 mm rays, enclosure >= 0.75 over >= 3 mm, own z range), E3 (ts_qc's rule
        plus median distance <= 1.5 mm over the level's points);
      - landmarks from E1: C6 = start of the entry run, C2 = end of the last C2 run, C1f = end of the first C1 run
        after C2 (exit from the C1 transverse foramen = VMR's C1 convention);
      - V3 loop "zcap": the study measure with the stretch z >= C2 bottom up to the first point above C1 top + 10 mm
        (no dura proxy, so no dependence on the C1 top through z_FM);
      - V2 foramen band (ts_metrics2 / ts_rank rule): V2 = z between C6 mid and C2 mid; median |dx|, median dy and
        IQR(dy) relative to the TS cord centroid at the same z; band = |dx| 12-22, dy 7-19, IQR <= 4 mm;
      - HU along the path (trilinear on the ts-dir CT; the 1.5 mm copy for A / VMR, i.e. the resolution of a TS
        donor) over the cervical stretch (s <= dura) and V2, CCA median HU (ts_screen.hu_stats rule: 1-voxel eroded
        mask, median; both CCAs, median of the two), both brachiocephalic veins likewise; for A and VMR also HU on the
        native CT;
  * path-only (no masks; for TaG-Net B): the V3-loop proxy "pathonly": V2 band = s in [0.20, 0.45] L, V3 window =
    s in [0.45, 0.90] L (L = path length), excursion_raw = median y(V2) - min y(V3) (RAS, + anterior), and
    excursion_fit = max over V3 of (linear y(z) fit on V2, extrapolated) - y.
Mask sets: VMR -> derived/tagnet/calibration/vmr_totalseg_v3_1p5 (TotalSegmentator 2.18.0 total_v3 run here on the
VMR native CT, 1.5 mm RAS copy by tag_totalseg_make1p5.py, unchanged) and vmr_totalseg_1p5 ("total");
A -> derived/tagnet/totalseg_v3_1p5 and totalseg_1p5 (tag_totalseg agent); TS subjects -> the dataset masks
carotid_data/va_sources/totalseg_v3/<sid> (as the study QC).
LICENCE: TaG-Net has no data licence -> everything derived from A / B is internal only. VMR 0248: research use, cite
Wilson et al. 2013. TS: CC BY 4.0. Arterial-derived traces: non-commercial.
"""
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/sp/v5_va")
import art_extract as AE  # noqa: E402  (unchanged study tool)
from tag_totalseg_qc import v3loop, qc_summary, cord_fn  # noqa: E402  (unchanged)

R = "/repo/carotid_data/va_sources"
DER = R + "/derived"
CAL = DER + "/tagnet/calibration"
OUT = "/sp/v5_va/results"
MASKSETS = {"VMR": {"total_v3": CAL + "/vmr_totalseg_v3_1p5", "total": CAL + "/vmr_totalseg_1p5"},
            "A": {"total_v3": DER + "/tagnet/totalseg_v3_1p5", "total": DER + "/tagnet/totalseg_1p5"}}
NATIVE_CT = {"VMR": DER + "/arterial/vmr0248/input/vmr0248_ct_native.nii.gz",
             "A": R + "/tagnet/SampleData/002/CTA.nii.gz"}
A_ETAGS = sorted(d for d in os.listdir(DER + "/tagnet/arterial") if d.startswith("tagA_"))
VMR_ETAGS = ["vmr_native", "vmr_1p5", "vmr_1p5_nclean", "vmr_1p5_nsharp"]


def rr(x, n=2):
    return None if x is None else round(float(x), n)


def i2w(A, I):
    return (A @ np.c_[I, np.ones(len(I))].T)[:3].T


def runs(mask):
    """[(i0, i1)] inclusive index runs of True."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            out.append((start, i if v else i - 1))
            start = None
    return out


# ----------------------------------------------------------------------------- ts dir helpers (cached)
_TS = {}


def crop_ts(ts, lo, hi):
    """Crop every mask (and the CT, same grid) of a load_ts dict to the world box [lo, hi]; affines are shifted so
    world coordinates are unchanged (ts_qc is translation-invariant in index space). Used for the TS dataset subjects,
    whose 1.5 mm volumes reach 500 mm (one EDT per level per ts_qc call on the full grid made a side take ~10 min and
    the first full run was OOM-killed)."""
    A = ts["A"]
    Ainv = np.linalg.inv(A)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    I = (Ainv @ np.c_[C, np.ones(8)].T)[:3].T
    shape = next(iter(ts["masks"].values()))[0].shape
    i0 = np.clip(np.floor(I.min(0)).astype(int), 0, np.array(shape) - 1)
    i1 = np.clip(np.ceil(I.max(0)).astype(int), 0, np.array(shape) - 1)
    sl = tuple(slice(a, b + 1) for a, b in zip(i0, i1))
    T = np.eye(4); T[:3, 3] = i0
    for nm, (mm, aa) in list(ts["masks"].items()):
        if mm.shape == shape and np.abs(aa - A).max() < 1e-4:
            ts["masks"][nm] = (np.ascontiguousarray(mm[sl]), aa @ T)
    if "ct" in ts and ts["ct"].shape == shape and np.abs(ts["ct_affine"] - A).max() < 1e-4:
        ts["ct"] = np.ascontiguousarray(ts["ct"][sl]); ts["ct_affine"] = ts["ct_affine"] @ T
    ts["A"] = A @ T
    ts["crop"] = {"i0": i0.tolist(), "i1": i1.tolist(), "shape_full": list(shape)}
    return ts


def ts_load(d, crop=None, hu_from=None):
    if d in _TS:
        return _TS[d]
    ts = AE.load_ts(d, None, None)
    if crop is not None:
        ts = crop_ts(ts, crop[0], crop[1])
    ext = {"edt": {}, "z": {}}
    for k in range(1, 8):
        key = "vertebrae_C%d" % k
        if key in ts["masks"]:
            mm, aa = ts["masks"][key]
            ext["edt"][k] = ndimage.distance_transform_edt(~mm, sampling=ts["sp"]).astype(np.float32)
            W = i2w(aa, np.argwhere(mm).astype(float))
            ext["z"][k] = (float(W[:, 2].min()), float(W[:, 2].max()))
    ext["cord"] = cord_fn(ts)
    ext["cord_robust"] = cord_robust(ts)
    # HU stats as ts_screen.hu_stats (1-voxel erosion, median)
    hu = {}
    if hu_from is not None:      # TS dataset subject: the screening values (ts_screen.hu_stats, same rule) on the full CT
        for nm in ("common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left",
                   "brachiocephalic_vein_right", "subclavian_artery_right", "subclavian_artery_left"):
            h = (hu_from.get("hu") or {}).get(nm)
            if h:
                hu[nm] = {"n": h.get("n"), "eroded": h.get("eroded"), "med": rr(h.get("med"), 1), "from": "ts_screen"}
    elif "ct" in ts:
        ct = ts["ct"]
        for nm in ("common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left",
                   "brachiocephalic_vein_right", "subclavian_artery_right", "subclavian_artery_left"):
            p = os.path.join(d, "segmentations", nm + ".nii.gz")
            if not os.path.isfile(p):
                continue
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.sum() < 5:
                continue
            e = ndimage.binary_erosion(m)
            use = e if e.sum() >= 20 else m
            hu[nm] = {"n": int(use.sum()), "eroded": bool(use is e), "med": rr(np.median(ct[use]), 1)}
    ext["hu"] = hu
    cc = [hu[k]["med"] for k in ("common_carotid_artery_left", "common_carotid_artery_right") if k in hu]
    ext["cca_med_HU"] = rr(np.median(cc), 1) if cc else None
    bc = [hu[k]["med"] for k in ("brachiocephalic_vein_left", "brachiocephalic_vein_right") if k in hu]
    ext["bcv_min_HU"] = rr(min(bc), 1) if bc else None
    ext["bcv_max_HU"] = rr(max(bc), 1) if bc else None
    _TS[d] = (ts, ext)
    return ts, ext


def sample(vol, A, P, order=1, cval=-1000.0):
    I = (np.linalg.inv(A) @ np.c_[P, np.ones(len(P))].T)[:3]
    return ndimage.map_coordinates(vol, I, order=order, mode="constant", cval=cval)


# ----------------------------------------------------------------------------- per-level profiles + rules
def level_profiles(P, ts, ext):
    """Arrays along the path resampled at 1 mm (as ts_qc): distance to each level mask, enclosure with 8 / 5 / 4 mm
    rays (16 rays in the normal plane, 0.5 mm steps; the 8 mm variant is ts_qc's loop verbatim)."""
    Pr, _, s = AE.resample(P, 1.0)
    T = AE.tangents(Pr, 3.0, 1.0)
    a = np.cross(T, [0, 0, 1.0]); bad = np.linalg.norm(a, axis=1) < 1e-3
    a[bad] = np.cross(T[bad], [1.0, 0, 0]); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(T, a)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    prof = {"s": s, "z": Pr[:, 2], "P": Pr}
    for k in range(7, 0, -1):
        key = "vertebrae_C%d" % k
        if key not in ts["masks"]:
            continue
        mm, aa = ts["masks"][key]
        Ainv = np.linalg.inv(aa)
        hit = {L: np.zeros((len(Pr), 16), bool) for L in (8.0, 5.0, 4.0)}
        for j, th in enumerate(ang):
            dirs = np.cos(th) * a + np.sin(th) * b
            for st in np.arange(0.5, 8.01, 0.5):
                X = Pr + st * dirs
                I = (Ainv @ np.c_[X, np.ones(len(X))].T)[:3]
                v = ndimage.map_coordinates(mm.astype(np.uint8), I, order=0, mode="constant", cval=0) > 0
                for L in hit:
                    if st <= L + 1e-9:
                        hit[L][:, j] |= v
        dd = ndimage.map_coordinates(ext["edt"][k], (Ainv @ np.c_[Pr, np.ones(len(Pr))].T)[:3], order=1, mode="nearest")
        prof["C%d" % k] = {"enc8": hit[8.0].mean(1), "enc5": hit[5.0].mean(1), "enc4": hit[4.0].mean(1), "d": dd,
                           "zr": ext["z"][k]}
    return prof


RULES = ("E0", "E0own", "E0r5", "E1", "E1h", "E1z", "E2", "E2h", "E3", "E5")
RULE_DEF = {
    "E0": "ts_qc (unchanged): enclosure (16 rays, 8 mm) >= 0.5 anywhere with z in the level's range +- 3 mm",
    "E0own": "E0 restricted to the level's OWN z range (no +- 3 mm margin)",
    "E0r5": "E0 with 5 mm rays",
    "E1": "inside run: >= 3 mm of path within 1.0 mm of the level mask, z in its own range",
    "E1h": "inside run: >= 3 mm of path within 0.5 mm of the level mask, z in its own range",
    "E1z": "inside run: >= 3 mm of path INSIDE the level mask (EDT 0), z in its own range",
    "E2": "enclosure with 5 mm rays >= 0.75 over >= 3 mm, z in its own range",
    "E2h": "enclosure with 4 mm rays >= 0.75 over >= 3 mm, z in its own range",
    "E3": "E0 and median distance to the mask over the level's points <= 1.5 mm",
    "E5": "majority-labelled foramen runs: path within 0.5 mm of ANY C1-C7 mask, runs merged over gaps <= 2 mm, kept if >= 3 mm, each labelled by the level holding most of its points; entry = label of the first run labelled C7..C3; level count = E1h"}


def entry_rules(prof, min_run_mm=3.0):
    s, z = prof["s"], prof["z"]
    res = {"per_level": {}, "rule_definitions": RULE_DEF}

    def runs_ok(m):
        return [(i0, i1) for (i0, i1) in runs(m) if (s[i1] - s[i0] + 1.0) >= min_run_mm]
    for k in range(7, 0, -1):
        L = "C%d" % k
        if L not in prof:
            continue
        p = prof[L]
        z0, z1 = p["zr"]
        own = (z >= z0) & (z <= z1)
        inlev = (z >= z0 - 3) & (z <= z1 + 3)
        rl = {"E1": runs_ok(own & (p["d"] <= 1.0)), "E1h": runs_ok(own & (p["d"] <= 0.5)),
              "E1z": runs_ok(own & (p["d"] <= 1e-6)), "E2": runs_ok(own & (p["enc5"] >= 0.75)),
              "E2h": runs_ok(own & (p["enc4"] >= 0.75))}
        row = {"z_range_mm": [rr(z0, 1), rr(z1, 1)], "n_inlev": int(inlev.sum()), "n_own": int(own.sum()),
               "max_enc8_inlev": rr(p["enc8"][inlev].max(), 3) if inlev.any() else 0.0,
               "max_enc8_own": rr(p["enc8"][own].max(), 3) if own.any() else 0.0,
               "max_enc5_inlev": rr(p["enc5"][inlev].max(), 3) if inlev.any() else 0.0,
               "median_d_inlev_mm": rr(np.median(p["d"][inlev]), 2) if inlev.any() else None,
               "min_d_own_mm": rr(p["d"][own].min(), 2) if own.any() else None,
               "longest_within1mm_run_mm": rr(max([s[i1] - s[i0] + 1.0 for i0, i1 in runs(own & (p["d"] <= 1.0))] or [0]), 1),
               "longest_inside_run_mm": rr(max([s[i1] - s[i0] + 1.0 for i0, i1 in runs(own & (p["d"] <= 1e-6))] or [0]), 1)}
        for nm, lst in rl.items():
            row[nm + "_runs_s_mm"] = [[rr(s[i0], 1), rr(s[i1], 1)] for i0, i1 in lst]
            row[nm + "_passed"] = len(lst) > 0
        row["E0_passed"] = bool(inlev.any() and p["enc8"][inlev].max() >= 0.5)
        row["E0own_passed"] = bool(own.any() and p["enc8"][own].max() >= 0.5)
        row["E0r5_passed"] = bool(inlev.any() and p["enc5"][inlev].max() >= 0.5)
        row["E3_passed"] = bool(row["E0_passed"] and np.median(p["d"][inlev]) <= 1.5)
        res["per_level"][L] = row
    pl = res["per_level"]
    for rule in RULES:
        e = None
        for k in (7, 6, 5, 4, 3):
            if pl.get("C%d" % k, {}).get(rule + "_passed"):
                e = "C%d" % k
                break
        res[rule + "_entry"] = e
        res[rule + "_n_C6_to_C1"] = int(sum(1 for k in (6, 5, 4, 3, 2, 1) if pl.get("C%d" % k, {}).get(rule + "_passed")))
        res[rule + "_levels"] = [L for L in ("C7", "C6", "C5", "C4", "C3", "C2", "C1") if pl.get(L, {}).get(rule + "_passed")]
    # E5: majority-labelled foramen runs. inside_any = within 0.5 mm of ANY C1..C7 mask; runs merged across gaps
    # <= 2 mm, kept if >= 3 mm; each run is labelled by the level holding most of its points (d_k <= 0.5); entry = the
    # label of the first run labelled C7..C3. (TS masks fill the transverse foramina; at the C7/C6 boundary the C7
    # label bleeds into the lower rim of the C6 foramen, so a C6 entry starts with 3-7 mm "inside C7" that is
    # CONTIGUOUS with the C6 run; a C7 entry would give its own ~10 mm C7 run.)
    levs = [k for k in range(7, 0, -1) if "C%d" % k in prof]
    if levs:
        D = np.stack([prof["C%d" % k]["d"] for k in levs])
        ins = D.min(0) <= 0.5
        rl = runs(ins)
        merged = []
        for i0, i1 in rl:
            if merged and s[i0] - s[merged[-1][1]] <= 3.0:     # gap of <= 2 mm between runs (1 mm sampling)
                merged[-1] = (merged[-1][0], i1)
            else:
                merged.append((i0, i1))
        lab_runs = []
        for i0, i1 in merged:
            if s[i1] - s[i0] + 1.0 < min_run_mm:
                continue
            cnts = (D[:, i0:i1 + 1] <= 0.5).sum(1)
            lab_runs.append({"s_mm": [rr(s[i0], 1), rr(s[i1], 1)], "level": "C%d" % levs[int(np.argmax(cnts))],
                             "counts": {"C%d" % k: int(c) for k, c in zip(levs, cnts) if c}})
        res["E5_runs"] = lab_runs
        first = next((r_ for r_ in lab_runs if r_["level"] in ("C7", "C6", "C5", "C4", "C3")), None)
        res["E5_entry"] = first["level"] if first else None
        res["E5_n_C6_to_C1"] = res["E1h_n_C6_to_C1"]
        res["E5_levels"] = res["E1h_levels"]
    # run-rule landmarks: entry = start of the first run of the entry level; C2 = end of the last C2 run;
    # C1f = end of the first C1 run after C2 (exit of the C1 transverse foramen); C1_last = end of the last C1 run
    for rule in ("E1", "E1h", "E1z", "E2", "E2h"):
        lm = {}
        e = res[rule + "_entry"]
        if e:
            lm["entry"] = pl[e][rule + "_runs_s_mm"][0][0]
        if pl.get("C2", {}).get(rule + "_runs_s_mm"):
            lm["C2"] = pl["C2"][rule + "_runs_s_mm"][-1][1]
        if pl.get("C1", {}).get(rule + "_runs_s_mm"):
            after = [r_ for r_ in pl["C1"][rule + "_runs_s_mm"] if r_[1] > lm.get("C2", -1)]
            if after:
                lm["C1f"] = after[0][1]
                lm["C1_last"] = pl["C1"][rule + "_runs_s_mm"][-1][1]
        res[rule + "_landmarks_mm"] = lm
    # E5 landmarks: entry = start of the first labelled run; C2 / C1f as E1h
    lm5 = {k: v for k, v in (res.get("E1h_landmarks_mm") or {}).items() if k != "entry"}
    first = next((r_ for r_ in (res.get("E5_runs") or []) if r_["level"] in ("C7", "C6", "C5", "C4", "C3")), None)
    if first:
        lm5["entry"] = first["s_mm"][0]
    res["E5_landmarks_mm"] = lm5
    return res


def cord_robust(ts):
    """Cord centroid per 0.1 mm z bin (as cord_fn) keeping only slices with >= 50 % of the median slice voxel count:
    the top slices of the TS cord mask (at the foramen magnum, where the V3 loop's most posterior point lies) are
    partial and their centroid jumps by several mm between mask sets; np.interp then clamps to the last full slice."""
    mm, aa = ts["masks"]["spinal_cord"]
    cW = i2w(aa, np.argwhere(mm).astype(float))
    zq = np.round(cW[:, 2], 1)
    zz = np.unique(zq)
    n = np.array([(zq == z).sum() for z in zz])
    cy = np.array([cW[zq == z, 1].mean() for z in zz])
    cx = np.array([cW[zq == z, 0].mean() for z in zz])
    ok = n >= 0.5 * np.median(n)
    return zz[ok], cy[ok], cx[ok], {"top_full_slice_z": rr(zz[ok].max(), 1), "top_any_slice_z": rr(zz.max(), 1),
                                    "n_partial_top_slices": int((zz > zz[ok].max()).sum())}


def v3_zcap(P, ts, ext, zcap_above_c1top=10.0, robust=False):
    zz, cy, cx = ext["cord"] if not robust else ext["cord_robust"][:3]
    vz = ext["z"]
    mm, aa = ts["masks"]["vertebrae_C1"]
    c1 = i2w(aa, np.argwhere(mm).astype(float))
    c1post = c1[c1[:, 1] < np.interp(c1[:, 2], zz, cy)]
    from scipy.spatial import cKDTree
    kd = cKDTree(c1post) if len(c1post) else None
    s = AE.arclen(P)
    hi = np.where(P[:, 2] >= vz[1][1] + zcap_above_c1top)[0]
    s_end = s[hi[0]] if len(hi) else s[-1]
    m3 = (s <= s_end) & (P[:, 2] >= vz[2][0])
    m2 = (P[:, 2] >= np.mean(vz[6])) & (P[:, 2] <= vz[3][1]) & (s <= s_end)
    dy = P[:, 1] - np.interp(P[:, 2], zz, cy)
    o = {"s_end_mm": rr(s_end, 1)}
    if m3.any() and m2.any():
        k = np.where(m3)[0][np.argmin(dy[m3])]
        o["dy_min_V3"] = rr(dy[k], 1)
        o["s_at_dy_min_mm"] = rr(s[k], 1)
        o["z_at_dy_min_rel_C1top"] = rr(P[k, 2] - vz[1][1], 1)
        o["dy_V2_median"] = rr(np.median(dy[m2]), 1)
        o["posterior_excursion_mm"] = rr(o["dy_V2_median"] - o["dy_min_V3"], 1)
        o["d_C1_posterior_half_mm"] = rr(kd.query(P[m3])[0].min(), 1) if kd is not None else None
    return o


def v3_pathonly(P):
    s = AE.arclen(P)
    L = s[-1]
    v2 = (s >= 0.20 * L) & (s <= 0.45 * L)
    v3 = (s >= 0.45 * L) & (s <= 0.90 * L)
    y2 = float(np.median(P[v2, 1]))
    k = np.where(v3)[0][np.argmin(P[v3, 1])]
    b, a = np.polyfit(P[v2, 2], P[v2, 1], 1)
    res_ = (a + b * P[v3, 2]) - P[v3, 1]
    return {"excursion_raw_mm": rr(y2 - P[k, 1], 1), "s_at_min_y_mm": rr(s[k], 1), "frac_L_at_min_y": rr(s[k] / L, 3),
            "excursion_fit_mm": rr(res_.max(), 1), "V2_slope_dy_dz": rr(b, 3), "length_mm": rr(L, 1)}


def v2_band(P, ts, ext, dura_s=None):
    zz, cy, cx = ext["cord"]
    vz = ext["z"]
    za, zb = np.mean(vz[6]), np.mean(vz[2])
    s = AE.arclen(P)
    sel = (P[:, 2] >= za) & (P[:, 2] <= zb)
    if dura_s is not None:
        sel &= s <= dura_s
    if sel.sum() < 4:
        return {"n": int(sel.sum())}
    dx = np.abs(P[sel, 0] - np.interp(P[sel, 2], zz, cx))
    dy = P[sel, 1] - np.interp(P[sel, 2], zz, cy)
    o = {"n": int(sel.sum()), "abs_dx_med": rr(np.median(dx), 1), "dy_med": rr(np.median(dy), 1),
         "dy_iqr": rr(np.subtract(*np.percentile(dy, [75, 25])), 1)}
    o["in_band"] = bool(12 <= o["abs_dx_med"] <= 22 and 7 <= o["dy_med"] <= 19 and o["dy_iqr"] <= 4)
    o["in_band_no_iqr"] = bool(12 <= o["abs_dx_med"] <= 22 and 7 <= o["dy_med"] <= 19)
    return o


def path_hu(P, ts, ext, dura_s, v2_sel=None):
    if "ct" not in ts:
        return None
    s = AE.arclen(P)
    h = sample(ts["ct"], ts["ct_affine"], P)
    o = {"whole_med": rr(np.median(h), 1), "cervical_med": rr(np.median(h[s <= dura_s]), 1) if dura_s else None}
    cca = ext["cca_med_HU"]
    if cca:
        o["whole_over_cca"] = rr(o["whole_med"] / cca, 3)
        o["cervical_over_cca"] = rr(o["cervical_med"] / cca, 3) if o["cervical_med"] is not None else None
    return o


_NAT = {}


def native_hu(group, P, dura_s):
    if group not in NATIVE_CT:
        return None
    if group not in _NAT:
        im = nib.load(NATIVE_CT[group])
        _NAT[group] = (np.asanyarray(im.dataobj).astype(np.float32), im.affine)
    v, A = _NAT[group]
    s = AE.arclen(P)
    h = sample(v, A, P)
    return {"whole_med": rr(np.median(h), 1), "cervical_med": rr(np.median(h[s <= dura_s]), 1) if dura_s else None}


def one(group, P, junction, side, msname, msdir, crop=None, hu_from=None):
    P = np.asarray(P, float)
    ts, ext = ts_load(msdir, crop, hu_from)
    z_fm = ts["z_c1_top"] + 6.0
    tr = {"P": P, "start_on_sa_world": list(map(float, junction))}
    q = AE.ts_qc(tr, ts, side, z_fm)
    dura = (q.get("landmarks_mm") or {}).get("dura")
    prof = level_profiles(P, ts, ext)
    o = {"mask_set": msname, "ts_dir": msdir, "z_c1_top_mm": rr(ts["z_c1_top"], 1), "z_fm_mm": rr(z_fm, 1),
         "ts_qc": qc_summary(q), "v3loop_study": v3loop(P, ts, z_fm, dura), "v3loop_zcap": v3_zcap(P, ts, ext), "v3loop_robust": v3_zcap(P, ts, ext, robust=True),
         "cord_top": ext["cord_robust"][3],
         "entry": entry_rules(prof), "v2_band": v2_band(P, ts, ext, dura),
         "hu_ts_ct": path_hu(P, ts, ext, dura), "cca_med_HU": ext["cca_med_HU"], "bcv_min_HU": ext["bcv_min_HU"],
         "bcv_max_HU": ext["bcv_max_HU"], "hu_masks": ext["hu"]}
    # also with E1 entry-landmark-based dura proxy: none (dura is ts_qc's)
    o["native_ct_hu"] = native_hu(group, P, dura)
    o["profile_1mm"] = {"s": [rr(x, 1) for x in prof["s"]], "z": [rr(x, 1) for x in prof["z"]],
                        "d": {L: [rr(x, 2) for x in prof[L]["d"]] for L in prof if L.startswith("C")},
                        "enc5": {L: [rr(x, 2) for x in prof[L]["enc5"]] for L in prof if L.startswith("C")}}
    return o


def orient(Q, start):
    Q = np.asarray(Q, float)
    if np.linalg.norm(Q[-1] - start) < np.linalg.norm(Q[0] - start):
        Q = Q[::-1]
    return Q


# ----------------------------------------------------------------------------- path sources
def paths_vmr():
    from sources import load_vmr0248
    vmr = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
    V = vmr["vessels"]
    out = []
    for side, nm, ct in (("R", "right_vertebral", "rva"), ("L", "left_vertebral", "lva")):
        P = np.asarray(V[nm]["points"], float)
        full = np.asarray(V[nm]["full_points"], float)
        out.append({"id": "VMR expert %s" % side, "side": side, "kind": "gt_expert", "quality": "gt", "P": P, "junction": full[0]})
        c = json.load(open(DER + "/vmr0248/ct_seg4/%s_stageA_vmtk.json" % ct))
        out.append({"id": "VMR ct_seg4 %s" % side, "side": side, "kind": "gt_ct", "quality": "gt",
                    "P": orient(c["points"], P[0]), "junction": orient(c["points"], P[0])[0]})
        for et in VMR_ETAGS:
            ex = json.load(open(DER + "/arterial/%s/va_extract.json" % et))
            tr = ex["sides"][side]["trace"]
            if not tr.get("ok"):
                continue
            T = np.asarray(tr["points"], float)
            out.append({"id": "VMR %s trace %s" % (et, side), "side": side, "kind": "trace", "quality": et[4:], "etag": et,
                        "P": T, "junction": tr["start_on_sa_world"]})
            vp = DER + "/arterial/%s/%sva_stageA_vmtk.json" % (et, side.lower())
            if os.path.isfile(vp):
                Q = orient(json.load(open(vp))["points"], T[0])
                out.append({"id": "VMR %s vmtk %s" % (et, side), "side": side, "kind": "vmtk", "quality": et[4:], "etag": et,
                            "P": Q, "junction": Q[0]})
    return out


def paths_A():
    out = []
    for side, sd, ct in (("R", "right", "rva"), ("L", "left", "lva")):
        g = json.load(open(DER + "/tagnet/gt/A_%s.json" % sd))
        v = g["vertebral"]
        full = np.asarray(v["full_points"], float)
        P = np.asarray(v["points"], float)
        out.append({"id": "A gt %s" % side, "side": side, "kind": "gt_points", "quality": "gt", "P": P, "junction": full[0]})
        out.append({"id": "A gt_full %s" % side, "side": side, "kind": "gt_full", "quality": "gt", "P": full, "junction": full[0]})
        c = json.load(open(DER + "/tagnet/ct_seg/ct_seg4/%s_stageA_vmtk.json" % ct))
        Q = orient(c["points"], P[0])
        out.append({"id": "A ct_seg4 %s" % side, "side": side, "kind": "gt_ct", "quality": "gt", "P": Q, "junction": Q[0]})
        for et in A_ETAGS:
            ex = json.load(open(DER + "/tagnet/arterial/%s/va_extract.json" % et))
            tr = ex["sides"][side]["trace"]
            if not tr.get("ok"):
                continue
            T = np.asarray(tr["points"], float)
            qual = et[5:]
            out.append({"id": "A %s trace %s" % (et, side), "side": side, "kind": "trace", "quality": qual, "etag": et,
                        "P": T, "junction": tr["start_on_sa_world"]})
            vp = DER + "/tagnet/arterial/%s/%sva_stageA_vmtk.json" % (et, side.lower())
            if os.path.isfile(vp):
                Q = orient(json.load(open(vp))["points"], T[0])
                if len(Q) > 5:
                    out.append({"id": "A %s vmtk %s" % (et, side), "side": side, "kind": "vmtk", "quality": qual, "etag": et,
                                "P": Q, "junction": Q[0]})
    return out


def run_group(group):
    t0 = time.time()
    res = {"about": __doc__.split("\n\n")[0], "group": group, "paths": {}}
    if group == "B":
        for side, sd in (("R", "right"), ("L", "left")):
            g = json.load(open(DER + "/tagnet/gt/B_%s.json" % sd))
            P = np.asarray(g["vertebral"]["points"], float)
            res["paths"]["B gt %s" % side] = {"side": side, "kind": "gt_points", "pathonly": v3_pathonly(P),
                                              "length_mm": rr(AE.arclen(P)[-1], 1)}
        return res
    if group in ("VMR", "A"):
        plist = paths_vmr() if group == "VMR" else paths_A()
        for p in plist:
            row = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in p.items() if k not in ("P", "junction")}
            row["length_mm"] = rr(AE.arclen(p["P"])[-1], 1)
            row["pathonly"] = v3_pathonly(p["P"])
            sets = MASKSETS[group]
            use = sets.keys() if (p["kind"].startswith("gt") or p["quality"] in ("native", "native_tsskull", "native_oracle")) \
                else ["total_v3"]
            for ms in use:
                try:
                    row[ms] = one(group, p["P"], p["junction"], p["side"], ms, sets[ms])
                except Exception as e:  # noqa: BLE001
                    row[ms] = {"error": repr(e)}
            res["paths"][p["id"]] = row
            v = row.get("total_v3", {})
            print("%-40s L %6.1f | E0 %s E1 %s lm %s | v3 study %s/%s robust %s/%s | po %s | band %s | %.0fs" % (
                p["id"], row["length_mm"], v.get("entry", {}).get("E0_entry"), v.get("entry", {}).get("E1_entry"),
                v.get("entry", {}).get("E1_landmarks_mm"),
                v.get("v3loop_study", {}).get("posterior_excursion_mm"), v.get("v3loop_study", {}).get("d_C1_posterior_half_mm"),
                v.get("v3loop_robust", {}).get("posterior_excursion_mm"), v.get("v3loop_robust", {}).get("d_C1_posterior_half_mm"),
                row["pathonly"]["excursion_raw_mm"], v.get("v2_band", {}).get("in_band"), time.time() - t0), flush=True)
    if group == "TS":
        subs = sorted(d for d in os.listdir(DER + "/arterial") if d.startswith("ts_s"))
        only = os.environ.get("TAG_SUBJECTS")
        if only:
            subs = [s for s in subs if s[3:] in only.split(",")]
        part = os.path.join(OUT, "tag_calibration_paths_TS%s.json" % os.environ.get("TAG_OUT_SUFFIX", ""))
        if os.environ.get("TAG_RESUME") and os.path.isfile(part):
            res["paths"] = json.load(open(part))["paths"]
        for et in subs:
            sid = et[3:]
            if "%s R" % sid in res["paths"] and "%s L" % sid in res["paths"]:
                continue
            msdir = R + "/totalseg_v3/" + sid
            ex = json.load(open(DER + "/arterial/%s/va_extract.json" % et))
            # crop box: every path point of both sides (trace + VMTK) +- 30 mm, joined with C1..C7 +- 15 mm
            pts = []
            for sd_ in ("R", "L"):
                t_ = ex["sides"][sd_]["trace"]
                if t_.get("ok"):
                    pts.append(np.asarray(t_["points"], float))
                vp_ = DER + "/arterial/%s/%sva_stageA_vmtk.json" % (et, sd_.lower())
                if os.path.isfile(vp_):
                    q_ = np.asarray(json.load(open(vp_))["points"], float)
                    if len(q_):
                        pts.append(q_)
            vb = []
            for k in range(1, 8):
                p_ = os.path.join(msdir, "segmentations", "vertebrae_C%d.nii.gz" % k)
                if os.path.isfile(p_):
                    im_ = nib.load(p_)
                    mk_ = np.asanyarray(im_.dataobj) > 0
                    if mk_.any():
                        vb.append(i2w(im_.affine, np.argwhere(mk_)[::7].astype(float)))
            crop = None
            if pts:
                Pall = np.vstack(pts)
                lo, hi = Pall.min(0) - 30.0, Pall.max(0) + 30.0
                if vb:
                    V = np.vstack(vb)
                    lo, hi = np.minimum(lo, V.min(0) - 15.0), np.maximum(hi, V.max(0) + 15.0)
                crop = (lo, hi)
            scr = "/sp/v5_va/results/ts_screen/%s.json" % sid
            hu_from = json.load(open(scr)) if os.path.isfile(scr) else None
            for side in ("R", "L"):
                tr = ex["sides"][side]["trace"]
                if not tr.get("ok"):
                    res["paths"]["%s %s" % (sid, side)] = {"trace_ok": False, "why": tr.get("why")}
                    continue
                T = np.asarray(tr["points"], float)
                row = {"subject": sid, "side": side, "trace_ok": True, "length_mm": rr(AE.arclen(T)[-1], 1),
                       "pathonly": v3_pathonly(T)}
                row["trace"] = one("TS", T, tr["start_on_sa_world"], side, "dataset", msdir, crop, hu_from)
                vp = DER + "/arterial/%s/%sva_stageA_vmtk.json" % (et, side.lower())
                if os.path.isfile(vp):
                    Q = orient(json.load(open(vp))["points"], T[0])
                    if len(Q) > 5:
                        row["vmtk"] = one("TS", Q, Q[0], side, "dataset", msdir, crop, hu_from)
                        row["vmtk_pathonly"] = v3_pathonly(Q)
                        row["vmtk_length_mm"] = rr(AE.arclen(Q)[-1], 1)
                # stored study QC (reproduction check)
                row["stored_ts_qc"] = qc_summary(ex["sides"][side].get("ts_qc"))
                res["paths"]["%s %s" % (sid, side)] = row
                v = row["trace"]
                print("%s %s L %6.1f | E0 %s (stored %s) E1 %s lm %s | v3 %s/%s robust %s/%s | po %s | band %s | %.0fs" % (
                    sid, side, row["length_mm"], v["entry"]["E0_entry"], (row["stored_ts_qc"] or {}).get("foramen_entry_level"),
                    v["entry"]["E1_entry"], v["entry"]["E1_landmarks_mm"],
                    v["v3loop_study"].get("posterior_excursion_mm"), v["v3loop_study"].get("d_C1_posterior_half_mm"),
                    v["v3loop_robust"].get("posterior_excursion_mm"), v["v3loop_robust"].get("d_C1_posterior_half_mm"),
                    row["pathonly"]["excursion_raw_mm"], v["v2_band"].get("in_band"), time.time() - t0), flush=True)
            _TS.clear()
            json.dump(res, open(part, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else
                      (o.tolist() if isinstance(o, np.ndarray) else str(o)))
    res["seconds"] = rr(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    for grp in sys.argv[1:]:
        out = run_group(grp)
        sfx = os.environ.get("TAG_OUT_SUFFIX", "")
        p = os.path.join(OUT, "tag_calibration_paths_%s%s.json" % (grp, sfx))
        json.dump(out, open(p, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else
                  (o.tolist() if isinstance(o, np.ndarray) else str(o)))
        print("wrote", p, flush=True)
