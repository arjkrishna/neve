#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_totalseg_qc.py (= /sp/v5_va/tag_totalseg_qc.py in the build containers)
"""tag_totalseg_qc.py -- does TotalSegmentator 2.18.0 "total" (neve-totalseg-gpu) reproduce the TotalSegmentator v3
DATASET masks well enough that art_extract's trace QC gives the same answers, and does that QC run on TaG-Net
patient A's tool-made ts dir?

    python3 /sp/v5_va/tag_totalseg_qc.py dice s0289 s1192     (neve-build-stagea; mounts as in art_run.sh)
    python3 /sp/v5_va/tag_totalseg_qc.py gt                     (patient A, GT centrelines)

dice <sid>...: dataset dir /repo/carotid_data/va_sources/totalseg_v3/<sid> against the tool dir
  /repo/carotid_data/va_sources/derived/tagnet/totalseg_check/<sid> (ct.nii.gz there is a hard link to the dataset's):
  - per class of the 28 dataset masks: Dice, volume ratio tool/dataset, average symmetric surface distance and HD95
    (mm, surface voxels, EDT on a crop), z top / bottom differences;
  - C1 top (art_extract's z_FM = C1 top + 6 mm), skull lowest point;
  - arterial_ts_args.main (unchanged) on both dirs;
  - art_extract.load_ts + art_extract.ts_qc (unchanged, imported) on the EXISTING study trace
    derived/arterial/ts_<sid>/va_extract.json (sides.*.trace.points / start_on_sa_world) with both mask sets, plus the
    stored dataset-mask QC from that json as a check of this harness;
  - V3-loop numbers with verify_arterial_yield_v3loop.py's definitions (re-implemented here on the trace points, with
    ts_qc's dura proxy as the arclength limit): dy = y - y_cord(z) (cord centroid per 0.1 mm z bin, clamped above the
    top cord slice), V2 = C6 mid .. C3 top, V3 stretch = z >= C2 bottom and s <= dura, excursion = median dy(V2) -
    min dy(V3), d_C1post = min distance of the V3 stretch to C1 voxels with y < y_cord.
gt: the TaG-Net GT R-VA (NIfTI 8) / L-VA (13) centrelines of patient A ordered by a shortest path over the 26-connected
  GT voxels from the VA voxel touching the own SCA label (14 / 15) to the VA voxel touching the BA label (9), resampled
  0.5 mm and smoothed sigma 1 mm with art_extract.resample / smooth (as trace_side does), then ts_qc and the V3-loop
  numbers against the tool ts dir derived/tagnet/totalseg. "junction" = "origin" = the SCA-adjacent GT voxel (no
  lumen trim: the GT has no radius). Timing and peak RSS are recorded (the ts dir is on the 0.405 x 0.405 x 0.4 mm
  CTA grid, ~6x the voxels of a 1.5 mm TotalSegmentator subject).
writes /sp/v5_va/results/tag_totalseg_qc_<mode>.json
"""
import json
import os
import resource
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, "/sp/v5_va")
import art_extract as AE  # noqa: E402  (unchanged study tool)
import arterial_ts_args  # noqa: E402  (unchanged study tool)

DS = "/repo/carotid_data/va_sources/totalseg_v3"
TOOL = os.environ.get("TAG_TOOL_DIR", "/repo/carotid_data/va_sources/derived/tagnet/totalseg_check")  # total_v3: .../totalseg_check_v3
ARTD = "/repo/carotid_data/va_sources/derived/arterial"
TAG_TS = os.environ.get("TAG_TS_DIR", "/repo/carotid_data/va_sources/derived/tagnet/totalseg")
GT = "/repo/carotid_data/va_sources/tagnet/SampleData/001/labeled_cl_gt.nii.gz"
NEEDED = ["subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right",
          "common_carotid_artery_left", "brachiocephalic_trunk", "aorta", "skull", "spinal_cord", "trachea"] + \
         ["vertebrae_C%d" % k for k in range(1, 8)]


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def r(x, n=2):
    return None if x is None else round(float(x), n)


def i2w(A, I):
    return (A @ np.c_[I, np.ones(len(I))].T)[:3].T


def surf(m):
    return m & ~ndimage.binary_erosion(m)


def compare(a, b, sp, A):
    """a = dataset, b = tool (bool arrays on the same grid)."""
    na, nb = int(a.sum()), int(b.sum())
    o = {"dataset_voxels": na, "tool_voxels": nb}
    if na == 0 and nb == 0:
        o["dice"] = None
        return o
    inter = int((a & b).sum())
    o["dice"] = r(2.0 * inter / (na + nb), 4)
    o["vol_ratio_tool_over_dataset"] = r(nb / na, 3) if na else None
    if na == 0 or nb == 0:
        return o
    u = a | b
    idx = np.argwhere(u)
    m = np.ceil(10.0 / sp).astype(int)
    lo = np.maximum(idx.min(0) - m, 0); hi = np.minimum(idx.max(0) + m + 1, np.array(a.shape))
    sa_, sb_ = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    Sa, Sb = surf(sa_), surf(sb_)
    da = ndimage.distance_transform_edt(~Sa, sampling=sp)
    db = ndimage.distance_transform_edt(~Sb, sampling=sp)
    d_ab = db[Sa]; d_ba = da[Sb]
    allds = np.r_[d_ab, d_ba]
    o["assd_mm"] = r(allds.mean())
    o["hd95_mm"] = r(np.percentile(allds, 95))
    za = i2w(A, np.argwhere(a)[:, :][::3].astype(float))[:, 2]
    zb = i2w(A, np.argwhere(b)[::3].astype(float))[:, 2]
    o["dz_top_mm"] = r(zb.max() - za.max(), 1)
    o["dz_bottom_mm"] = r(zb.min() - za.min(), 1)
    return o


def cord_fn(ts):
    mm, aa = ts["masks"]["spinal_cord"]
    cW = i2w(aa, np.argwhere(mm).astype(float))
    zq = np.round(cW[:, 2], 1)
    zz = np.unique(zq)
    cy = np.array([cW[zq == z, 1].mean() for z in zz])
    cx = np.array([cW[zq == z, 0].mean() for z in zz])
    return zz, cy, cx


def v3loop(P, ts, z_fm, dura_s):
    """verify_arterial_yield_v3loop.py definitions on the path P (ordered origin -> distal)."""
    zz, cy, cx = cord_fn(ts)
    vz = {}
    for k in range(1, 8):
        key = "vertebrae_C%d" % k
        if key in ts["masks"]:
            mm, aa = ts["masks"][key]
            Wz = i2w(aa, np.argwhere(mm)[::3].astype(float))[:, 2]
            vz["C%d" % k] = (float(Wz.min()), float(Wz.max()))
    mm, aa = ts["masks"]["vertebrae_C1"]
    c1 = i2w(aa, np.argwhere(mm).astype(float))
    ycord_c1 = np.interp(c1[:, 2], zz, cy)
    c1post = c1[c1[:, 1] < ycord_c1]
    from scipy.spatial import cKDTree
    kd = cKDTree(c1post) if len(c1post) else None
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    du = dura_s if dura_s is not None else s[-1]
    m3 = (s <= du) & (P[:, 2] >= vz["C2"][0])
    z6m = np.mean(vz["C6"]); z3t = vz["C3"][1]
    m2 = (s <= du) & (P[:, 2] >= z6m) & (P[:, 2] <= z3t)
    dy = P[:, 1] - np.interp(P[:, 2], zz, cy)
    out = {}
    if m3.any():
        k = np.where(m3)[0][np.argmin(dy[m3])]
        out["dy_min_V3"] = r(dy[k], 1)
        out["z_at_dy_min_rel_C1top"] = r(P[k, 2] - vz["C1"][1], 1)
        out["d_C1_posterior_half_mm"] = r(kd.query(P[m3])[0].min(), 1) if kd is not None else None
    if m2.any():
        out["dy_V2_median"] = r(np.median(dy[m2]), 1)
    if "dy_min_V3" in out and "dy_V2_median" in out:
        out["posterior_excursion_mm"] = r(out["dy_V2_median"] - out["dy_min_V3"], 1)
    out["passes_v3loop_gate (excursion >= 15, d_C1post <= 5)"] = bool(out.get("posterior_excursion_mm", 0) >= 15 and
                                                                       (99 if out.get("d_C1_posterior_half_mm") is None else out["d_C1_posterior_half_mm"]) <= 5)
    return out


def qc_summary(q):
    if q is None:
        return None
    lv = q.get("vertebra_levels", {})
    return {"junction_to_TS_SA_mask_mm": q.get("junction_to_TS_SA_mask_mm"),
            "origin_to_TS_SA_mask_mm": q.get("origin_to_TS_SA_mask_mm"),
            "foramen_levels_passed": q.get("foramen_levels_passed"), "n_C6_to_C1_passed": q.get("n_C6_to_C1_passed"),
            "foramen_entry_level": q.get("foramen_entry_level"), "landmarks_mm": q.get("landmarks_mm"),
            "landmarks_complete": q.get("landmarks_complete"),
            "max_enclosure": {k: v.get("max_enclosure") for k, v in lv.items()},
            "median_dist_to_vertebra_mm": {k: v.get("median_dist_to_vertebra_mm") for k, v in lv.items()}}


def run_args(d):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        arterial_ts_args.main(d)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def mode_dice(sids):
    res = {}
    for sid in sids:
        t0 = time.time()
        dd, td = os.path.join(DS, sid), os.path.join(TOOL, sid)
        ct = nib.load(os.path.join(dd, "ct.nii.gz"))
        A, shape = ct.affine, ct.shape
        sp = np.sqrt((A[:3, :3] ** 2).sum(0))
        o = {"ct_shape": list(shape), "spacing_mm": sp.round(3).tolist(), "classes": {}}
        names = sorted(f[:-7] for f in os.listdir(os.path.join(dd, "segmentations")) if f.endswith(".nii.gz"))
        offgrid = []
        for nm in names:
            ia = nib.load(os.path.join(dd, "segmentations", nm + ".nii.gz"))
            ib = nib.load(os.path.join(td, "segmentations", nm + ".nii.gz"))
            if ib.shape != ia.shape or np.abs(ib.affine - ia.affine).max() > 1e-4:
                offgrid.append(nm)
            a = np.asanyarray(ia.dataobj) > 0
            b = np.asanyarray(ib.dataobj) > 0
            o["classes"][nm] = compare(a, b, sp, A)
        o["tool_masks_off_dataset_grid"] = offgrid
        dn = [o["classes"][k]["dice"] for k in NEEDED if o["classes"].get(k, {}).get("dice") is not None]
        o["needed_classes"] = NEEDED
        o["needed_dice_median"] = r(np.median(dn), 4)
        o["needed_dice_min"] = r(min(dn), 4)
        o["needed_dice_min_class"] = NEEDED[[o["classes"].get(k, {}).get("dice") for k in NEEDED].index(min(dn))]
        print(sid, {k: (o["classes"][k]["dice"], o["classes"][k].get("assd_mm"), o["classes"][k].get("dz_top_mm"),
                        o["classes"][k].get("dz_bottom_mm")) for k in NEEDED if k in o["classes"]}, flush=True)
        # arterial args with both
        o["arterial_ts_args"] = {"dataset": run_args(dd), "tool": run_args(td)}
        print(sid, "args", json.dumps(o["arterial_ts_args"]), flush=True)
        # QC on the existing study trace
        ex_p = os.path.join(ARTD, "ts_" + sid, "va_extract.json")
        o["qc"] = {}
        if os.path.isfile(ex_p):
            ex = json.load(open(ex_p))
            tsd = AE.load_ts(dd, None, None)
            tst = AE.load_ts(td, None, None)
            for side in ("R", "L"):
                tr0 = ex["sides"].get(side, {}).get("trace", {})
                if not tr0.get("ok"):
                    o["qc"][side] = {"trace_ok": False}
                    continue
                tr = {"P": np.asarray(tr0["points"], float), "start_on_sa_world": tr0["start_on_sa_world"]}
                row = {"trace_length_mm": r(tr0.get("length_mm"), 1)}
                for lab, ts in (("dataset", tsd), ("tool", tst)):
                    z_fm = ts["z_c1_top"] + 6.0
                    q = AE.ts_qc(tr, ts, side, z_fm)
                    row[lab] = qc_summary(q)
                    row[lab]["z_c1_top_mm"] = r(ts["z_c1_top"], 1)
                    row[lab]["v3loop"] = v3loop(tr["P"], ts, z_fm, q.get("landmarks_mm", {}).get("dura"))
                row["stored_in_va_extract (dataset masks, study run)"] = qc_summary(ex["sides"][side].get("ts_qc"))
                o["qc"][side] = row
                print(sid, side, json.dumps({k: (v.get("foramen_entry_level"), v.get("landmarks_mm"), v.get("origin_to_TS_SA_mask_mm"),
                                                 v.get("v3loop", {}).get("posterior_excursion_mm"), v.get("v3loop", {}).get("d_C1_posterior_half_mm"))
                                             for k, v in row.items() if isinstance(v, dict)}), flush=True)
            del tsd, tst
        o["seconds"] = r(time.time() - t0, 1)
        res[sid] = o
    return res


def order_gt_va(g, A, va, sca, ba):
    """Shortest path over the 26-connected GT VA voxels from the SCA-adjacent voxel to the BA-adjacent voxel."""
    P = np.argwhere(g == va)
    idx = {tuple(p): i for i, p in enumerate(P)}
    offs = np.argwhere(np.ones((3, 3, 3), bool)) - 1
    offs = offs[np.any(offs != 0, axis=1)]
    sp = np.sqrt((A[:3, :3] ** 2).sum(0))
    rows, cols, w = [], [], []
    deg = np.zeros(len(P), int)
    for i, p in enumerate(P):
        for d in offs:
            j = idx.get(tuple(p + d))
            if j is not None:
                rows.append(i); cols.append(j); w.append(float(np.linalg.norm(d * sp))); deg[i] += 1
    G = coo_matrix((w, (rows, cols)), shape=(len(P), len(P))).tocsr()
    sca_set = set(map(tuple, np.argwhere(g == sca)))
    ba_set = set(map(tuple, np.argwhere(g == ba)))

    def touching(S):
        return [i for i, p in enumerate(P) if any(tuple(p + d) in S for d in offs)]
    s_i, e_i = touching(sca_set), touching(ba_set)
    info = {"n_voxels": int(len(P)), "n_touch_SCA": len(s_i), "n_touch_BA": len(e_i),
            "n_degree1": int((deg == 1).sum()), "n_degree_ge3": int((deg >= 3).sum())}
    _lab, ncomp = ndimage.label(g == va, structure=np.ones((3, 3, 3)))
    info["n_components"] = int(ncomp)
    if not s_i or not e_i:
        info["error"] = "no SCA- or BA-adjacent VA voxel"
        return None, info
    dist, pred = dijkstra(G, indices=s_i[0], return_predecessors=True)
    e = min(e_i, key=lambda j: dist[j])
    if not np.isfinite(dist[e]):
        info["error"] = "SCA end and BA end not connected within the VA label"
        return None, info
    path = [e]
    while path[-1] != s_i[0]:
        path.append(pred[path[-1]])
    path = path[::-1]
    info["path_voxels"] = len(path)
    info["voxels_off_path"] = int(len(P) - len(path))
    W = i2w(A, P[path].astype(float))
    info["path_length_mm"] = r(AE.arclen(W)[-1], 1)
    return W, info


def mode_gt():
    t0 = time.time()
    gi = nib.load(GT)
    g = np.asanyarray(gi.dataobj)
    A = gi.affine
    ts = AE.load_ts(TAG_TS, None, None)
    t_load = time.time() - t0
    z_fm = ts["z_c1_top"] + 6.0
    res = {"ts_dir": TAG_TS, "z_c1_top_mm": r(ts["z_c1_top"], 1), "z_fm_mm": r(z_fm, 1), "load_ts_seconds": r(t_load, 1),
           "load_ts_has_ct": "ct" in ts, "cca_hu_used_by_load_ts": ts.get("cca_hu"),
           "screening_trace_available": bool(ts.get("trace")), "sides": {}}
    for side, va, sca in (("R", 8, 14), ("L", 13, 15)):
        t1 = time.time()
        W, info = order_gt_va(g, A, va, sca, 9)
        row = {"gt_path": info}
        if W is not None:
            Pr, _, _ = AE.resample(W, 0.5)
            Ps = AE.smooth(Pr, 1.0, 0.5)
            Ps[0], Ps[-1] = Pr[0], Pr[-1]
            tr = {"P": Ps, "start_on_sa_world": W[0].tolist()}
            q = AE.ts_qc(tr, ts, side, z_fm)
            row["ts_qc"] = q
            row["summary"] = qc_summary(q)
            row["v3loop"] = v3loop(Ps, ts, z_fm, q.get("landmarks_mm", {}).get("dura"))
            row["length_mm"] = r(AE.arclen(Ps)[-1], 1)
        row["seconds"] = r(time.time() - t1, 1)
        res["sides"][side] = row
        print(side, json.dumps({k: v for k, v in row.items() if k != "ts_qc"}, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)
    res["seconds"] = r(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dice":
        out = mode_dice(sys.argv[2:])
    else:
        out = mode_gt()
    p = "/sp/v5_va/results/tag_totalseg_qc_%s%s.json" % (mode, os.environ.get("TAG_OUT_SUFFIX", ""))
    json.dump(out, open(p, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", p, "peak RSS GB", rss_gb())
