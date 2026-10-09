#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\va\sc_post.py (= /sp/v6/va/sc_post.py in the build containers)
"""sc_post.py -- turn the wall agent's subclavian runs (wall/subclavian_requests.json -> ctseg6 request mode) into the
per-side junction record graft_v6_va reads: /work/subclavian/seg/current/<donor>/<side>/junction.json

Input per donor (v6_wall): /repo/carotid_data/v6_work/wall/subclavian/<donor_id>/
    {r,l}junction_stageA.vtp   subclavian label + the VA's first 25 mm, one closed surface (stage-A protocol)
    labels_kji.npz             the run's labels (4 rsub, 5 lsub, 1 rva, 2 lva), origin_mm, spacing_mm
    seg_qc.json                incl. the requested vs covered stretch
Output junction.json: surface (path readable in every container), its genus / closedness, the inscribed radius along
MY path (EDT of subclavian + VA labels, sampled on the path every 1 mm over the covered stretch) as misr_path [[s, r]],
the covered s range, the CT half-max-free wall-detail proxy (share of surface points of the stretch more than 0.3 mm
off a tube of the local median equivalent radius; D5), and the source files' mtimes.
Runs in neve-build-stagea (numpy, scipy, pyvista). Idempotent (rewrites).
    python3 sc_post.py [donor_id ...]
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

WALL = os.environ.get("SC_WALL", "/repo/carotid_data/v6_work/wall/subclavian")
PATHS = "/work/subclavian/paths"
OUT = "/work/subclavian/seg/current"
SUBLAB = {"R": 4, "L": 5}
VALAB = {"R": 1, "L": 2}


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def genus_closed(path):
    import pyvista as pv
    m = pv.read(path)
    m = m.extract_surface().triangulate().clean()
    m1 = m.connectivity("largest")
    e = m1.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
    V, F = m1.n_points, m1.n_cells
    E = F * 3 // 2
    chi = V - E + F
    return {"n_points": int(m.n_points), "components": int(m.connectivity().point_data["RegionId"].max() + 1)
            if m.n_points else 0, "open_or_nonmanifold_edges": int(e.n_cells), "genus_largest": float((2 - chi) / 2)}, m1


def one(donor, side):
    d = os.path.join(WALL, donor)
    tag = "r" if side == "R" else "l"
    surf = os.path.join(d, "%sjunction_stageA.vtp" % tag)
    lab = os.path.join(d, "labels_kji.npz")
    if not (os.path.isfile(surf) and os.path.isfile(lab)):
        return None
    pj = json.load(open(os.path.join(PATHS, "%s_%s.json" % (donor, side))))
    P = np.asarray(pj["path"], float)
    s = arclen(P)
    z = np.load(lab)
    L = z["labels"]
    org, sp = np.asarray(z["origin_mm"], float), np.asarray(z["spacing_mm"], float)
    m = (L == SUBLAB[side]) | (L == VALAB[side])
    msub = L == SUBLAB[side]
    edt = ndimage.distance_transform_edt(m, sampling=(sp[2], sp[1], sp[0]))   # array [k, j, i]
    ijk = (P - org) / sp                                                       # (i, j, k) per point
    inside = np.all((ijk >= 0) & (ijk <= np.array(L.shape[::-1]) - 1), axis=1)
    r = np.full(len(P), np.nan)
    if inside.any():
        q = ijk[inside]
        r[inside] = ndimage.map_coordinates(edt, [q[:, 2], q[:, 1], q[:, 0]], order=1, mode="nearest")
    # covered = where the subclavian label exists within 1.5 voxels of the path
    lab_at = np.zeros(len(P), bool)
    if inside.any():
        ii = np.round(ijk[inside]).astype(int)
        lab_at[np.nonzero(inside)[0]] = msub[ii[:, 2], ii[:, 1], ii[:, 0]] | m[ii[:, 2], ii[:, 1], ii[:, 0]]
    cov = s[lab_at & (r > 0.5)]
    misr = [[float(a), float(b)] for a, b in zip(s[::2], r[::2]) if np.isfinite(b) and b > 0.5]
    gi, mesh = genus_closed(surf)
    # wall-detail proxy on the stretch [cov0, ostium + 20]: distance of surface points to the path minus the local
    # median equivalent radius; share > 0.3 mm (the D5 measure's threshold, here against the path's own tube)
    from scipy.spatial import cKDTree
    pts = np.asarray(mesh.points, float)
    t = cKDTree(P)
    dd, ii = t.query(pts)
    s_pt = s[ii]
    lo_s = float(cov.min()) if len(cov) else 0.0
    hi_s = float(pj["s_ostium_mm"]) + 20.0
    sel = (s_pt >= lo_s) & (s_pt <= hi_s) & (np.abs(s_pt - pj["s_ostium_mm"]) > 6.0)   # the ostium itself excluded
    detail = None
    if sel.sum() > 50:
        rr = np.zeros(len(P))
        for k in range(len(P)):
            w = (np.abs(s_pt - s[k]) <= 5.0) & sel
            rr[k] = np.median(dd[w]) if w.sum() > 10 else np.nan
        dev = np.abs(dd[sel] - rr[ii[sel]])
        dev = dev[np.isfinite(dev)]
        detail = {"share_gt_0.3mm": float((dev > 0.3).mean()), "n_points": int(len(dev)),
                  "note": "surface points of the subclavian stretch (ostium +-6 mm excluded) vs a tube of the local "
                          "median distance (10 mm window); proxy of the union > 0.3 mm measure"}
    qc = {}
    qf = os.path.join(d, "seg_qc.json")
    if os.path.isfile(qf):
        q = json.load(open(qf))
        qc = (q.get("v6") or {}).get("subclavian_requests") or {}
    out = {"donor_id": donor, "side": side, "surface": surf, "labels": lab, "surface_info": gi,
           "covered_s_mm": [float(cov.min()), float(cov.max())] if len(cov) else None,
           "want_s_mm": [float(pj.get("ts_origin", {}).get("s_on_path_mm", 12.0) or 12.0) + 10.0, float(pj["s_ostium_mm"]) + 20.0],
           "misr_path": misr, "misr_median_stretch": float(np.nanmedian(r[(s >= lo_s) & (s <= hi_s)])) if len(cov) else None,
           "wall_detail_proxy": detail, "wall_qc": qc, "frame": "donor RAS mm (unmirrored); graft_v6_va mirrors",
           "mtimes": {f: time.strftime("%F %T", time.localtime(os.path.getmtime(f))) for f in (surf, lab)},
           "built": time.strftime("%F %T")}
    od = os.path.join(OUT, donor, side)
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "junction.json"), "w"), indent=1)
    return out


def main():
    donors = sys.argv[1:] or sorted(os.path.basename(x) for x in glob.glob(os.path.join(WALL, "*")) if os.path.isdir(x))
    for dn in donors:
        for side in ("R", "L"):
            try:
                o = one(dn, side)
            except Exception as e:  # noqa: BLE001
                print("ERROR", dn, side, type(e).__name__, e, flush=True)
                continue
            if o is None:
                print("missing", dn, side, flush=True)
                continue
            print("%-20s %s cov %s misr med %.2f genus %s open %d detail %s" % (
                dn, side, o["covered_s_mm"] and [round(v, 1) for v in o["covered_s_mm"]], o["misr_median_stretch"] or -1,
                o["surface_info"]["genus_largest"], o["surface_info"]["open_or_nonmanifold_edges"],
                o["wall_detail_proxy"] and round(o["wall_detail_proxy"]["share_gt_0.3mm"], 3)), flush=True)


if __name__ == "__main__":
    main()
