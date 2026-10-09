#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\va\sc_post3.py (= /sp/v6/va/sc_post3.py in the build containers)
"""sc_post3.py -- junction records for graft_v6_va from the wall agent's v3 junctions (published 2026-10-04 00:56:
jfix/junction6.py = cleaned subclavian label (1-voxel opening + closing) + a CT-lumen bridge at the ostium crotch;
marker subclavian/<donor>/.junction_v3). Supersedes sc_post2.py's own capsule-bridged surfaces.

Per donor side: surface = /repo/carotid_data/v6_work/wall/subclavian/<donor>/<x>junction_stageA.vtp (theirs, unchanged);
inscribed radius along MY subclavian path from the EDT of their final junction mask (junction_labels_kji.npz
<x>junction), the covered stretch, genus / open edges, the wall-detail proxy (as sc_post.py), and their bridge QC
(seg_qc.json v6.junction_bridge.<x>junction). A side is marked junction_failed when the bridge fell back to a tube or its
bottleneck stays < 0.5 of the VA's inscribed radius (wall agent: ia172_071 L, bottleneck 0.54 mm vs a 2.1 mm VA) ->
graft_v6_va keeps v5's junction there.
Output: /work/subclavian/seg/v6va3/<donor>/<R|L>/junction.json (+ _summary.json). neve-build-stagea. Idempotent.
    python3 sc_post3.py [donor_id ...]
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/sp/v6/va")
import sc_post as S  # noqa: E402

WALL = "/repo/carotid_data/v6_work/wall/subclavian"
PATHS = S.PATHS
OUT = "/work/subclavian/seg/v6va3"
BOTTLENECK_MIN = 0.5


def one(donor, side):
    d = os.path.join(WALL, donor)
    x = "r" if side == "R" else "l"
    surf = os.path.join(d, "%sjunction_stageA.vtp" % x)
    jl = os.path.join(d, "junction_labels_kji.npz")
    mk = os.path.join(d, ".junction_v3")
    if not (os.path.isfile(surf) and os.path.isfile(jl) and os.path.isfile(mk)):
        return {"donor_id": donor, "side": side, "status": "no v3 junction (%s)" % [os.path.isfile(f) for f in (surf, jl, mk)]}
    pj = json.load(open(os.path.join(PATHS, "%s_%s.json" % (donor, side))))
    P = np.asarray(pj["path"], float)
    s = S.arclen(P)
    s_ost = float(pj["s_ostium_mm"])
    z = np.load(jl, allow_pickle=True)
    m = z["%sjunction" % x].astype(bool)
    org, sp = np.asarray(z["origin_mm"], float), np.asarray(z["spacing_mm"], float)
    edt = ndimage.distance_transform_edt(m, sampling=(sp[2], sp[1], sp[0]))
    ijk = (P - org) / sp
    inside = np.all((ijk >= 0) & (ijk <= np.array(m.shape[::-1]) - 1), axis=1)
    r = np.full(len(P), np.nan)
    lab_at = np.zeros(len(P), bool)
    if inside.any():
        q = ijk[inside]
        r[inside] = ndimage.map_coordinates(edt, [q[:, 2], q[:, 1], q[:, 0]], order=1, mode="nearest")
        iq = np.round(q).astype(int)
        lab_at[np.nonzero(inside)[0]] = m[iq[:, 2], iq[:, 1], iq[:, 0]]
    cov = s[lab_at & (r > 0.5)]
    misr = [[float(a), float(b)] for a, b in zip(s[::2], r[::2]) if np.isfinite(b) and b > 0.5]
    gi, mesh = S.genus_closed(surf)
    from scipy.spatial import cKDTree
    pts = np.asarray(mesh.points, float)
    dd, ix = cKDTree(P).query(pts)
    s_pt = s[ix]
    lo_s = float(cov.min()) if len(cov) else 0.0
    hi_s = s_ost + 20.0
    sel = (s_pt >= lo_s) & (s_pt <= hi_s) & (np.abs(s_pt - s_ost) > 6.0)
    detail = None
    if sel.sum() > 50:
        rr = np.full(len(P), np.nan)
        for k in range(len(P)):
            w = (np.abs(s_pt - s[k]) <= 5.0) & sel
            if w.sum() > 10:
                rr[k] = np.median(dd[w])
        dev = np.abs(dd[sel] - rr[ix[sel]])
        dev = dev[np.isfinite(dev)]
        detail = {"share_gt_0.3mm": float((dev > 0.3).mean()), "n_points": int(len(dev))}
    qc = json.load(open(os.path.join(d, "seg_qc.json")))
    jb = ((qc.get("v6") or {}).get("junction_bridge") or {}).get("%sjunction" % x) or {}
    aft = jb.get("after") or {}
    failed = None
    # a tube bridge (the wall agent's flagged fallback) is a few mm of tube in the ostium crotch: used, flagged
    if aft.get("bottleneck_ratio") is not None and aft["bottleneck_ratio"] < BOTTLENECK_MIN:
        failed = "junction bottleneck %.2f mm = %.2f of the VA's inscribed radius (< %.1f)" % (
            aft.get("bottleneck_r_mm") or -1, aft["bottleneck_ratio"], BOTTLENECK_MIN)
    out = {"donor_id": donor, "side": side, "status": "ok", "surface": surf, "junction_labels": jl,
           "marker": open(mk).read().strip(), "surface_rule": "v6_wall jfix/junction6.py v3 (clean_sub + CT-lumen bridge)",
           "junction_bridge": jb, "junction_failed": failed, "ostial_tube_bridge": bool(aft.get("fallback_tube")), "O_v": [float(np.interp(s_ost, s, P[:, i])) for i in range(3)],
           "s_ostium_mm": s_ost, "surface_info": gi,
           "covered_s_mm": [float(cov.min()), float(cov.max())] if len(cov) else None,
           "want_s_mm": [float((pj.get("ts_origin") or {}).get("s_on_path_mm", 12.0) or 12.0) + 10.0, s_ost + 20.0],
           "misr_path": misr, "misr_median_stretch": float(np.nanmedian(r[(s >= lo_s) & (s <= hi_s)])) if len(cov) else None,
           "wall_detail_proxy": detail, "frame": "donor RAS mm (unmirrored); graft_v6_va mirrors",
           "mtimes": {f: time.strftime("%F %T", time.localtime(os.path.getmtime(f))) for f in (surf, jl)},
           "built": time.strftime("%F %T")}
    od = os.path.join(OUT, donor, side)
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "junction.json"), "w"), indent=1)
    return out


def main():
    donors = sys.argv[1:] or sorted(os.path.basename(x) for x in glob.glob(os.path.join(WALL, "*"))
                                    if os.path.isdir(x) and os.path.isfile(os.path.join(x, ".junction_v3")))
    rows = {}
    for dn in donors:
        for side in ("R", "L"):
            try:
                o = one(dn, side)
            except Exception as e:  # noqa: BLE001
                import traceback
                print("ERROR", dn, side, type(e).__name__, e, traceback.format_exc()[-400:], flush=True)
                continue
            if o.get("status") != "ok":
                print(dn, side, o.get("status"), flush=True)
                rows["%s_%s" % (dn, side)] = o
                continue
            a = (o["junction_bridge"].get("after") or {})
            print("%-18s %s cov %s want %s misr %.2f genus %s open %d detail %s | bridge ratio %s fallback %s | FAILED %s" % (
                dn, side, o["covered_s_mm"] and [round(v, 1) for v in o["covered_s_mm"]], [round(v, 1) for v in o["want_s_mm"]],
                o["misr_median_stretch"] or -1, o["surface_info"]["genus_largest"], o["surface_info"]["open_or_nonmanifold_edges"],
                o["wall_detail_proxy"] and round(o["wall_detail_proxy"]["share_gt_0.3mm"], 3), a.get("bottleneck_ratio"),
                a.get("fallback_tube"), o["junction_failed"]), flush=True)
            rows["%s_%s" % (dn, side)] = {k: o[k] for k in ("covered_s_mm", "want_s_mm", "misr_median_stretch", "surface_info",
                                                           "wall_detail_proxy", "junction_failed", "surface")}
            rows["%s_%s" % (dn, side)]["bridge_after"] = a
    if not sys.argv[1:]:
        json.dump(rows, open(os.path.join(OUT, "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
