#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_ctseg_compare.py (= /sp/v5_va/scaleup_fixes_ctseg_compare.py in the build containers)
"""scaleup_fixes_ctseg_compare.py -- (scaleup_fixes, 2026-10-02) compare every pilot ct_seg4 wall with its pipeline-v2 re-run
(scaleup_ct_seg.py, tag ct_seg4v2; same seeds = the pilot's selected traces): labels_kji.npz voxel-for-voxel per VA label,
the label s-range (seeds_fix), and the pinch-fix events (seg_qc.json pinch_fix). A wall where the fix did not trigger must
be IDENTICAL. Writes results/scaleup_fixes_ctseg_compare.json. Read-only on the walls. neve-build-stagea (numpy)."""
import glob
import json
import os

import numpy as np

TH = "/repo/carotid_data/va_sources/derived/thin_slice"
OUT = "/sp/v5_va/results/scaleup_fixes_ctseg_compare.json"
rows = []
for src in ("ia172", "song"):
    for d4 in sorted(glob.glob(os.path.join(TH, src, "*", "ct_seg", "ct_seg4"))):
        case = d4.split("/")[-3]
        d2 = os.path.join(os.path.dirname(d4), "ct_seg4v2")
        if not os.path.isfile(os.path.join(d2, "seg_qc.json")) or not os.path.isfile(os.path.join(d4, "labels_kji.npz")):
            continue
        a, b = np.load(os.path.join(d4, "labels_kji.npz")), np.load(os.path.join(d2, "labels_kji.npz"))
        q4, q2 = json.load(open(os.path.join(d4, "seg_qc.json"))), json.load(open(os.path.join(d2, "seg_qc.json")))
        A, B = a["labels"], b["labels"]
        same_grid = A.shape == B.shape and np.allclose(a["origin_mm"], b["origin_mm"]) and np.allclose(a["spacing_mm"], b["spacing_mm"])
        if not same_grid and np.allclose(a["spacing_mm"], b["spacing_mm"]):
            # both crops lie on the same CT grid (moved stations widen the crop): embed both in their union box
            spv = np.asarray(a["spacing_mm"], float)
            oa, ob = np.asarray(a["origin_mm"], float), np.asarray(b["origin_mm"], float)
            d = (ob - oa) / spv
            if np.allclose(d, np.round(d), atol=1e-3):
                d = np.round(d).astype(int)[::-1]                     # [k, j, i] offset of B relative to A
                lo = np.minimum(0, d)
                hi = np.maximum(np.array(A.shape), d + np.array(B.shape))
                A2 = np.zeros(tuple(hi - lo), A.dtype); B2 = np.zeros(tuple(hi - lo), B.dtype)
                sa = tuple(slice(-lo[i], -lo[i] + A.shape[i]) for i in range(3))
                sb = tuple(slice(d[i] - lo[i], d[i] - lo[i] + B.shape[i]) for i in range(3))
                A2[sa] = A; B2[sb] = B
                A, B = A2, B2
                b = {"origin_mm": oa + lo[::-1] * spv, "spacing_mm": spv, "labels": B, "present": b["present"]}
                same_grid = True
        ev = (q2.get("pinch_fix") or {}).get("events") or []
        for v, k in (("rva", 1), ("lva", 2)):
            if v not in list(a["present"]) and v not in list(b["present"]):
                continue
            evv = [e for e in ev if e.get("vessel") == v]
            r = {"source": src, "case": case, "vessel": v, "side": "R" if v == "rva" else "L", "same_grid": bool(same_grid),
                 "pinch_triggered": bool(evv),
                 "pinch_events": [{kk: e.get(kk) for kk in ("pass", "stations_flagged", "flagged_s_runs_mm", "moved",
                                                            "move_mm_max", "flagged_after_recast")} for e in evv],
                 "label_s_range_pilot": (q4.get("seeds_fix") or {}).get(v, {}).get("label_s_range_mm"),
                 "station_length_pilot": (q4.get("seeds_fix") or {}).get(v, {}).get("station_length_mm"),
                 "label_s_range_v2": (q2.get("seeds_fix") or {}).get(v, {}).get("label_s_range_mm"),
                 "station_length_v2": (q2.get("seeds_fix") or {}).get(v, {}).get("station_length_mm")}
            if same_grid:
                ma, mb = A == k, B == k
                r.update({"voxels_pilot": int(ma.sum()), "voxels_v2": int(mb.sum()), "identical": bool(np.array_equal(ma, mb)),
                          "only_pilot": int((ma & ~mb).sum()), "only_v2": int((mb & ~ma).sum()),
                          "dice": round(float(2 * (ma & mb).sum() / max(ma.sum() + mb.sum(), 1)), 5)})
                if not r["identical"]:
                    # where along the VA the labels differ: differing voxels -> nearest v2 station s, 10 mm bins
                    from scipy.spatial import cKDTree
                    st = json.load(open(os.path.join(d2, "stations_rays.json")))[v]
                    Ps, ss = np.asarray(st["P"], float), np.asarray(st["s"], float)
                    dif = np.argwhere(ma ^ mb)
                    W = np.asarray(b["origin_mm"], float) + dif[:, ::-1] * np.asarray(b["spacing_mm"], float)
                    sv = ss[cKDTree(Ps).query(W)[1]]
                    h = np.bincount((sv // 10).astype(int))
                    r["differing_voxels_per_10mm_bin"] = {"%d-%d" % (10 * i, 10 * i + 10): int(n) for i, n in enumerate(h) if n >= 20}
            else:
                r.update({"voxels_pilot": int((A == k).sum()), "voxels_v2": int((B == k).sum()), "identical": False})
            rows.append(r)
summary = {"n_walls": len(rows), "n_triggered": sum(r["pinch_triggered"] for r in rows),
           "n_not_triggered": sum(not r["pinch_triggered"] for r in rows),
           "not_triggered_identical": sum((not r["pinch_triggered"]) and r.get("identical", False) for r in rows),
           "not_triggered_but_different": [(r["case"], r["side"]) for r in rows if not r["pinch_triggered"] and not r.get("identical")],
           "triggered": [(r["case"], r["side"], r.get("dice"), r["label_s_range_pilot"], r["label_s_range_v2"]) for r in rows if r["pinch_triggered"]]}
json.dump({"what": __doc__, "summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    print("%s %s %s trig %s ident %s dice %s vox %s->%s s %s->%s diff-bins %s" % (r["source"], r["case"], r["side"], r["pinch_triggered"],
          r.get("identical"), r.get("dice"), r.get("voxels_pilot"), r.get("voxels_v2"), r["label_s_range_pilot"], r["label_s_range_v2"], r.get("differing_voxels_per_10mm_bin")))
