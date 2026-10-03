#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_ctseg_offside.py (= /sp/v5_va/scaleup_fixes_ctseg_offside.py in the build containers)
"""scaleup_fixes_ctseg_offside.py -- (scaleup_fixes, 2026-10-02) where do the few differing voxels of a NON-triggered VA lie
when the OTHER VA of the case triggered the pinch fix (Song 008 R, 032 R, 100 R)? Prints, per case, the s (along the v2
station polyline) of every differing voxel, its distance to the other VA's label and the labels it moved between.
Read-only. neve-build-stagea."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
TH = "/repo/carotid_data/va_sources/derived/thin_slice"
out = {}
for sc in sys.argv[1:]:
    src, c, side = sc.split("/")
    v, k = ("rva", 1) if side == "R" else ("lva", 2)
    d4 = os.path.join(TH, src, c, "ct_seg", "ct_seg4"); d2 = os.path.join(TH, src, c, "ct_seg", "ct_seg4v2")
    a, b = np.load(d4 + "/labels_kji.npz"), np.load(d2 + "/labels_kji.npz")
    A, B = a["labels"], b["labels"]
    spv = np.asarray(a["spacing_mm"], float); oa, ob = np.asarray(a["origin_mm"], float), np.asarray(b["origin_mm"], float)
    d = np.round((ob - oa) / spv).astype(int)[::-1]
    lo = np.minimum(0, d); hi = np.maximum(np.array(A.shape), d + np.array(B.shape))
    A2 = np.zeros(tuple(hi - lo), A.dtype); B2 = np.zeros(tuple(hi - lo), B.dtype)
    A2[tuple(slice(-lo[i], -lo[i] + A.shape[i]) for i in range(3))] = A
    B2[tuple(slice(d[i] - lo[i], d[i] - lo[i] + B.shape[i]) for i in range(3))] = B
    o = oa + lo[::-1] * spv
    dif = np.argwhere((A2 == k) ^ (B2 == k))
    W = o + dif[:, ::-1] * spv
    st = json.load(open(d2 + "/stations_rays.json"))[v]
    Ps, ss = np.asarray(st["P"], float), np.asarray(st["s"], float)
    sv = ss[cKDTree(Ps).query(W)[1]]
    oth = 3 - k
    Wo = o + np.argwhere(B2 == oth)[:, ::-1] * spv
    do = cKDTree(Wo).query(W)[0] if len(Wo) else np.full(len(W), np.nan)
    trans = {}
    for (kk, jj, ii) in dif:
        key = "%d->%d" % (A2[kk, jj, ii], B2[kk, jj, ii]); trans[key] = trans.get(key, 0) + 1
    out[sc] = {"n_diff": int(len(dif)), "s_mm_min_max": [round(float(sv.min()), 1), round(float(sv.max()), 1)] if len(sv) else None,
               "s_hist_5mm": {"%d" % (5 * i): int(n) for i, n in enumerate(np.bincount((sv // 5).astype(int))) if n},
               "dist_to_other_VA_label_mm_median": round(float(np.median(do)), 1) if len(do) else None,
               "label_transitions_pilot_to_v2": trans, "station_length_mm": float(ss[-1])}
    print(sc, json.dumps(out[sc]))
json.dump(out, open("/sp/v5_va/results/scaleup_fixes_ctseg_offside.json", "w"), indent=1)
