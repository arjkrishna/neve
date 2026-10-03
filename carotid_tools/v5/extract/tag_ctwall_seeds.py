#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_ctwall_seeds.py (= /sp/v5_va/tag_ctwall_seeds.py in the build containers)
"""tag_ctwall_seeds.py <tag> -- VMTK seeds for a tag_ct_seg.py run, repaired where the label does not reach an end.

tag_ct_seg.py (= vmr_ct_seg3.py) writes <v>_seeds.json = centroids of the label voxels whose station s lies in the
first / last 1.5 mm of the station line. On TaG-Net A the right VA's ct_seg4 label stops 3.5 mm short of the merge
(the 0.8 mm-rule opening at s 211.75-215, where the thin right V4 (I_c ~ 330 HU) meets the brighter left VA / BA,
cut the end off and "keep the largest piece" dropped it), so the target seed is NaN. Here every seed is re-derived
the same way (voxel -> nearest station of its own vessel, as the script's sid), and a NaN seed is replaced by the
centroid of the label's own first / last 1.5 mm. Non-NaN seeds are kept byte-identical in value (checked).
Writes <v>_seeds.json (backup of the original as <v>_seeds_orig.json) and seeds_fix.json. eve-training-fixed.
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
from tag_ctwall_common import SEG_ROOT, jdump  # noqa: E402

tag = sys.argv[1]
D = os.path.join(SEG_ROOT, tag)
z = np.load(os.path.join(D, "labels_kji.npz"))
L, org, sp = z["labels"], z["origin_mm"], z["spacing_mm"]
st = json.load(open(os.path.join(D, "stations_rays.json")))
out = {}
for k, v in ((1, "rva"), (2, "lva"), (3, "ba")):
    P, s = np.asarray(st[v]["P"]), np.asarray(st[v]["s"])
    idx = np.argwhere(L == k)
    W = org + idx[:, ::-1] * sp
    _, j = cKDTree(P).query(W)
    sv = s[j]
    a = W[sv <= 1.5].mean(0) if (sv <= 1.5).any() else None
    b = W[sv >= s[-1] - 1.5].mean(0) if (sv >= s[-1] - 1.5).any() else None
    p = os.path.join(D, "%s_seeds.json" % v)
    old = json.load(open(p))
    rec = {"label_s_range_mm": [float(sv.min()), float(sv.max())], "station_length_mm": float(s[-1]),
           "missing_at_start_mm": float(sv.min()), "missing_at_end_mm": float(s[-1] - sv.max())}
    new = dict(old)
    changed = False
    for key, val, lo, hi in (("source", a, sv.min(), sv.min() + 1.5), ("target", b, sv.max() - 1.5, sv.max())):
        o = np.asarray(old[key], float)
        if np.all(np.isfinite(o)):
            assert val is not None and np.allclose(o, val, atol=1e-6), (v, key, o, val)
            continue
        m = (sv >= lo) & (sv <= hi)
        new[key] = W[m].mean(0).tolist()
        rec["%s_replaced" % key] = {"by": "centroid of the label's own %s 1.5 mm (s %.2f-%.2f)" % (
            "first" if key == "source" else "last", lo, hi), "point": new[key]}
        changed = True
    if changed:
        bk = os.path.join(D, "%s_seeds_orig.json" % v)
        if not os.path.exists(bk):
            shutil.copy(p, bk)
        new["note"] = old["note"] + "; repaired by tag_ctwall_seeds.py (label does not reach the station end)"
        jdump(new, p)
    out[v] = rec
    print(v, json.dumps(rec), flush=True)
jdump(out, os.path.join(D, "seeds_fix.json"))
