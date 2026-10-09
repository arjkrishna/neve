#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\sub_genus.py (= /sp/v6/wall/jfix/sub_genus.py in the build containers)
"""v6_wall (e): where do the subclavian labels' handles come from? Genus of the stage-A surface of the subclavian label
raw, after a 1-voxel opening (thin bridges to neighbours), after a 1-voxel closing (holes / slits in the wall), and
both; voxel change. Read-only on the labels (labels_kji.npz of the request runs).
    python3 sub_genus.py <unit:side,...|all>  -> /v6w/subclavian_j2/sub_genus.json"""
import glob
import json
import sys

import numpy as np
from scipy import ndimage
from skimage import measure

sys.path.insert(0, "/sp/v6/wall/jfix")
import junction6 as J6  # noqa: E402

arg = sys.argv[1]
if arg == "all":
    todo = [(f.split("/")[-2], s) for f in sorted(glob.glob("/v6w/subclavian/*/labels_kji.npz")) for s in ("R", "L")]
else:
    todo = [tuple(x.split(":")) for x in arg.split(",")]
ball = ndimage.generate_binary_structure(3, 1)
out = {}
for u, sd in todo:
    z = np.load("/v6w/subclavian/%s/labels_kji.npz" % u)
    L, sp, org = z["labels"], z["spacing_mm"], z["origin_mm"]
    m0 = L == (4 if sd == "R" else 5)
    if not m0.any():
        continue

    def genus(m):
        lab, n = ndimage.label(m)
        if n > 1:
            m = lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
        vol = np.pad(m.transpose(2, 1, 0).astype(np.uint8), 1)
        v, f, _, _ = measure.marching_cubes(vol, level=0.5, spacing=tuple(sp))
        return J6.genus_of(v, f)["genus_largest"], m
    g0, k0 = genus(m0)
    mo = ndimage.binary_opening(m0, structure=ball)
    go, ko = genus(mo)
    mc = ndimage.binary_closing(m0, structure=ball)
    gc, kc = genus(mc)
    moc = ndimage.binary_closing(mo, structure=ball)
    goc, koc = genus(moc)
    n0 = int(k0.sum())
    out["%s:%s" % (u, sd)] = {"raw": g0, "open1": go, "close1": gc, "open1_close1": goc, "voxels": n0,
                              "open_change": round(float((k0 ^ ko).sum()) / n0, 4),
                              "close_change": round(float((k0 ^ kc).sum()) / n0, 4),
                              "open_close_change": round(float((k0 ^ koc).sum()) / n0, 4)}
    print(u, sd, out["%s:%s" % (u, sd)], flush=True)
json.dump(out, open("/v6w/subclavian_j2/sub_genus.json", "w"), indent=1)
