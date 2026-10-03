# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_archcut.py (= /sp/v5_va/scaleup_ia172_archcut.py in the build containers)
"""scaleup_ia172_archcut = pilot_ia172_archcut.py (copied 2026-10-02, scaleup_lane_ia172) with ONE change: output
/sp/v5_va/results/scaleup_ia172_archcut.json (the post-fetch confirmation of the pre-fetch check scaleup_ia172_prearch.py).

pilot_ia172_archcut: does the bottom slice cut the aortic arch into two limbs (ascending + descending)?
Arterial's start point needs that (donor_pool_plan 1.2: head CTAs whose first slice is above the arch top fail).
Native CT: in each of the lowest slices (0, 2, 5, 10, 20 mm up), count connected components of > 150 HU with area
>= 150 mm^2 that overlap the dilated quick-look aorta mask (or lie within 25 mm of it); report their areas and the
centre distance of the two largest. -> /sp/v5_va/results/pilot_ia172_archcut.json
usage: python3 pilot_ia172_archcut.py <case> ..."""
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
DER = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
OUT = "/sp/v5_va/results/scaleup_ia172_archcut.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for case in sys.argv[1:]:
    im = nib.load(os.path.join(IMG, case, "ct.nii.gz"))
    sp = im.header.get_zooms()[:3]
    ts_im = nib.load(os.path.join(DER, case, "ts_fast_1p5.nii.gz"))
    ts = np.asarray(ts_im.dataobj)
    ao = ndimage.binary_dilation(ts == 52, iterations=8)  # ~12 mm
    M = np.linalg.inv(ts_im.affine) @ im.affine
    rec = {}
    for dz in (0, 2, 5, 10, 20):
        k = int(round(dz / sp[2]))
        sl = np.asarray(im.dataobj[:, :, k]).astype(np.float32)
        lab, n = ndimage.label(sl > 150)
        if n == 0:
            rec["%dmm" % dz] = []; continue
        area = np.bincount(lab.ravel())[1:] * sp[0] * sp[1]
        comps = []
        for i in np.where(area >= 150)[0]:
            ij = np.argwhere(lab == i + 1)
            c = ij.mean(axis=0)
            p = M @ np.array([c[0], c[1], k, 1.0])
            q = np.clip(np.round(p[:3]).astype(int), 0, np.array(ts.shape) - 1)
            if ao[q[0], q[1], q[2]]:
                # roundness: area vs bounding-circle area (bone/heart blobs are irregular)
                r = np.sqrt(((ij - c) ** 2).sum(axis=1)).max() * sp[0]
                comps.append({"area_mm2": round(float(area[i]), 0), "centre_mm": [round(float(c[0] * sp[0]), 1),
                              round(float(c[1] * sp[1]), 1)], "fill_of_circle": round(float(area[i] / (np.pi * r * r)), 2)})
        comps.sort(key=lambda d: -d["area_mm2"])
        rec["%dmm" % dz] = comps[:4]
    b = rec.get("0mm", [])
    rec["n_aortic_components_bottom"] = len(b)
    if len(b) >= 2:
        d = np.hypot(b[0]["centre_mm"][0] - b[1]["centre_mm"][0], b[0]["centre_mm"][1] - b[1]["centre_mm"][1])
        rec["bottom_two_largest_centre_dist_mm"] = round(float(d), 1)
    res[case] = rec
    print(case, rec["n_aortic_components_bottom"], json.dumps({k: [(c["area_mm2"], c["fill_of_circle"]) for c in v]
                                                                for k, v in rec.items() if k[:-2].isdigit()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
