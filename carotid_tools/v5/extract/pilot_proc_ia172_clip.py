#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_clip.py (= /sp/v5_va/pilot_proc_ia172_clip.py in the build containers)
"""pilot_proc_ia172_clip.py <case> -- the Arterial input of a 172-series pilot case: the native ct.nii.gz with HU clipped
to [-1024, 3071] (neve-build-stagea).

Why: the 172-series CTs hold -2048 outside the scan circle (21-31 % of voxels) and dental streaks down to -5530 HU
(pilot_fetch_ia172.md section 3). Arterial's extracranial nnU-Net model normalises with ZScoreNormalization
(plans.json: a per-image z-score) and Arterial's code has no HU clip, so the padding would shift its normalisation.
TotalSegmentator's models use CTNormalization (a fixed clip to training percentiles, lower bound >= -1024), for which this
clip is a no-op, so the TotalSegmentator masks were made from the raw CT. Same grid, affine, qform/sform and dtype (int16);
only values change. -> <der>/arterial/ct_clip.nii.gz (+ ct_clip.json). Deleted after Arterial by the driver.
"""
import json
import os
import sys
import time

import numpy as np
import nibabel as nib

case = sys.argv[1]
SRC = "/repo/carotid_data/va_sources/thin_slice/ia172/%s/ct.nii.gz" % case
OUT = "/repo/carotid_data/va_sources/derived/thin_slice/ia172/%s/arterial" % case
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
im = nib.load(SRC)
a = np.asanyarray(im.dataobj)
lo, hi = int((a < -1024).sum()), int((a > 3071).sum())
c = np.clip(a, -1024, 3071).astype(np.int16)
n = nib.Nifti1Image(c, im.affine, im.header)
n.set_data_dtype(np.int16)
n.header["scl_slope"] = 1.0
n.header["scl_inter"] = 0.0
tmp = os.path.join(OUT, "ct_clip.tmp.nii.gz")
nib.save(n, tmp)
os.replace(tmp, os.path.join(OUT, "ct_clip.nii.gz"))
rec = {"case": case, "source": SRC, "clip_HU": [-1024, 3071], "n_below": lo, "n_above": hi, "n_total": int(a.size),
       "frac_below": round(lo / a.size, 4), "shape": list(a.shape), "affine": im.affine.round(4).tolist(),
       "seconds": round(time.time() - t0, 1)}
json.dump(rec, open(os.path.join(OUT, "ct_clip.json"), "w"), indent=1)
print(json.dumps(rec))
