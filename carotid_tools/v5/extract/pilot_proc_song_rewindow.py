#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_rewindow.py (= /sp/v5_va/pilot_proc_song_rewindow.py in the build containers)
"""pilot_proc_song_rewindow.py <case> -- one more re-windowed Song CT for Arterial (neve-build-stagea; pilot_proc_song).

Writes derived/thin_slice/song/<case>/ct_negzero_air1000.nii.gz (int16, same grid / affine as the shipped ct.nii.gz):
  = the fetch agent's ct_negzero-100 (-100 HU on every originally negative voxel = air, fat, lung; 0 on 0..50 HU soft
    tissue; the shipped value elsewhere) with the voxels OUTSIDE the fetch agent's body_mask.nii.gz set to -1000 HU.
Why: Arterial's extracranial nnU-Net z-scores the WHOLE image (ZScoreNormalization, no mask). On AHMU1218100 the runs on
the shipped image (82 % zeros) and on ct_bg-100 (outside-body zeros -> -100) both lost every neck vessel below the chin
(TS-CCA slices with a mask component <= 0.07). This variant restores the two contrasts a real CTA has around the neck
vessels (fat -100 vs soft tissue 0) and the air background (-1000) that dominates the whole-image mean / std.
The values are a re-windowing of the shipped data, not a recovery of the lost HU (0..50 HU tissue stays 0, fat is a flat
-100, air inside the body stays -100).
"""
import json
import os
import sys

import numpy as np
import nibabel as nib

case = sys.argv[1]
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % case
nz = nib.load(os.path.join(DER, "ct_negzero-100.nii.gz"))
a = np.asanyarray(nz.dataobj).astype(np.int16)
bm = np.asanyarray(nib.load(os.path.join(DER, "body_mask.nii.gz")).dataobj) > 0
assert bm.shape == a.shape
out = a.copy()
out[~bm] = -1000
im = nib.Nifti1Image(out, nz.affine, nz.header)
im.set_data_dtype(np.int16)
im.header["scl_slope"], im.header["scl_inter"] = 1, 0
nib.save(im, os.path.join(DER, "ct_negzero_air1000.nii.gz"))
rec = {"case": case, "file": "ct_negzero_air1000.nii.gz", "frac_minus1000": round(float((out == -1000).mean()), 4),
       "frac_minus100": round(float((out == -100).mean()), 4), "frac_0": round(float((out == 0).mean()), 4),
       "outside_body_nonzero_before": int(((a != -100) & (a != 0) & ~bm).sum())}
json.dump(rec, open(os.path.join(DER, "ct_negzero_air1000.json"), "w"), indent=1)
print(json.dumps(rec))
