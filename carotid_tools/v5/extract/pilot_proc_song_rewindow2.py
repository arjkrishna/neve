#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_rewindow2.py (= /sp/v5_va/pilot_proc_song_rewindow2.py in the build containers)
"""pilot_proc_song_rewindow2.py <case> -- a TEXTURED re-windowing of a Song CT, ONLY as an Arterial input (neve-build-stagea;
pilot_proc_song, 2026-10-01 22:25). Writes derived/thin_slice/song/<case>/ct_tex.nii.gz (int16, same grid / affine):
  shipped value where it is > 0 (every voxel > 50 HU: contrast, bone, enhanced tissue -- unchanged);
  +0.0 voxels (originally 0..50 HU: soft tissue, CSF)        -> 40 HU + N(0, 12) noise;
  -0.0 voxels inside the body mask (fat, airway, lung)         -> -100 HU + N(0, 12) noise;
  voxels outside the fetch agent's body_mask                   -> -1000 HU.
  Noise: numpy default_rng(seed 20261001), Gaussian, then a 0.6-voxel Gaussian blur of the noise field only (CT noise is
  spatially correlated); values of the filled voxels are clipped to their class ranges (soft tissue 1..50, fat -150..-50).
Why: on AHMU1218100 Arterial's extracranial (neck) model returned no CCA and only VA fragments below the mandible on the
shipped image, on ct_bg-100 and on ct_negzero-100 (artcheck: TS-CCA slices with a mask component <= 0.48), while the head
model segmented normally. The difference left between these inputs and a real CTA is the flat, noise-free 0 / -100
background. This file is used ONLY to obtain Arterial's binary vessel mask (the trace / course); every HU-based
measurement (QC HU, ct_seg4 wall, edge FWHM) stays on the shipped image. The filled values are synthetic.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

case = sys.argv[1]
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % case
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % case
im = nib.load(os.path.join(SRC, "ct.nii.gz"))
a = np.asanyarray(im.dataobj)
bm = np.asanyarray(nib.load(os.path.join(DER, "body_mask.nii.gz")).dataobj) > 0
pos0 = (a == 0) & ~np.signbit(a)
neg0 = (a == 0) & np.signbit(a)
rng = np.random.default_rng(20261001)
nz = rng.standard_normal(a.shape).astype(np.float32)
nz = ndimage.gaussian_filter(nz, 0.6)
nz *= 12.0 / max(float(nz.std()), 1e-6)
out = np.rint(a).astype(np.float32)
st = np.clip(40.0 + nz, 1, 50)
ft = np.clip(-100.0 + nz, -150, -50)
out[pos0] = st[pos0]
out[neg0] = ft[neg0]
out[~bm] = -1000.0
out = np.rint(out).astype(np.int16)
o = nib.Nifti1Image(out, im.affine)
o.set_data_dtype(np.int16)
o.header["scl_slope"], o.header["scl_inter"] = 1, 0
nib.save(o, os.path.join(DER, "ct_tex.nii.gz"))
rec = {"case": case, "file": "ct_tex.nii.gz", "frac_pos0": round(float(pos0.mean()), 4), "frac_neg0": round(float(neg0.mean()), 4),
       "frac_outside_body": round(float((~bm).mean()), 4), "unchanged_gt50": round(float((a > 0).mean()), 4),
       "seed": 20261001, "noise_sd_HU": 12.0, "noise_blur_vox": 0.6}
json.dump(rec, open(os.path.join(DER, "ct_tex.json"), "w"), indent=1)
print(json.dumps(rec))
