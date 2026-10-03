#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_totalseg_make1p5.py (= /sp/v5_va/tag_totalseg_make1p5.py in the build containers)
"""tag_totalseg_make1p5.py <fine_ts_dir> <out_ts_dir> -- a 1.5 mm isotropic RAS copy of a 0.4 mm ts dir, laid out like a
TotalSegmentator v3 dataset subject (1.5 mm, RAS, int16 CT) so that art_extract's load_ts / ts_qc run in ~1 GB instead
of ~18 GB. neve-build-stagea.

Grid: RAS, 1.5 mm, voxel 0 at the RAS-canonical first voxel centre of the CTA, n = floor((N - 1) * sp / 1.5) + 1 per axis,
affine consistent with world (no corner-aligned zoom). Masks: every <class>.nii.gz sampled at the 1.5 mm voxel centres
by nearest neighbour (the 0.4 mm masks are TotalSegmentator's own 1.5 mm predictions written back by nearest neighbour,
so this recovers them up to +-0.2 mm). CT: Gaussian pre-filter (sigma = sqrt(1.5^2 - sp^2) / 2.355 mm per axis) then
trilinear at the voxel centres, rounded to int16 (used by load_ts only for the HU of bridged gaps).

Why not run TotalSegmentator on a 1.5 mm file made with its own resampling.change_spacing: that function resizes with
ndimage.zoom (corner-aligned) but writes an affine scaled about voxel 0, so such a file is misregistered by up to one
voxel at the far edge (1.5 mm at the top of patient A's scan; Dice 0.78-0.95 against the 0.4 mm masks when tried).
Inside TotalSegmentator the down / up round trip cancels, so the 0.4 mm masks are correctly placed.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, inv_ornt_aff, apply_orientation
from scipy import ndimage

fine, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "segmentations"), exist_ok=True)
ct = nib.load(os.path.join(fine, "ct.nii.gz"))
A0 = ct.affine
x = ornt_transform(io_orientation(A0), axcodes2ornt(("R", "A", "S")))
data = apply_orientation(np.asanyarray(ct.dataobj), x)
Ar = A0 @ inv_ornt_aff(x, data.shape)                      # RAS-canonical affine of the fine grid
sp = np.abs(np.diag(Ar)[:3])
n = [int(np.floor((N - 1) * s / 1.5)) + 1 for N, s in zip(data.shape, sp)]
A1 = np.diag([1.5, 1.5, 1.5, 1.0]); A1[:3, 3] = Ar[:3, 3]
I = np.indices(n).reshape(3, -1).astype(np.float32)
W = A1 @ np.r_[I, np.ones((1, I.shape[1]), np.float32)]
J0 = (np.linalg.inv(A0) @ W)[:3]                            # coordinates in the ORIGINAL fine grid
sig = [np.sqrt(max(1.5 ** 2 - s ** 2, 0)) / 2.355 / s for s in np.abs(np.sqrt((A0[:3, :3] ** 2).sum(0)))]
v = ndimage.gaussian_filter(np.asanyarray(ct.dataobj).astype(np.float32), sig)
c = ndimage.map_coordinates(v, J0, order=1, mode="nearest").reshape(n)
del v


def save(arr, path):
    im = nib.Nifti1Image(arr, A1)
    im.set_qform(A1, code=1); im.set_sform(A1, code=1)
    nib.save(im, path)


save(np.clip(np.rint(c), -32768, 32767).astype(np.int16), os.path.join(out, "ct.nii.gz"))
names = sorted(f for f in os.listdir(os.path.join(fine, "segmentations")) if f.endswith(".nii.gz"))
vox = {}
for f in names:
    m = np.asanyarray(nib.load(os.path.join(fine, "segmentations", f)).dataobj).astype(np.uint8)
    s = ndimage.map_coordinates(m, J0, order=0, mode="constant", cval=0).reshape(n).astype(np.uint8)
    save(s, os.path.join(out, "segmentations", f))
    vox[f[:-7]] = int(s.sum())
info = {"from": fine, "shape": n, "affine": A1.tolist(), "axcodes": "RAS", "ct_prefilter_sigma_vox": [round(x_, 3) for x_ in sig],
        "n_masks": len(names), "voxels": {k: vox[k] for k in ("vertebrae_C1", "subclavian_artery_right", "aorta") if k in vox},
        "note": "derived from TaG-Net patient A (no data licence): internal only"}
json.dump(info, open(os.path.join(out, "make1p5_provenance.json"), "w"), indent=1)
print(json.dumps(info))
