#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_archtop.py (= /sp/v5_va/scaleup_song_archtop.py in the build containers)
"""scaleup_song_archtop.py CASE [CASE ...] (scaleup_lane_song, 2026-10-02; read-only; neve-build-stagea)
Arch top from the TotalSegmentator total_v3 aorta mask (derived/.../totalseg/segmentations/aorta.nii.gz, CTA grid):
the highest RAS z with >= 20 aorta voxels, the z of the image bottom / top, and the qc_pass1 automatic arch top for
comparison. Used to set ARCH_TOP_Z for the fetch QC crops when the automatic aorta track fails (as the pilot did by eye
on AHMU1218113). (04:25) + the median shipped-CT HU inside the 1-voxel-eroded aorta mask over its top 60 mm (the
AORTA_HU fallback of scaleup_song_fetch_qc.py when the track finds no limb)."""
import json, sys
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
for c in sys.argv[1:]:
    p = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s/totalseg/segmentations/aorta.nii.gz" % c
    q = json.load(open("/repo/carotid_data/va_sources/thin_slice/song/%s/qc_pass1.json" % c))
    im = nib.as_closest_canonical(nib.load(p))
    a = np.asanyarray(im.dataobj) > 0
    A = im.affine; nz = a.shape[2]
    zmm = A[2, 3] + np.arange(nz) * A[2, 2]
    cnt = a.sum(axis=(0, 1))
    ks = np.where(cnt >= 20)[0]
    top = float(zmm[ks.max()]) if len(ks) else None
    hu = None
    if top is not None:
        ct = nib.as_closest_canonical(nib.load("/repo/carotid_data/va_sources/thin_slice/song/%s/ct.nii.gz" % c))
        assert np.allclose(ct.affine, A, atol=1e-3)
        sel = ndi.binary_erosion(a) & (zmm[None, None, :] >= top - 60.0)
        v = np.asanyarray(ct.dataobj)[sel]
        hu = round(float(np.median(v)), 1) if v.size else None
    print(json.dumps({"case": c, "ts_aorta_top_z": None if top is None else round(top, 1),
                      "ts_aorta_hu_median_top60": hu,
                      "image_z": [round(float(zmm[0]), 1), round(float(zmm[-1]), 1)],
                      "aorta_voxels_bottom_slice": int(cnt[0]),
                      "qc_pass1_arch_top_z": q["aorta_auto"].get("arch_top_z"),
                      "qc_pass1_aorta_hu": q["aorta_auto"].get("aorta_hu_med"),
                      "qc_pass1_above_bottom_mm": q["aorta_auto"].get("arch_top_above_bottom_mm")}))
