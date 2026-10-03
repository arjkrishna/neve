#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_tsprep.py (= /sp/v5_va/pilot_proc_song_tsprep.py in the build containers)
"""pilot_proc_song_tsprep.py <case> <input> -- the TotalSegmentator input of a Song pilot case (neve-build-stagea).

<input> = shipped | negzero | bg
  shipped : thin_slice/song/<case>/ct.nii.gz as shipped (float64, every voxel <= 50 HU stored as 0)
  negzero : derived/thin_slice/song/<case>/ct_negzero-100.nii.gz (the fetch agent's side file: -100 HU on the -0.0 voxels,
            i.e. on every originally negative voxel; equal to the shipped image elsewhere)
  bg      : derived/thin_slice/song/<case>/ct_bg-100.nii.gz (-100 HU on every 0 outside the body mask)
Writes derived/thin_slice/song/<case>/<tsdir>/ct_ts_input.nii.gz as int16 (same grid / affine, exact: the values are integers;
checked) for the GPU container, (the driver makes <tsdir>/ct.nii.gz = a host hard link to the SHIPPED ct.nii.gz, (what arterial_ts_args and
tag_totalseg_make1p5 read; the grid is identical). <tsdir> = TS_DIRNAME env (default totalseg).
Why an int16 copy: the shipped files are float64 (1.6 GB in RAM per case); int16 is exact here and 4x smaller.
pilot_proc_song, 2026-10-01.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib

case, which = sys.argv[1], sys.argv[2]
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % case
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % case
TSD = os.path.join(DER, os.environ.get("TS_DIRNAME", "totalseg"))
os.makedirs(TSD, exist_ok=True)
p = {"shipped": os.path.join(SRC, "ct.nii.gz"), "negzero": os.path.join(DER, "ct_negzero-100.nii.gz"),
     "bg": os.path.join(DER, "ct_bg-100.nii.gz")}[which]
im = nib.load(p)
a = np.asanyarray(im.dataobj)
b = np.rint(a).astype(np.int16)
assert np.array_equal(b.astype(a.dtype), a) or np.abs(b - a).max() == 0, "non-integer values"
out = nib.Nifti1Image(b, im.affine, im.header)
out.set_data_dtype(np.int16)
out.header["scl_slope"], out.header["scl_inter"] = 1, 0
nib.save(out, os.path.join(TSD, "ct_ts_input.nii.gz"))
rec = {"case": case, "ts_input": which, "source": p, "dtype_in": str(a.dtype), "written": "int16 exact",
       "min": float(b.min()), "max": float(b.max()), "frac_m100": round(float((b == -100).mean()), 4)}
json.dump(rec, open(os.path.join(TSD, "ts_input.json"), "w"), indent=1)
print(json.dumps(rec))
