# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_convert.py (= /sp/v5_va/pilot_ia172_convert.py in the build containers)
"""pilot_ia172_convert: one DICOM series -> ct.nii.gz (RAS, raw HU, int16), with geometry checks, plus a 1.5 mm RAS
copy for the quick-look TotalSegmentator run.

usage: python3 pilot_ia172_convert.py <dicom_dir> <out_ct.nii.gz> <out_dir_derived>
  writes <out_ct.nii.gz>, <out_dir_derived>/convert.json and <out_dir_derived>/ct_1p5.nii.gz

Checks (all recorded in convert.json):
  * one SeriesInstanceUID, no LOCALIZER images, one Rows/Columns/PixelSpacing/ImageOrientationPatient;
  * slices sorted by IPP . n (n = row x col direction); uniform spacing (max |dz - median| <= 0.01 mm), no gaps
    (no dz > 1.5 x median), no duplicates (dz < 0.01 mm);
  * gantry tilt / shear: in-plane drift of IPP along the stack (must be < 0.05 mm over the whole stack);
  * oblique: IOP axis-aligned to 1e-4. If oblique or sheared the volume is resampled (linear) onto an axis-aligned RAS
    grid with the same spacing, and that is recorded ("resampled": true). Otherwise voxels are copied 1:1 (only axis
    flips to RAS).
  * HU = stored * RescaleSlope + RescaleIntercept, per slice; stored as int16 when exact.
Container: neve-build-stagea, PYTHONPATH=/sp/va_pool/pylib (pydicom 2.4.4 + pylibjpeg for JPEG lossless).
"""
import glob
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage

t0 = time.time()
src, out_ct, dd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dd, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "*.dcm")), key=lambda p: int(os.path.basename(p).split(".")[0]))
rec = {"dicom_dir": src, "n_files": len(files)}

slices = []
skipped = []
for p in files:
    ds = pydicom.dcmread(p)
    it = [str(x).upper() for x in getattr(ds, "ImageType", [])]
    if "LOCALIZER" in it or not hasattr(ds, "ImagePositionPatient"):
        skipped.append(os.path.basename(p)); continue
    slices.append(ds)
rec["skipped_files"] = skipped


def uniq(f):
    return sorted({json.dumps(f(d)) for d in slices})


checks = {
    "SeriesInstanceUID": uniq(lambda d: str(d.SeriesInstanceUID)),
    "Rows_Columns": uniq(lambda d: [int(d.Rows), int(d.Columns)]),
    "PixelSpacing": uniq(lambda d: [round(float(x), 6) for x in d.PixelSpacing]),
    "ImageOrientationPatient": uniq(lambda d: [round(float(x), 5) for x in d.ImageOrientationPatient]),
    "SliceThickness": uniq(lambda d: round(float(getattr(d, "SliceThickness", 0) or 0), 4)),
    "AcquisitionNumber": uniq(lambda d: int(getattr(d, "AcquisitionNumber", -1) or -1)),
    "ConvolutionKernel": uniq(lambda d: str(getattr(d, "ConvolutionKernel", ""))),
    "RescaleSlope_Intercept": uniq(lambda d: [float(getattr(d, "RescaleSlope", 1)), float(getattr(d, "RescaleIntercept", 0))]),
    "GantryDetectorTilt": uniq(lambda d: float(getattr(d, "GantryDetectorTilt", 0) or 0)),
    "ReconstructionDiameter": uniq(lambda d: float(getattr(d, "ReconstructionDiameter", 0) or 0)),
}
rec["unique_values"] = {k: [json.loads(x) for x in v] for k, v in checks.items()}
for k in ("SeriesInstanceUID", "Rows_Columns", "PixelSpacing", "ImageOrientationPatient"):
    if len(checks[k]) != 1:
        raise SystemExit("inconsistent %s: %s" % (k, checks[k]))

iop = np.array([float(x) for x in slices[0].ImageOrientationPatient])
rdir, cdir = iop[:3], iop[3:]
n = np.cross(rdir, cdir)
ps = [float(x) for x in slices[0].PixelSpacing]  # [row spacing (between rows), column spacing]
ipp = np.array([[float(x) for x in d.ImagePositionPatient] for d in slices])
dproj = ipp @ n
order = np.argsort(dproj)
slices = [slices[i] for i in order]
ipp = ipp[order]; dproj = dproj[order]
dz = np.diff(dproj)
med = float(np.median(dz))
geo = {
    "n_slices": len(slices),
    "dz_median": round(med, 5), "dz_min": round(float(dz.min()), 5), "dz_max": round(float(dz.max()), 5),
    "dz_max_dev": round(float(np.abs(dz - med).max()), 5),
    "n_gaps": int((dz > 1.5 * med).sum()), "n_duplicates": int((dz < 0.01).sum()),
    "uniform": bool(np.abs(dz - med).max() <= 0.01),
    "instance_order_matches_position": None,
}
inst = [int(getattr(d, "InstanceNumber", 0) or 0) for d in slices]
di = np.diff(inst)
geo["instance_order_matches_position"] = bool((di > 0).all() or (di < 0).all())
drift = (ipp - ipp[0]) - np.outer(dproj - dproj[0], n)
geo["inplane_drift_mm"] = round(float(np.linalg.norm(drift, axis=1).max()), 4)
axis_aligned = bool(np.all(np.isclose(np.abs(iop), np.round(np.abs(iop)), atol=1e-4)))
geo["axis_aligned_iop"] = axis_aligned
geo["z_extent_mm_centres"] = round(float(dproj[-1] - dproj[0]), 3)
rec["geometry"] = geo
if geo["n_gaps"] or geo["n_duplicates"] or not geo["uniform"]:
    rec["geometry_problem"] = True

# voxels
nr, nc = int(slices[0].Rows), int(slices[0].Columns)
vol = np.empty((nc, nr, len(slices)), np.float32)
exact = True
for k, d in enumerate(slices):
    a = d.pixel_array.astype(np.float32)
    s, b = float(getattr(d, "RescaleSlope", 1)), float(getattr(d, "RescaleIntercept", 0))
    if s != 1 or b != round(b):
        exact = False
    vol[:, :, k] = (a * s + b).T
rec["hu_exact_integer"] = exact

A = np.eye(4)
A[:3, 0] = rdir * ps[1]
A[:3, 1] = cdir * ps[0]
step = (ipp[-1] - ipp[0]) / max(1, len(slices) - 1)
sheared = geo["inplane_drift_mm"] >= 0.05
A[:3, 2] = step if sheared else n * med
A[:3, 3] = ipp[0]
A = np.diag([-1, -1, 1, 1]) @ A  # LPS -> RAS
resampled = (not axis_aligned) or sheared
rec["resampled"] = bool(resampled)
if resampled:
    # axis-aligned RAS target grid, same spacings, covering the source corners
    sp = np.array([ps[1], ps[0], med])
    corners = np.array([[i, j, k, 1] for i in (0, nc - 1) for j in (0, nr - 1) for k in (0, len(slices) - 1)]) @ A.T
    lo, hi = corners[:, :3].min(0), corners[:, :3].max(0)
    shp = np.ceil((hi - lo) / sp).astype(int) + 1
    T = np.diag(list(sp) + [1.0]); T[:3, 3] = lo
    M = np.linalg.inv(A) @ T
    grid = np.indices(shp, dtype=np.float32).reshape(3, -1)
    coords = M[:3, :3] @ grid + M[:3, 3:4]
    vol = ndimage.map_coordinates(vol, coords, order=1, cval=-1024).reshape(shp).astype(np.float32)
    A = T
    rec["resample_reason"] = ("oblique IOP" if not axis_aligned else "") + (" gantry tilt/shear" if sheared else "")
img = nib.Nifti1Image(vol, A)
img = nib.as_closest_canonical(img)
data = np.asarray(img.dataobj)
if exact and data.min() >= -32768 and data.max() <= 32767:
    data = np.round(data).astype(np.int16)
out = nib.Nifti1Image(data, img.affine)
out.set_qform(img.affine, code=1); out.set_sform(img.affine, code=1)
out.header.set_xyzt_units("mm", "sec")
out.header["descrip"] = b"ia172 raw HU RAS"
nib.save(out, out_ct)
rec["out_ct"] = out_ct
rec["shape"] = list(data.shape)
rec["spacing_mm"] = [round(float(z), 5) for z in out.header.get_zooms()[:3]]
rec["axcodes"] = "".join(nib.aff2axcodes(out.affine))
rec["affine"] = [[round(float(x), 5) for x in r] for r in out.affine]
rec["dtype"] = str(data.dtype)
rec["z_extent_mm"] = round(float(data.shape[2] * rec["spacing_mm"][2]), 2)
vmin = float(data.min())
rec["min_value"] = vmin
rec["frac_at_min_value"] = round(float((data == vmin).mean()), 4)
pc = np.percentile(data, [1, 50, 99.9])
rec["hu_p1_p50_p999_all_voxels"] = [round(float(x), 1) for x in pc]
body = data > -500
rec["hu_p1_p50_p999_above_m500"] = [round(float(x), 1) for x in np.percentile(data[body], [1, 50, 99.9])]
del body

# 1.5 mm RAS copy (Gaussian prefilter, FWHM = target spacing; linear) for the quick-look TotalSegmentator run
sp = np.array(rec["spacing_mm"])
sig = (1.5 / sp) / 2.3548
sm = ndimage.gaussian_filter(data.astype(np.float32), sigma=np.sqrt(np.maximum(sig ** 2 - (1 / 2.3548) ** 2, 0)))
shp = np.floor((np.array(data.shape) - 1) * sp / 1.5).astype(int) + 1
zoom_idx = [np.arange(s) * 1.5 / spi for s, spi in zip(shp, sp)]
g = np.meshgrid(*zoom_idx, indexing="ij")
lo = ndimage.map_coordinates(sm, [x.ravel() for x in g], order=1, cval=-1024).reshape(shp)
del sm, g
A15 = out.affine.copy()
A15[:3, :3] = out.affine[:3, :3] @ np.diag(1.5 / sp)
lo_img = nib.Nifti1Image(np.round(lo).astype(np.int16), A15)
lo_img.set_qform(A15, code=1); lo_img.set_sform(A15, code=1)
nib.save(lo_img, os.path.join(dd, "ct_1p5.nii.gz"))
rec["ct_1p5"] = {"path": os.path.join(dd, "ct_1p5.nii.gz"), "shape": list(lo.shape)}
rec["seconds"] = round(time.time() - t0, 1)
json.dump(rec, open(os.path.join(dd, "convert.json"), "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("shape", "spacing_mm", "axcodes", "z_extent_mm", "resampled", "geometry",
                                       "hu_p1_p50_p999_all_voxels", "seconds")}))
