#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_args.py (= /sp/v5_va/pilot_proc_song_args.py in the build containers)
"""pilot_proc_song_args.py <case> -- Arterial arguments for a Song 2024 pilot case (neve-build-stagea).
= pilot_proc_ia172_args.py (the 172-series pilot agent's script, read-only) with THREE changes (pilot_proc_song):
  (1) song paths; (2) the RAS assertion relaxed to "axis-aligned, +z superior" (Song CTs are LAS; every computation
  here goes through the affine, and arterial_ts_args already ran on LAS AortaSeg and LPS TaG-Net CTs);
  (3) the arch HU from the Song meta.json (contrast.aorta_hu_median_auto).
The ia172 docstring follows.

Adapted from pilot_proc_aortaseg_args.py (the AortaSeg pilot agent's script, read-only; same logic, ia172 paths, one mask
set, a fallback for arch-apex cases):
1. arterial_ts_args.main (UNCHANGED, imported) on <der>/totalseg (TotalSegmentator total_v3 masks on the CTA grid,
   ct.nii.gz hard link) with --resample 0, i.e. the tool's crop / FOV / skull logic but NO --resample-mm: the brief asks
   for Arterial at the NATIVE grid (0.39-0.51 mm in-plane x 0.25-0.5 mm slice interval; finer than the tool's 0.6 mm
   resample trigger anyway).
2. The chosen bottom slice is checked with Arterial's own rule (centerline_extraction.utils.aortic_arch_endpoint_check:
   8-connected components of the bottom slice of the mask with >= round(500 x 0.07385 / voxel volume) voxels; two are
   needed = ascending + descending aorta) on the TS aorta mask and on the CT inside the TS aorta (HU >= 0.5 x the arch
   median of the fetch quick look).
3. run args (<der>/arterial/args_run.txt):
     --crop-k k0:k1   (tool)            --fov-mm 220 --fov-center=x,y  (tool; the "=" form, a negative x is otherwise
                                                                         parsed as an option by the wrapper)
     --skull-mask <der>/totalseg/segmentations/skull.nii.gz  (TS skull, includes the mandible: the tag_arterial
                                                              best-known native protocol, TS skull for the head/neck split)
   retry args (<der>/arterial/args_retry.txt, used only if Arterial's aortic-arch check fails): the bottom moved to the
   two-limb slice whose smaller TS limb is largest.
   FALLBACK (arch never shows two limbs, the tool returns ok:false -- expected for 090 / 118, whose bottom slice cuts
   only the arch apex): crop from the bottom slice (k0 = 0) to the skull top, FOV from the trachea centroid over the
   lowest 60 mm (the tool's rule, re-implemented), TS skull. Arterial's centerline / labelling stage is then expected to
   fail, but its segmentation is still written and art_extract traces the binary mask from the TS subclavian mask
   (its documented no-label path). No retry crop exists for such a case.
Writes <der>/arterial/args.json.
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/sp/v5_va")
import arterial_ts_args  # noqa: E402  (unchanged study tool)

case = sys.argv[1]
SRC = "/repo/carotid_data/va_sources/thin_slice/song/%s" % case
DER = "/repo/carotid_data/va_sources/derived/thin_slice/song/%s" % case
TSD = os.path.join(DER, "totalseg")
os.makedirs(os.path.join(DER, "arterial"), exist_ok=True)


def run_tool(d):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        arterial_ts_args.main(d, 220.0, 0.0)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


ct = nib.load(os.path.join(SRC, "ct.nii.gz"))
A = ct.affine
assert A[2, 2] > 0 and abs(A[:3, :3] - np.diag(np.diag(A)[:3])).max() < 1e-6, A   # song: axis-aligned, +z superior (LAS allowed)
sp = np.abs(np.diag(A)[:3])
vox = float(np.prod(sp))
thr_counts = int(abs(round(500 * (0.07385254 / vox))))
arr = np.asanyarray(ct.dataobj)
meta = json.load(open(os.path.join(SRC, "meta.json")))
ao_hu = float(meta["contrast"]["aorta_hu_median_auto"])
out = {"case": case, "spacing_mm": sp.tolist(), "arterial_bottom_component_min_voxels": thr_counts,
       "arterial_bottom_component_min_mm2": round(thr_counts * sp[0] * sp[1], 1), "ts_task": "total_v3",
       "tool": run_tool(TSD)}
print("tool", json.dumps(out["tool"]), flush=True)


def comps(m2):
    lab, n = ndimage.label(m2, structure=np.ones((3, 3)))
    rows = []
    for c in range(1, n + 1):
        ii = np.argwhere(lab == c)
        if len(ii) < 10:
            continue
        w = (A @ np.c_[ii, np.zeros(len(ii)), np.ones(len(ii))].T)[:3]
        rows.append({"voxels": int(len(ii)), "area_mm2": round(len(ii) * sp[0] * sp[1], 1),
                     "x_mm": round(float(w[0].mean()), 1), "y_mm": round(float(w[1].mean()), 1),
                     "passes_arterial_size": bool(len(ii) >= thr_counts)})
    rows.sort(key=lambda r: -r["voxels"])
    return rows


ao = np.asanyarray(nib.load(os.path.join(TSD, "segmentations/aorta.nii.gz")).dataobj) > 0


def slice_check(k):
    r = {"k": int(k), "z_mm": round(float(A[2, 3] + A[2, 2] * k), 2)}
    cs = comps(ao[:, :, k])
    r["ts_aorta"] = {"n_ge_arterial_size": sum(c["passes_arterial_size"] for c in cs), "components": cs[:4]}
    sl = arr[:, :, k].astype(np.float32)
    cs = comps((sl >= 0.5 * ao_hu) & (sl <= 900) & ao[:, :, k])
    r["ct_ge_half_arch_inside_ts_aorta"] = {"threshold_HU": round(0.5 * ao_hu, 1),
                                            "n_ge_arterial_size": sum(c["passes_arterial_size"] for c in cs),
                                            "components": cs[:4]}
    return r


kz = np.where(ao.any((0, 1)))[0]
scan_top = int(min(arr.shape[2], (kz.max() + 1) if len(kz) else 60))
out["two_limb_scan"] = [slice_check(k) for k in range(0, min(scan_top, 250), 3)]
skull = os.path.join(TSD, "segmentations/skull.nii.gz")
rec = out["tool"]
if rec.get("ok"):
    k0 = int(rec["args"].split("--crop-k ")[1].split(":")[0])
    k1 = int(rec["args"].split("--crop-k ")[1].split()[0].split(":")[1])
    fov = None
    if "--fov-center" in rec["args"]:
        xy = rec["args"].split("--fov-center ")[1].split()[0]
        fov = "--fov-mm 220 --fov-center=%s" % xy
    mode = "tool"
else:
    # fallback: bottom slice to the skull top; FOV centre = trachea centroid over the lowest 60 mm (the tool's rule)
    k0 = 0
    sk = np.asanyarray(nib.load(skull).dataobj) > 0
    k1 = int(np.where(sk.any((0, 1)))[0].max()) + 1
    tr = np.asanyarray(nib.load(os.path.join(TSD, "segmentations/trachea.nii.gz")).dataobj) > 0
    sub = tr[:, :, k0:min(k1, k0 + int(60 / sp[2]))]
    ii = np.argwhere(sub).astype(float)
    ii[:, 2] += k0
    w = (A @ np.c_[ii, np.ones(len(ii))].T)[:3]
    fov_now = [arr.shape[0] * sp[0], arr.shape[1] * sp[1]]
    fov = "--fov-mm 220 --fov-center=%.1f,%.1f" % (w[0].mean(), w[1].mean()) if max(fov_now) > 220 * 1.1 else None
    mode = "fallback: no two-limb slice (arch apex only); bottom slice k0 = 0, Arterial's centreline stage expected to fail"
if fov and "--fov-center " in fov:
    fov = fov.replace("--fov-center ", "--fov-center=")
out["chosen_bottom"] = slice_check(k0)
print("chosen", json.dumps(out["chosen_bottom"]), flush=True)
best, bk = -1, None
for r in out["two_limb_scan"]:
    c = r["ts_aorta"]["components"]
    if len(c) >= 2 and c[1]["passes_arterial_size"] and c[1]["voxels"] > best and r["k"] != k0:
        best, bk = c[1]["voxels"], r["k"]
run = "--crop-k %d:%d %s --skull-mask %s" % (k0, k1, fov or "", skull)
retry = "" if bk is None else "--crop-k %d:%d %s --skull-mask %s" % (bk, k1, fov or "", skull)
out.update({"mode": mode, "run_args": " ".join(run.split()), "retry_args": " ".join(retry.split()), "retry_bottom_k": bk,
            "native_grid": "no --resample-mm (brief: native resolution)", "skull_mask": skull,
            "skull_mask_why": "TS skull (total_v3; includes the mandible): the tag_arterial best-known native protocol "
                              "(TS skull for the head/neck split)"})
json.dump(out, open(os.path.join(DER, "arterial", "args.json"), "w"), indent=1)
open(os.path.join(DER, "arterial", "args_run.txt"), "w").write(out["run_args"] + "\n")
open(os.path.join(DER, "arterial", "args_retry.txt"), "w").write(out["retry_args"] + "\n")
print("mode:", mode)
print("run:", out["run_args"])
print("retry:", out["retry_args"])
