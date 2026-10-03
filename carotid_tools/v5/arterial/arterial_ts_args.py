#!/usr/bin/env python
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\arterial_ts_args.py (= /sp/v5_va/arterial_ts_args.py in the build containers)
"""Suggest run_arterial.sh arguments for a TotalSegmentator v3 subject so that the CT matches the
assumptions hard-coded in Arterial's centerline stage:

  * the bottom slice must cut the aortic arch so that BOTH the ascending and the descending aorta
    appear as separate blobs (aortic_arch_endpoint_check / seed heuristics);
  * the in-plane FOV should be ~220 mm (512 px x 0.43 mm) centred on the neck: the start point is
    picked as the endpoint closest to voxel (350*0.43/sp, 0, 0) of the LAS grid, i.e. ~40 mm left of
    the FOV centre at its posterior edge (= the descending aorta when the FOV is centred on the neck);
  * the vessel bounding box must stay within 2.5e6..3.5e7 mm^3 (crop neck-to-leg CTAs).

usage: python arterial_ts_args.py <ts_subject_dir> [--fov 220] [--resample 0.5]
prints one JSON line: {"ok": bool, "why": ..., "args": "<options for run_arterial.sh>"}
Uses the TS masks aorta, trachea, skull (all in the CT grid). Paths in "args" are container
paths (/repo/...) and are accepted by run_arterial.sh as is.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation
from scipy import ndimage


def main(d, fov=220.0, resample=0.5):
    seg = os.path.join(d, "segmentations")
    ct = nib.load(os.path.join(d, "ct.nii.gz"))
    A = ct.affine
    orn = io_orientation(A)
    sax = int(np.where(orn[:, 0] == 2)[0][0])
    up = orn[sax, 1] > 0                      # voxel index increases towards S
    n = ct.shape[sax]

    def mask(name):
        p = os.path.join(seg, name + ".nii.gz")
        return np.asanyarray(nib.load(p).dataobj) > 0 if os.path.isfile(p) else None

    def k_to_s(k):                           # source voxel index along sax -> "height" index (0 = bottom)
        return k if up else n - 1 - k

    out = {"subject": os.path.basename(d.rstrip("/")), "ok": False}
    ao = mask("aorta")
    if ao is None or not ao.any():
        out["why"] = "no aorta in FOV"
        print(json.dumps(out)); return
    ao_h = np.moveaxis(ao, sax, 2)
    if not up:
        ao_h = ao_h[:, :, ::-1]
    hs = np.where(ao_h.any((0, 1)))[0]
    two = [h for h in hs if ndimage.label(ao_h[:, :, h])[1] >= 2]
    if not two:
        out["why"] = "aortic arch never shows two limbs in one axial slice (arch cut above the arch top?)"
        print(json.dumps(out)); return
    h0 = int(two[0] + 0.25 * (two[-1] - two[0]))          # a slice well inside the two-limb range
    sk = mask("skull")
    if sk is not None and sk.any():
        sk_h = np.moveaxis(sk, sax, 2)
        if not up:
            sk_h = sk_h[:, :, ::-1]
        h1 = int(np.where(sk_h.any((0, 1)))[0].max()) + 1
    else:
        h1 = n
    k0, k1 = (h0, h1) if up else (n - h1, n - h0)
    # in-plane centre: trachea centroid over the lowest 60 mm of the kept slab (fallback: aorta centroid)
    sp = np.sqrt((A[:3, :3] ** 2).sum(0))
    tr = mask("trachea")
    ctr = None
    for m in (tr, ao):
        if m is None:
            continue
        mh = np.moveaxis(m, sax, 2)
        if not up:
            mh = mh[:, :, ::-1]
        sl = mh[:, :, h0:min(h1, h0 + int(60 / sp[sax]))]
        if sl.any():
            ijk = np.argwhere(sl).astype(float)
            ijk[:, 2] += h0
            # back to source voxel indices
            src = np.zeros_like(ijk)
            other = [a for a in range(3) if a != sax]
            src[:, other[0]] = ijk[:, 0]; src[:, other[1]] = ijk[:, 1]
            src[:, sax] = ijk[:, 2] if up else n - 1 - ijk[:, 2]
            w = (A @ np.c_[src, np.ones(len(src))].T)[:3]
            ctr = (float(w[0].mean()), float(w[1].mean()))
            out["fov_center_from"] = "trachea" if m is tr else "aorta"
            break
    fov_now = [float(ct.shape[a] * sp[a]) for a in range(3) if a != sax]
    args = [f"--crop-k {k0}:{k1}"]
    if max(fov_now) > fov * 1.1 and ctr is not None:
        args.append(f"--fov-mm {fov:g} --fov-center {ctr[0]:.1f},{ctr[1]:.1f}")
    if sk is not None and sk.any():
        args.append(f"--skull-mask {os.path.join(seg, 'skull.nii.gz')}")
    if resample and max(sp) > resample * 1.2:
        args.append(f"--resample-mm {resample:g}")
    out.update({"ok": True, "two_limb_slices": [int(two[0]), int(two[-1])], "bottom_h": h0, "top_h": h1,
                "fov_mm_now": [round(x, 1) for x in fov_now], "args": " ".join(args)})
    print(json.dumps(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    fov = float(a[a.index("--fov") + 1]) if "--fov" in a else 220.0
    rs = float(a[a.index("--resample") + 1]) if "--resample" in a else 0.5
    main(a[0], fov, rs)
