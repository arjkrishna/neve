#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_protrusion.py (= /sp/v5_va/pilot_proc_ia172_protrusion.py in the build containers)
"""pilot_proc_ia172_protrusion.py <case> [...] -- what the ct_seg4 label's outward relief is made of (neve-build-stagea).

For every VA label voxel of <der>/ct_seg/ct_seg4 (labels_kji.npz): nearest station j of its own vessel (stations_rays.json),
radial distance rho from the station line, excess = rho - median_j(r_e) (the station's median edge radius over 64 rays),
and the normalised native HU h = (HU - I_bg,j) / (I_c,j - I_bg,j) (1 = lumen core, 0.5 = the half-max edge, 0 = tissue).
"Protrusion" voxels: excess > 0.75 mm (beyond the station's own median edge by more than ~1.5 in-plane voxels).
  * a real lumen bulge / oval wider than the edge blur (FWHM ~1.3 mm) fills with lumen-bright voxels (h >= 0.75 inside);
  * a structure merged at the half-max threshold that is dimmer than the lumen (a partly enhanced vein, the perivertebral
    plexus) or thin (a small branch, a partial-volume halo) reads 0.5 <= h < 0.75 throughout;
  * bone touching the lumen is cut by the upper threshold, so it rarely appears here.
Reports per VA: label voxels, protrusion share, h quantiles of the protrusion voxels, share with h >= 0.75, the
same per 10 mm of s, and whether each protrusion voxel lies within 1 mm of a TS bone mask (vertebrae, skull).
Writes <der>/ct_seg/ct_seg4/protrusion.json.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
SRCR = "/repo/carotid_data/va_sources/thin_slice/ia172"


def one(case, tag="ct_seg4"):
    D = os.path.join(ROOT, case, "ct_seg", tag)
    z = np.load(os.path.join(D, "labels_kji.npz"))
    L, org, sp = z["labels"], z["origin_mm"], z["spacing_mm"]
    st = json.load(open(os.path.join(D, "stations_rays.json")))
    ct = nib.load(os.path.join(SRCR, case, "ct.nii.gz"))
    A0 = ct.affine
    a = np.round((org - A0[:3, 3]) / np.diag(A0)[:3]).astype(int)          # labels_kji origin -> NIfTI index (RAS, + diagonal)
    b = a + np.array(L.shape[::-1]) - 1
    blk = np.asarray(ct.dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).astype(np.float32).transpose(2, 1, 0)
    assert blk.shape == L.shape, (blk.shape, L.shape)
    # TS bone within 1 mm (vertebrae + skull), on the CT grid
    seg = os.path.join(ROOT, case, "totalseg/segmentations")
    bone = np.zeros(L.shape, bool)
    for nm in ["vertebrae_C%d" % k for k in range(1, 8)] + ["vertebrae_T1", "skull"]:
        p = os.path.join(seg, nm + ".nii.gz")
        if os.path.isfile(p):
            bone |= np.asarray(nib.load(p).dataobj[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]).transpose(2, 1, 0) > 0
    near_bone = ndimage.distance_transform_edt(~bone, sampling=sp[::-1]) <= 1.0 if bone.any() else np.zeros(L.shape, bool)
    out = {}
    for k, v in ((1, "rva"), (2, "lva")):
        if v not in st or not (L == k).any():
            continue
        P = np.asarray(st[v]["P"], float)
        s = np.asarray(st[v]["s"], float)
        re = np.asarray(st[v]["r_e"], float)
        Ic, Ib = np.asarray(st[v]["Ic"], float), np.asarray(st[v]["Ib"], float)
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        idx = np.argwhere(L == k)
        X = org + idx[:, ::-1] * sp
        _, j = cKDTree(P).query(X)
        d = X - P[j]
        d -= np.einsum("ij,ij->i", d, T[j])[:, None] * T[j]
        rho = np.linalg.norm(d, axis=1)
        exc = rho - np.median(re, axis=1)[j]
        hu = blk[tuple(idx.T)]
        h = (hu - Ib[j]) / np.maximum(Ic[j] - Ib[j], 1.0)
        pr = exc > 0.75
        nb = near_bone[tuple(idx.T)]
        rec = {"label_voxels": int(len(idx)), "protrusion_voxels": int(pr.sum()), "protrusion_share": round(float(pr.mean()), 4),
               "h_protrusion_q10_q50_q90": np.round(np.percentile(h[pr], [10, 50, 90]), 3).tolist() if pr.any() else None,
               "protrusion_share_lumen_bright_h_ge_0.75": round(float(np.mean(h[pr] >= 0.75)), 3) if pr.any() else None,
               "protrusion_share_intermediate_0.5_0.75": round(float(np.mean((h[pr] >= 0.5) & (h[pr] < 0.75))), 3) if pr.any() else None,
               "protrusion_share_within_1mm_of_TS_bone": round(float(np.mean(nb[pr])), 3) if pr.any() else None,
               "h_all_label_q50": round(float(np.median(h)), 3)}
        bins = []
        for a0 in np.arange(0.0, s[-1], 10.0):
            m = (s[j] >= a0) & (s[j] < a0 + 10.0)
            mp = m & pr
            bins.append([round(float(a0)), int(m.sum()), round(float(mp.sum() / max(m.sum(), 1)), 3),
                         round(float(np.median(h[mp])), 3) if mp.any() else None])
        rec["per_10mm_[from, voxels, protrusion_share, h_median_protrusion]"] = bins
        out[v] = rec
        print(case, v, json.dumps({kk: vv for kk, vv in rec.items() if not kk.startswith("per_10mm")}), flush=True)
    json.dump(out, open(os.path.join(D, "protrusion.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for c in sys.argv[1:]:
        try:
            one(c)
        except Exception as e:  # noqa: BLE001
            print(c, "failed:", repr(e))
