#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_protrusion.py (= /sp/v5_va/pilot_proc_song_protrusion.py in the build containers)
"""pilot_proc_song_protrusion.py <case> [...] -- what the ct_seg4 label's outward relief is made of (neve-build-stagea).
SONG COPY (pilot_proc_song, 2026-10-02 01:15) of pilot_proc_ia172_protrusion.py with these diffs:
  (1) LAS-safe block read (Song CTs are LAS): the label grid (labels_kji.npz origin = world min corner, positive spacing)
      is mapped to NIfTI indices per axis and axes with a negative affine diagonal are flipped (as ctseg_common.load_ct_crop);
      the TS masks (same grid as the CT; TS ran on an exact copy) are read the same way;
  (2) contamination classes (the user's ranking excludes relief from vein / carotid / bone contamination): per VA the
      protrusion share within 1 mm of (a) TS bone (C1-C7, T1, skull: as ia172), (b) TS carotid (common_carotid_artery_*),
      (c) TS veins (brachiocephalic_vein_*, superior_vena_cava; total_v3 has no jugular / vertebral-plexus class),
      (d) other TS arteries (subclavian_artery_*, brachiocephalic_trunk, aorta); plus the over-bright share h > 1.3
      (brighter than the station's lumen core: dense venous contrast or bone) beside ia172's intermediate 0.5-0.75 share;
      'contaminated' = intermediate OR over-bright OR within 1 mm of (b) / (c) (revisions 01:20 / 01:27: the TS
      vertebra masks FILL the transverse foramina -- 66-71 % of 094's VA label lies inside them -- so neither "inside"
      nor "within 1 mm of" TS bone can flag bone contamination; (d) is the VA's own parent; all still reported);
      'clean_lumen_bright' = the rest with h >= 0.75; shares of the protrusion voxels and per 10 mm;
  (3) argv tag: PILOT_SEG_TAG env (default ct_seg4).
Method (unchanged): for every VA label voxel, nearest station j of its own vessel (stations_rays.json), radial distance rho
from the station line, excess = rho - median_j(r_e), normalised native HU h = (HU - I_bg,j) / (I_c,j - I_bg,j) on the
SHIPPED CT (the CT ct_seg4 ran on). Protrusion = excess > 0.75 mm. Writes <der>/ct_seg/<tag>/protrusion.json.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/song"
SRCR = "/repo/carotid_data/va_sources/thin_slice/song"
TAG = os.environ.get("PILOT_SEG_TAG", "ct_seg4")


def block(img, org, sp, shape_kji):
    """img block on the label grid [k, j, i] (world = org + (i, j, k) * sp), flipping negative-diagonal axes."""
    A = img.affine
    assert np.abs(A[:3, :3] - np.diag(np.diag(A[:3, :3]))).max() < 1e-6, A
    n = np.array(shape_kji[::-1])
    i0 = np.rint((org - A[:3, 3]) / np.diag(A)[:3]).astype(int)
    i1 = np.rint((org + (n - 1) * sp - A[:3, 3]) / np.diag(A)[:3]).astype(int)
    lo, hi = np.minimum(i0, i1), np.maximum(i0, i1)
    sh = np.array(img.shape[:3])
    assert (lo >= 0).all() and (hi < sh).all(), (lo, hi, sh)
    b = np.asarray(img.dataobj[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1])
    for ax in range(3):
        if A[ax, ax] < 0:
            b = np.flip(b, axis=ax)
    b = np.ascontiguousarray(b.transpose(2, 1, 0))
    assert b.shape == tuple(shape_kji), (b.shape, shape_kji)
    return b


def one(case, tag=TAG):
    D = os.path.join(ROOT, case, "ct_seg", tag)
    z = np.load(os.path.join(D, "labels_kji.npz"))
    L, org, sp = z["labels"], np.asarray(z["origin_mm"], float), np.asarray(z["spacing_mm"], float)
    st = json.load(open(os.path.join(D, "stations_rays.json")))
    ct = nib.load(os.path.join(SRCR, case, "ct.nii.gz"))
    blk = block(ct, org, sp, L.shape).astype(np.float32)
    # frame check: the label's own I_c should be lumen-bright on this block
    seg = os.path.join(ROOT, case, "totalseg/segmentations")

    def mask(names):
        m = np.zeros(L.shape, bool)
        for nm in names:
            p = os.path.join(seg, nm + ".nii.gz")
            if os.path.isfile(p):
                m |= block(nib.load(p), org, sp, L.shape) > 0
        return m

    def near(m):
        return ndimage.distance_transform_edt(~m, sampling=sp[::-1]) <= 1.0 if m.any() else np.zeros(L.shape, bool)

    bone_m = mask(["vertebrae_C%d" % k for k in range(1, 8)] + ["vertebrae_T1", "skull"])
    classes = {
        "bone_inside": bone_m,
        "bone": near(bone_m),
        "carotid": near(mask(["common_carotid_artery_left", "common_carotid_artery_right"])),
        "vein": near(mask(["brachiocephalic_vein_left", "brachiocephalic_vein_right", "superior_vena_cava"])),
        "other_artery": near(mask(["subclavian_artery_left", "subclavian_artery_right", "brachiocephalic_trunk", "aorta"])),
    }
    out = {"_about": "pilot_proc_song_protrusion.py (ia172 protrusion + LAS read + contamination classes); tag " + tag}
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
        exc = rho - np.nanmedian(re, axis=1)[j]
        hu = blk[tuple(idx.T)]
        h = (hu - Ib[j]) / np.maximum(Ic[j] - Ib[j], 1.0)
        pr = exc > 0.75
        cl = {c: m[tuple(idx.T)] for c, m in classes.items()}
        inter = (h >= 0.5) & (h < 0.75)
        over = h > 1.3
        # 01:20 revision: V2 runs in the transverse foramina, so "within 1 mm of TS bone" covers 75-87 % of the whole label
        # (094) and cannot separate contamination; the flag uses INSIDE the TS bone mask; the parent subclavian / aorta
        # (other_artery) is the VA's own origin and is reported but not counted.
        # 01:27 second revision: 66-71 % of 094's VA label lies INSIDE the TS vertebra masks (total_v3 fills the transverse
        # foramina), so TS bone cannot flag bone contamination at all; bone contact is left to the over-bright share here
        # and to the edge-ray contact rule (grade.py contact_free); both bone shares are still reported.
        anyc = inter | over | cl["carotid"] | cl["vein"]
        clean = (~anyc) & (h >= 0.75)

        def sh(m):
            return round(float(np.mean(m[pr])), 3) if pr.any() else None

        rec = {"label_voxels": int(len(idx)), "protrusion_voxels": int(pr.sum()), "protrusion_share": round(float(pr.mean()), 4),
               "h_protrusion_q10_q50_q90": np.round(np.percentile(h[pr], [10, 50, 90]), 3).tolist() if pr.any() else None,
               "protrusion_share_lumen_bright_h_ge_0.75": sh(h >= 0.75),
               "protrusion_share_intermediate_0.5_0.75": sh(inter),
               "protrusion_share_over_bright_h_gt_1.3": sh(over),
               "protrusion_share_inside_TS_bone": sh(cl["bone_inside"]),
               "protrusion_share_within_1mm_of_TS_bone": sh(cl["bone"]),
               "label_share_inside_TS_bone": round(float(cl["bone_inside"].mean()), 4),
               "protrusion_share_within_1mm_of_TS_carotid": sh(cl["carotid"]),
               "protrusion_share_within_1mm_of_TS_vein": sh(cl["vein"]),
               "protrusion_share_within_1mm_of_TS_other_artery": sh(cl["other_artery"]),
               "protrusion_share_contaminated_any": sh(anyc),
               "protrusion_share_clean_lumen_bright": sh(clean),
               "label_share_within_1mm_of_TS_bone": round(float(cl["bone"].mean()), 4),
               "h_all_label_q50": round(float(np.median(h)), 3)}
        bins = []
        for a0 in np.arange(0.0, s[-1], 10.0):
            m = (s[j] >= a0) & (s[j] < a0 + 10.0)
            mp = m & pr
            bins.append([round(float(a0)), int(m.sum()), round(float(mp.sum() / max(m.sum(), 1)), 3),
                         round(float(np.median(h[mp])), 3) if mp.any() else None,
                         round(float(np.mean(anyc[mp])), 3) if mp.any() else None])
        rec["per_10mm_[from, voxels, protrusion_share, h_median_protrusion, contaminated_share]"] = bins
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
