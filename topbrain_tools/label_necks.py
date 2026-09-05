#!/usr/bin/env python3
"""Screen the raw TopBrain label masks for NECKS before anything is built.

A neck is a run of skeleton voxels whose inscribed radius (Euclidean distance
transform of the label) is under --thin mm: the segmentation is one or two
voxels thin there. An internal carotid is never that narrow, so a neck is a
label error, and downstream it becomes a centerline radius dip that the
mesher turns into a pinched or severed lumen. Both anatomies the
component-aware check later rejected (mr_015, mr_003_L) and both it called
borderline (mr_013, mr_014) are necks by this test.

Writes <out>: {vessel: {necks, longest_run_mm, edt_min, p05, reject}}.

    python topbrain_tools/label_necks.py <mask_dir> --out topbrain_data/label_necks.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

LABELS = {4: "rICA", 6: "lICA"}


def neck_runs(skel, edt, thin, spacing):
    """Length in mm of every connected run of thin skeleton voxels."""
    from scipy import ndimage
    thin_mask = skel & (edt < thin)
    lab, n = ndimage.label(thin_mask, structure=np.ones((3, 3, 3)))
    runs = []
    for i in range(1, n + 1):
        pts = np.argwhere(lab == i) * spacing
        # chord of the run, a lower bound on its length
        runs.append(float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) > 1 else float(min(spacing)))
    return sorted(runs, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask_dir")
    ap.add_argument("--out", default="topbrain_data/label_necks.json")
    ap.add_argument("--thin", type=float, default=0.6, help="inscribed radius below this is a neck")
    ap.add_argument("--reject-run", type=float, default=3.0, help="longest neck run (mm) that rejects a vessel")
    a = ap.parse_args()
    import nibabel as nib
    from scipy import ndimage
    from skimage.morphology import skeletonize

    out = {}
    print("%-20s %6s %8s %8s %8s %8s" % ("vessel", "necks", "longest", "edt_min", "p05", "verdict"))
    for f in sorted(glob.glob(os.path.join(a.mask_dir, "*.nii.gz"))):
        stem = os.path.basename(f)[:-7]
        img = nib.load(f)
        d = np.asarray(img.dataobj)
        sp = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
        for lab, tag in LABELS.items():
            m = d == lab
            if not m.any():
                continue
            lb, k = ndimage.label(m, structure=np.ones((3, 3, 3)))
            sizes = ndimage.sum(m, lb, range(1, k + 1))
            comp = lb == (int(np.argmax(sizes)) + 1)
            edt = ndimage.distance_transform_edt(comp, sampling=sp)
            sk = skeletonize(comp)
            e = edt[sk]
            runs = neck_runs(sk, edt, a.thin, sp)
            longest = runs[0] if runs else 0.0
            rec = {"necks": int((e < a.thin).sum()), "longest_run_mm": round(longest, 2),
                   "edt_min": round(float(e.min()), 2), "p05": round(float(np.percentile(e, 5)), 2),
                   "reject": bool(longest > a.reject_run)}
            out["%s_%s" % (stem, tag)] = rec
            print("%-20s %6d %8.2f %8.2f %8.2f %8s" % (stem[-6:] + "_" + tag, rec["necks"], longest,
                                                      rec["edt_min"], rec["p05"], "REJECT" if rec["reject"] else ""), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    rej = [k for k, v in out.items() if v["reject"]]
    print("\n%d vessels, %d with necks, %d rejected: %s" % (
        len(out), sum(v["necks"] > 0 for v in out.values()), len(rej), ", ".join(rej)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
