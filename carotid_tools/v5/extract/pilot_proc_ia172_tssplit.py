#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_tssplit.py (= /sp/v5_va/pilot_proc_ia172_tssplit.py in the build containers)
"""pilot_proc_ia172_tssplit.py <case> -- split TotalSegmentator's multilabel output (--ml) into one uint8
<class>.nii.gz per class, on the CTA grid, exactly as TotalSegmentator's own per-class saving lays them out
(<der>/totalseg/segmentations/<class>.nii.gz, same affine / qform / sform as the multilabel = the input CT).

Why: on these 300-370 M-voxel grids TotalSegmentator saves the 117 per-class files single-threaded ("Shape of output
image is very big"), ~15 min with the GPU idle but locked. With --ml the GPU container writes one file and exits, and
this CPU step (no GPU lock) writes the per-class files the study tools read. Class names: TotalSegmentator 2.18.0
map_to_binary.class_map["total_v3"] (saved in derived/thin_slice/ia172/_ts_total_v3_labels.json from the same image).
neve-build-stagea. Writes the 117 masks, then deletes the multilabel file.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import nibabel as nib

case = sys.argv[1]
ROOT = "/repo/carotid_data/va_sources/derived/thin_slice/ia172"
D = os.path.join(ROOT, case, "totalseg")
ML = os.path.join(D, "segmentations_ml.nii.gz")
OUT = os.path.join(D, "segmentations")
CMAP = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "_ts_total_v3_labels.json")))["total_v3"].items()}
_lab = None
_img = None


def init():
    global _lab, _img
    _img = nib.load(ML)
    _lab = np.asanyarray(_img.dataobj).astype(np.uint8)


def one(k):
    m = (_lab == k).astype(np.uint8)
    im = nib.Nifti1Image(m, _img.affine, _img.header)
    im.set_data_dtype(np.uint8)
    nib.save(im, os.path.join(OUT, CMAP[k] + ".nii.gz"))
    return CMAP[k], int(m.sum())


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "4")), initializer=init) as p:
        res = p.map(one, sorted(CMAP), chunksize=4)
    vox = dict(res)
    json.dump({"from": ML, "n_classes": len(res), "voxels": vox, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(D, "split.json"), "w"), indent=0)
    assert len(res) == len(CMAP) and os.path.isfile(os.path.join(OUT, "vertebrae_C1.nii.gz"))
    os.remove(ML)
    print("split %d classes in %.0f s; C1 %d vox, skull %d vox" % (len(res), time.time() - t0, vox["vertebrae_C1"], vox["skull"]))
