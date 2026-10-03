# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_ia172_slicecheck.py (= /sp/v5_va/pilot_ia172_slicecheck.py in the build containers)
"""pilot_ia172_slicecheck: per-slice sanity of the converted volumes.
For every axial slice: minimum value (padding), fraction below -1100 HU (outside the reconstruction circle), and the
mean absolute difference of the body voxels (-500..2000 HU in both) to the two neighbouring slices, relative to the
median of that statistic (a corrupted or mis-ordered slice stands out as a spike).
-> /sp/v5_va/results/pilot_ia172_slicecheck.json.  usage: python3 pilot_ia172_slicecheck.py <case> ..."""
import json
import os
import sys

import nibabel as nib
import numpy as np

IMG = "/repo/carotid_data/va_sources/thin_slice/ia172"
OUT = "/sp/v5_va/results/pilot_ia172_slicecheck.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for case in sys.argv[1:]:
    v = np.asarray(nib.load(os.path.join(IMG, case, "ct.nii.gz")).dataobj)
    n = v.shape[2]
    mins = v.min(axis=(0, 1)).astype(int)
    vals, cnt = np.unique(mins, return_counts=True)
    mode = int(vals[np.argmax(cnt)])
    odd = [int(k) for k in np.where(mins != mode)[0]]
    outside = (v < -1100).mean(axis=(0, 1))
    d = np.zeros(n - 1)
    for k in range(n - 1):
        a = v[::2, ::2, k].astype(np.float32); b = v[::2, ::2, k + 1].astype(np.float32)
        m = (a > -500) & (a < 2000) & (b > -500) & (b < 2000)
        d[k] = np.abs(a[m] - b[m]).mean() if m.any() else np.nan
    med = float(np.nanmedian(d))
    spikes = [int(k) for k in np.where(d > 3 * med)[0]]
    rec = {"n_slices": n, "padding_mode": mode, "slices_with_other_min": odd[:50],
           "other_min_values": sorted({int(mins[k]) for k in odd})[:10],
           "outside_circle_frac_median": round(float(np.median(outside)), 4),
           "neighbour_absdiff_median_HU": round(med, 2), "neighbour_absdiff_max_HU": round(float(np.nanmax(d)), 2),
           "spike_pairs_gt_3x_median": spikes[:50]}
    for k in odd[:5]:
        lo, hi = max(0, k - 1), min(n - 2, k)
        rec.setdefault("absdiff_around_odd_slices", {})[str(k)] = [round(float(d[j]), 2) for j in range(lo, hi + 1)]
    res[case] = rec
    print(case, json.dumps(rec), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
