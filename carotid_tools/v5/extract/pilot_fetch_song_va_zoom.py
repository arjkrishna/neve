#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_va_zoom.py (= /sp/v5_va/pilot_fetch_song_va_zoom.py in the build containers)
"""pilot_fetch_song_va_zoom.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:36 EDT): zoomed axial crops for the by-eye
VA verdict. 10 levels arch+25..+185 mm, each 76 x 66 mm centred on the vertebral-column track of qc_va_check.json
(dx -38..+38, dy -30..+36), window 0..1.6 x aorta median (VA lumen mid-grey, cortical bone white), 'R'/'L' marked.
Writes <case>/qc_va_zoom.png. Read-only otherwise. usage: pilot_fetch_song_va_zoom.py CASE [CASE ...] (neve-build-stagea)"""
import sys, os, json
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
IMG = "/repo/carotid_data/va_sources/thin_slice/song"
for case in sys.argv[1:]:
    d = os.path.join(IMG, case)
    if os.path.exists(os.path.join(d, "qc_va_zoom.png")):
        print(case, "skip (done)", flush=True); continue
    q = json.load(open(os.path.join(d, "qc_pass1.json"))); vc = json.load(open(os.path.join(d, "qc_va_check.json")))
    tr = np.array(vc["column"]["track_every_5mm"], float)
    ras = nib.as_closest_canonical(nib.load(os.path.join(d, "ct.nii.gz")))
    A = ras.affine; sp = np.diag(A)[:3]; org = A[:3, 3]; nx, ny, nz = ras.shape
    zmm = org[2] + np.arange(nz) * sp[2]
    za = float(q["aorta_auto"]["arch_top_z"]); aoh = float(q["aorta_auto"]["aorta_hu_med"]); vmax = max(450.0, 1.6 * aoh)
    levels = [25, 45, 65, 85, 105, 125, 145, 160, 172, 185]
    fig, ax = plt.subplots(2, 5, figsize=(30, 13.5))
    ext = [org[0], org[0] + nx * sp[0], org[1], org[1] + ny * sp[1]]
    for a, dz in zip(ax.ravel(), levels):
        z = za + dz
        if z > zmm[-1]:
            a.axis("off"); continue
        k = int(np.argmin(np.abs(zmm - z)))
        i = int(np.argmin(np.abs(tr[:, 0] - zmm[k]))); cx, cy = tr[i, 1], tr[i, 2]
        s = np.asarray(ras.dataobj[:, :, k], dtype=np.float32)
        a.imshow(s.T, cmap="gray", origin="lower", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
        a.set_xlim(cx + 38, cx - 38); a.set_ylim(cy - 30, cy + 36)
        a.text(cx + 35, cy + 32, "R", color="r", fontsize=16, va="top"); a.text(cx - 35, cy + 32, "L", color="r", fontsize=16, va="top", ha="right")
        a.plot(cx, cy, "g+", ms=10)
        a.set_title("%s arch%+d (z %.0f, k %d)" % (case, dz, zmm[k], k), fontsize=12); a.tick_params(labelsize=7)
    fig.suptitle("%s  zoomed axial crops on the vertebral column; window 0..%d HU (aorta median %d): VA lumen = mid-grey round "
                 "in or beside the transverse foramen, cortical bone white" % (case, vmax, aoh), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(d, "qc_va_zoom.png"), dpi=55); plt.close(fig)
    print(case, "qc_va_zoom.png", flush=True)
