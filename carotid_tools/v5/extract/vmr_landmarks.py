#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_landmarks.py (= /sp/v5_va/vmr_landmarks.py in the build containers)
"""Bony landmarks along the VMR 0248 VAs (CT): transverse-foramen enclosure profile along the
medial (VMTK, CT-segmentation) centerline, and views to name the levels.

enclosure(s) = share of 48 rays in the normal plane that meet bone (HU >= 400, 2 consecutive
samples) between r_MISR + 0.3 and r_MISR + 7 mm. A transverse foramen encircles the VA
(enclosure ~1); the gaps between transverse processes do not.
eve-training-fixed. Writes results/vmr/landmark_views.png + results/vmr/enclosure.json."""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmr_common import CT, DERIVED, WORK, arclen, jdump, load_vmr, resample, rmf, smooth

TAG = sys.argv[1] if len(sys.argv) > 1 else "ct_seg2"
vmr = load_vmr(with_ct=True)
ct = CT(vmr["ct"])
res = {}
cls = {}
for nm in ("rva", "lva"):
    d = json.load(open(os.path.join(DERIVED, TAG, "%s_stageA_vmtk.json" % nm)))
    P0, R0 = np.asarray(d["points"]), np.asarray(d["radii"])
    s, P, R = resample(P0, 0.25, R0)
    Ps = smooth(P, 4)
    T, U, W = rmf(Ps)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    dirs = np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * W[:, None, :]
    rr = np.arange(0.3, 7.0, 0.2)
    enc = np.zeros(len(P))
    dist_bone = np.full((len(P), 48), np.nan)
    for i in range(len(P)):
        G = P[i][None, None, :] + (R[i] + rr)[None, :, None] * dirs[i][:, None, :]
        h = ct.sample(G.reshape(-1, 3)).reshape(48, len(rr)) >= 400
        hit2 = h[:, 1:] & h[:, :-1]
        any_hit = hit2.any(1)
        enc[i] = any_hit.mean()
        first = np.argmax(hit2, axis=1)
        dist_bone[i, any_hit] = rr[first[any_hit]]
    enc_s = ndimage.uniform_filter1d(enc, 9)
    res[nm] = {"s": s, "z": P[:, 2], "x": P[:, 0], "y": P[:, 1], "enclosure": enc, "enclosure_smooth": enc_s,
               "bone_gap_median_mm": np.nanmedian(dist_bone, axis=1), "r": R}
    cls[nm] = (s, P, R)
    print(nm, "length %.1f" % s[-1], flush=True)
jdump({k: {kk: np.round(np.asarray(vv), 3) for kk, vv in v.items()} for k, v in res.items()},
      os.path.join(WORK, "enclosure_%s.json" % TAG))

# ---------------------------------------------------------------- views
fig = plt.figure(figsize=(30, 22))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
for k, nm in enumerate(("rva", "lva")):
    r = res[nm]
    ax = (ax1, ax2)[k]
    ax.plot(r["enclosure"], r["s"], lw=0.5, color="0.6")
    ax.plot(r["enclosure_smooth"], r["s"], lw=1.2)
    ax.plot((r["z"] - r["z"].min()) / 200.0, r["s"], "g", lw=0.8, label="(z - zmin)/200")
    ax.set_ylim(r["s"][-1], 0)
    ax.set_yticks(np.arange(0, r["s"][-1], 5))
    ax.grid(alpha=0.3)
    ax.set_title("%s enclosure by bone (HU>=400) vs s from origin" % nm)
    for sv in range(0, int(r["s"][-1]), 10):
        i = int(np.searchsorted(r["s"], sv))
        ax.text(1.02, sv, "z%.0f" % r["z"][i], fontsize=7)
# midsagittal slab MIP of bone, x mid between the VAs
xm = 0.5 * (np.median(res["rva"]["x"]) + np.median(res["lva"]["x"]))
ix = int(round((xm - ct.org[0]) / ct.sp[0]))
slab = ct.arr[:, :, ix - 5:ix + 6].max(axis=2)  # [k,j]
ax3 = fig.add_subplot(1, 4, 3)
ax3.imshow(slab, cmap="gray", vmin=-200, vmax=1200, origin="lower",
           extent=[ct.org[1], ct.org[1] + ct.sp[1] * (ct.dims[1] - 1), ct.org[2], ct.org[2] + ct.sp[2] * (ct.dims[2] - 1)])
ax3.set_ylim(-300, -100)
ax3.set_xlim(40, 160)
ax3.set_yticks(np.arange(-300, -100, 5))
ax3.grid(alpha=0.3)
ax3.set_title("midsagittal slab MIP x=%.1f +-2 mm (y anterior to the right)" % xm)
# coronal slab MIP through the VA y-range
ys = np.r_[res["rva"]["y"], res["lva"]["y"]]
j0, j1 = [int(round((v - ct.org[1]) / ct.sp[1])) for v in (np.percentile(ys, 5) - 3, np.percentile(ys, 95) + 3)]
cor = ct.arr[:, j0:j1 + 1, :].max(axis=1)
ax4 = fig.add_subplot(1, 4, 4)
ax4.imshow(cor, cmap="gray", vmin=-200, vmax=1200, origin="lower",
           extent=[ct.org[0], ct.org[0] + ct.sp[0] * (ct.dims[0] - 1), ct.org[2], ct.org[2] + ct.sp[2] * (ct.dims[2] - 1)])
for nm, c in (("rva", "r"), ("lva", "c")):
    s, P, R = cls[nm]
    ax4.plot(P[:, 0], P[:, 2], c, lw=0.8)
    for sv in range(0, int(s[-1]), 10):
        i = int(np.searchsorted(s, sv))
        ax4.text(P[i, 0], P[i, 2], str(sv), color=c, fontsize=8)
ax4.set_ylim(-300, -90)
ax4.set_xlim(-170, -40)
ax4.set_yticks(np.arange(-300, -90, 5))
ax4.grid(alpha=0.3)
ax4.set_title("coronal slab MIP y %.0f..%.0f (patient right = +x = image right)" % (np.percentile(ys, 5) - 3, np.percentile(ys, 95) + 3))
plt.tight_layout()
plt.savefig(os.path.join(WORK, "landmark_views_%s.png" % TAG), dpi=50)
print("ok")
