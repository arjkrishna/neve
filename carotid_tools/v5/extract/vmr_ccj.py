#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_ccj.py (= /sp/v5_va/vmr_ccj.py in the build containers)
"""Cranio-cervical junction views: midsagittal zoom (basion / opisthion), and the V3 course of both VAs
over C1 / occiput in axial, coronal, sagittal slab MIPs, path labelled by s (VMTK, CT seg)."""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vmr_common import CT, DERIVED, WORK, load_vmr, resample

vmr = load_vmr(with_ct=True)
ct = CT(vmr["ct"])
A = ct.arr
x = ct.org[0] + np.arange(ct.dims[0]) * ct.sp[0]
y = ct.org[1] + np.arange(ct.dims[1]) * ct.sp[1]
z = ct.org[2] + np.arange(ct.dims[2]) * ct.sp[2]
def ix(v, ax): return int(round((v - ct.org[ax]) / ct.sp[ax]))
cl = {}
for nm in ("rva", "lva"):
    d = json.load(open(os.path.join(DERIVED, "ct_seg2", "%s_stageA_vmtk.json" % nm)))
    s, P, R = resample(np.asarray(d["points"]), 0.25, np.asarray(d["radii"]))
    cl[nm] = (s, P)
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
xm = -107.9
sl = A[:, :, ix(xm, 0) - 3:ix(xm, 0) + 4].mean(2)
ax = axs[0, 0]
ax.imshow(sl, cmap="gray", vmin=-200, vmax=1300, origin="lower", extent=[y[0], y[-1], z[0], z[-1]])
ax.set_xlim(55, 135); ax.set_ylim(-175, -115)
ax.set_xticks(np.arange(55, 135, 2)); ax.set_yticks(np.arange(-175, -115, 2)); ax.grid(alpha=0.35)
ax.tick_params(labelsize=7)
ax.set_title("midsagittal x=-107.9 (anterior right)")
zr = (-170, -130)
ax = axs[0, 1]
ax.imshow(A[ix(zr[0], 2):ix(zr[1], 2)].max(0), cmap="gray", vmin=-200, vmax=1300, origin="lower",
          extent=[x[0], x[-1], y[0], y[-1]])
ax.set_xlim(-160, -55); ax.set_ylim(40, 150)
ax.set_title("axial MIP z %d..%d (x: patient right = +x = image right; y anterior up)" % zr)
ax2 = axs[0, 2]
yr = (60, 125)
ax2.imshow(A[:, ix(yr[0], 1):ix(yr[1], 1)].max(1), cmap="gray", vmin=-200, vmax=1300, origin="lower",
           extent=[x[0], x[-1], z[0], z[-1]])
ax2.set_xlim(-160, -55); ax2.set_ylim(-190, -100)
ax2.set_title("coronal MIP y %d..%d" % yr)
for nm, c in (("rva", "r"), ("lva", "c")):
    s, P = cl[nm]
    m = (s > 90)
    axs[0, 1].plot(P[m, 0], P[m, 1], c, lw=1)
    ax2.plot(P[m, 0], P[m, 2], c, lw=1)
    for sv in range(90, int(s[-1]), 5):
        i = int(np.searchsorted(s, sv))
        axs[0, 1].text(P[i, 0], P[i, 1], str(sv), color=c, fontsize=7)
        ax2.text(P[i, 0], P[i, 2], str(sv), color=c, fontsize=7)
# sagittal MIPs through each VA x band
for k, (nm, c) in enumerate((("rva", "r"), ("lva", "c"))):
    s, P = cl[nm]
    m = (s > 90)
    xa, xb = P[m, 0].min() - 2, P[m, 0].max() + 2
    ax = axs[1, k]
    ax.imshow(A[:, :, ix(xa, 0):ix(xb, 0)].max(2), cmap="gray", vmin=-200, vmax=1300, origin="lower",
              extent=[y[0], y[-1], z[0], z[-1]])
    ax.plot(P[m, 1], P[m, 2], c, lw=1)
    for sv in range(90, int(s[-1]), 5):
        i = int(np.searchsorted(s, sv))
        ax.text(P[i, 1], P[i, 2], str(sv), color=c, fontsize=7)
    ax.set_xlim(50, 140); ax.set_ylim(-190, -100)
    ax.set_title("%s sagittal MIP x %.0f..%.0f (anterior right)" % (nm, xa, xb))
# enclosure overlay
enc = json.load(open(os.path.join(WORK, "enclosure_ct_seg2.json")))
ax = axs[1, 2]
for nm, c in (("rva", "r"), ("lva", "c")):
    ax.plot(enc[nm]["s"], enc[nm]["enclosure_smooth"], c, label=nm)
    ax.plot(enc[nm]["s"], (np.array(enc[nm]["z"]) + 270) / 200, c + ":", lw=0.8)
ax.set_xticks(np.arange(0, 230, 5)); ax.grid(alpha=0.3); ax.legend(); ax.tick_params(labelsize=7)
ax.set_title("enclosure (solid) and (z+270)/200 (dotted) vs s")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "ccj_views.png"), dpi=50)
print("ok")
