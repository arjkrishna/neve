#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_fm.py (= /sp/v5_va/vmr_fm.py in the build containers)
"""Zoomed midsagittal slices (3 x positions) around the foramen magnum for picking basion / opisthion."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vmr_common import CT, WORK, load_vmr
vmr = load_vmr(with_ct=True)
ct = CT(vmr["ct"])
y = ct.org[1] + np.arange(ct.dims[1]) * ct.sp[1]
z = ct.org[2] + np.arange(ct.dims[2]) * ct.sp[2]
fig, axs = plt.subplots(1, 3, figsize=(30, 11))
for ax, xm in zip(axs, (-111.0, -107.9, -104.8)):
    i = int(round((xm - ct.org[0]) / ct.sp[0]))
    ax.imshow(ct.arr[:, :, i], cmap="gray", vmin=-100, vmax=1200, origin="lower", extent=[y[0], y[-1], z[0], z[-1]], interpolation="nearest")
    ax.set_xlim(65, 120); ax.set_ylim(-160, -125)
    ax.set_xticks(np.arange(65, 120, 1)); ax.set_yticks(np.arange(-160, -125, 1)); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6)
    ax.set_title("x = %.1f (anterior right)" % xm)
plt.tight_layout()
plt.savefig(os.path.join(WORK, "fm_zoom.png"), dpi=55)
print("ok")
