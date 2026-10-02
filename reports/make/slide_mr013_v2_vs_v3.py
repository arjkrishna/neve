#!/usr/bin/env python3
"""Slide: topcow_mr_013, right ICA, as v2 built it and as v3 built it.

v2 panel: the LEFT half of  saved/figs/topbrain_v2/topbrain_mr_013.png
v3 panel: the LEFT half of  saved/figs/topbrain_v3/topbrain_mr_013.png

Both sources are 2720 x 1520 renders from the same figure script with the same
axes limits and the same camera, so taking the same 1360 x 1520 box out of each
puts the two meshes at exactly the same scale. The shared legend is lifted out
of the v2 half and placed once, at the top left.

    python reports/make/slide_mr013_v2_vs_v3.py
        -> reports/slides_v2/slide_mr013_v2_vs_v3.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "saved", "figs")
OUT = os.path.join(ROOT, "reports", "slides_v2")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})

V2_SRC = os.path.join(FIGS, "topbrain_v2", "topbrain_mr_013.png")
V3_SRC = os.path.join(FIGS, "topbrain_v3", "topbrain_mr_013.png")
C_V2, C_V3 = "#1f77b4", "#2ca02c"


def half(path, side):
    a = np.asarray(Image.open(path).convert("RGB"))
    w = a.shape[1] // 2
    return (a[:, w:] if side == "right" else a[:, :w]).copy()


def ink_box(a, pad=0):
    ink = (a < 250).any(2)
    r = np.where(ink.any(1))[0]
    c = np.where(ink.any(0))[0]
    return [r[0] - pad, r[-1] + pad, c[0] - pad, c[-1] + pad]


def lift_legend(a, y0=95, y1=300, x0=0, x1=760, pad=6):
    """Cut the legend box out of a half and return it; leave white behind."""
    r0, r1, c0, c1 = ink_box(a[y0:y1, x0:x1], pad)
    r0, r1, c0, c1 = y0 + r0, y0 + r1, x0 + c0, x0 + c1
    leg = a[r0:r1, c0:c1].copy()
    a[r0:r1, c0:c1] = 255
    return leg


v2 = half(V2_SRC, "left")
v3 = half(V3_SRC, "left")
legend = lift_legend(v2)
lift_legend(v3)

# one crop box for both, so neither panel is rescaled relative to the other
b1, b2 = ink_box(v2, 12), ink_box(v3, 12)
r0, r1 = min(b1[0], b2[0]), max(b1[1], b2[1])
c0, c1 = min(b1[2], b2[2]), max(b1[3], b2[3])
h, w = v2.shape[:2]
r0, c0 = max(r0, 0), max(c0, 0)
r1, c1 = min(r1, h), min(c1, w)
v2, v3 = v2[r0:r1, c0:c1], v3[r0:r1, c0:c1]

# The render leaves a tall white band between each panel title and its axes.
# Split on that band -- using rows shared by both panels, so both stay at the
# same scale -- and re-stack the title tight above the plot.
rows = (v2 < 250).any(2).any(1) | (v3 < 250).any(2).any(1)
idx = np.where(rows)[0]
gap = np.where(np.diff(idx) > 20)[0][0]
t0, t1, p0 = idx[0] - 4, idx[gap] + 4, idx[gap + 1] - 4
pad = np.full((18, v2.shape[1], 3), 255, np.uint8)
v2 = np.vstack([v2[t0:t1], pad, v2[p0:]])
v3 = np.vstack([v3[t0:t1], pad, v3[p0:]])

fig = plt.figure(figsize=(16, 9), facecolor="white")
fig.text(0.5, 0.955, "topcow_mr_013  (right ICA)  —  the same anatomy under v2 and v3",
         ha="center", fontsize=19, weight="bold", color="#1f4e79")
fig.text(0.285, 0.885, "v2", ha="center", va="center", fontsize=28, weight="bold", color=C_V2)
fig.text(0.735, 0.885, "v3", ha="center", va="center", fontsize=28, weight="bold", color=C_V3)

for x, img, col in ((0.055, v2, C_V2), (0.515, v3, C_V3)):
    ax = fig.add_axes([x, 0.02, 0.44, 0.845])
    ax.imshow(img, interpolation="lanczos", aspect="equal")
    ax.set_axis_off()

lax = fig.add_axes([0.012, 0.855, 0.155, 0.105])
lax.imshow(legend, interpolation="lanczos", aspect="equal")
lax.set_axis_off()

p = os.path.join(OUT, "slide_mr013_v2_vs_v3.png")
fig.savefig(p, dpi=150, facecolor="white")
plt.close(fig)
print("wrote", p, "| panel crop", v2.shape, "legend", legend.shape)
