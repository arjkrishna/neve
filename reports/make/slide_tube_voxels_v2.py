#!/usr/bin/env python3
"""Slide: one vessel, two meshers, voxel by voxel.

A straight vessel of known radius is put through the v1 construction
(paint 0/1 voxels -> Gaussian blur twice -> trace the 0.5 contour) and the v2
construction (signed distance on the same voxels -> trace the 0 contour). The
numbers printed in the cells are the actual field values; the contours are
what marching squares finds, which is the 2-D case of marching cubes.

Nothing is faked: the blur is scipy's Gaussian at sigma = 1 voxel applied
twice, on the same 0.6 mm voxels the v1 mesher uses; for a long straight
vessel the mid-slice of the 3-D result is exactly this 2-D picture.

    python reports/make/slide_tube_voxels_v2.py -> reports/slides_v2/slide_tube_v1_vs_v2.png
                                                   reports/slides_v2/slide_tube_vanishing.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage
from skimage import measure

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slides_v2")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

SPACING = 0.6            # v1's in-plane voxel size, mm
HALF = 9                 # voxels each side of the centre shown
# The vessel axis sits 0.2 mm off a voxel centre in x and y, exactly where the
# probe's test tubes landed on the v1 grid, so the numbers here reproduce the
# measured table (r 1.2 -> 0.66, 1.6 -> 1.11, 1.0 absent). Put the axis ON a
# centre and a 1.2 mm vessel becomes a speck: only 9 voxels get painted and
# the blurred peak is 0.505. v1's output depends on where the centerline
# happens to fall relative to the grid -- another thing v2 does not do.
OFFSET = (0.2, 0.2)
C_V1, C_V2, C_TRUE = "#c0392b", "#1f77b4", "#222222"


def fields(r, spacing=SPACING, half=HALF):
    """Voxel-centre grid, the v1 blurred field and the v2 signed distance."""
    idx = np.arange(-half, half + 1)
    X, Y = np.meshgrid(idx * spacing - OFFSET[0], idx * spacing - OFFSET[1], indexing="xy")
    rho = np.hypot(X, Y)
    binary = (rho < r).astype(float)                     # v1 step 1: voxel centre inside the sphere
    blurred = ndimage.gaussian_filter(ndimage.gaussian_filter(binary, 1.0), 1.0)   # v1 step 2, sigma = 1 voxel, twice
    sdf = r - rho                                        # v2: signed distance, positive inside
    return X, Y, binary, blurred, sdf


def contour(field, level, spacing=SPACING, half=HALF):
    """Marching squares on the voxel grid; returns polylines in mm."""
    out = []
    for c in measure.find_contours(field, level):
        out.append(np.c_[(c[:, 1] - half) * spacing - OFFSET[0], (c[:, 0] - half) * spacing - OFFSET[1]])
    return out


def inscribed(polys):
    if not polys:
        return None
    return min(float(np.hypot(p[:, 0], p[:, 1]).min()) for p in polys)


def draw_cells(ax, X, Y, field, cmap, vmin, vmax, fmt="%.2f", txt_thr=None, spacing=SPACING):
    ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
              extent=[X.min() - spacing / 2, X.max() + spacing / 2, Y.min() - spacing / 2, Y.max() + spacing / 2],
              interpolation="nearest")
    # grid lines and values
    for v in np.arange(X.min() - spacing / 2, X.max() + spacing, spacing):
        ax.axvline(v, color="white", lw=0.5, alpha=0.6); ax.axhline(v, color="white", lw=0.5, alpha=0.6)
    lim = 3.1
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            x, y = X[i, j], Y[i, j]
            if abs(x) > lim or abs(y) > lim:
                continue
            val = field[i, j]
            if txt_thr is not None and abs(val) < txt_thr:
                continue
            dark = (val - vmin) / (vmax - vmin) > 0.55 if cmap != "RdBu" else abs(val) > 0.6 * max(abs(vmin), abs(vmax))
            ax.text(x, y, fmt % val, ha="center", va="center", fontsize=6.2, color="white" if dark else "#222222")
    ax.set_xlim(-lim - spacing / 2, lim + spacing / 2); ax.set_ylim(-lim - spacing / 2, lim + spacing / 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def true_circle(ax, r):
    ax.add_patch(Circle((0, 0), r, fc="none", ec=C_TRUE, lw=1.6, ls="--", zorder=6))
    ax.plot(0, 0, "o", color=C_TRUE, ms=3, zorder=7)


def slide_main(r=1.6):
    X, Y, binary, blurred, sdf = fields(r)
    c1 = contour(blurred, 0.5); c2 = contour(sdf, 0.0)
    r1, r2 = inscribed(c1), inscribed(c2)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.955, "One vessel, two meshers — the same %.1f mm vessel on the same %.1f mm voxels, cell by cell" % (r, SPACING),
             ha="center", fontsize=18, weight="bold", color="#1f4e79")

    gs = fig.add_gridspec(2, 3, left=0.05, right=0.97, top=0.86, bottom=0.10, wspace=0.10, hspace=0.32)
    # ---- row 1: v1
    ax = fig.add_subplot(gs[0, 0]); draw_cells(ax, X, Y, binary, "Greys", 0, 1, fmt="%.0f"); true_circle(ax, r)
    ax.set_title("v1 step 1 — paint voxels\ncentre inside the sphere → 1, else 0", fontsize=11)
    ax = fig.add_subplot(gs[0, 1]); draw_cells(ax, X, Y, blurred, "Reds", 0, 1, txt_thr=0.005); true_circle(ax, r)
    ax.set_title("v1 step 2 — Gaussian blur, σ = 1 voxel, applied twice\nthe 1s leak outward, the edge spreads over ~2 voxels", fontsize=11)
    ax = fig.add_subplot(gs[0, 2]); draw_cells(ax, X, Y, blurred, "Reds", 0, 1, txt_thr=0.005); true_circle(ax, r)
    for p in c1:
        ax.plot(p[:, 0], p[:, 1], color=C_V1, lw=3)
    ax.set_title("v1 step 3 — trace the 0.5 contour\nmeshed radius %.2f mm: %.2f mm of the vessel lost" % (r1, r - r1), fontsize=11, color=C_V1)
    # ---- row 2: v2
    ax = fig.add_subplot(gs[1, 0]); draw_cells(ax, X, Y, sdf, "RdBu", -2.4, 2.4); true_circle(ax, r)
    ax.set_title("v2 step 1 — store the signed distance to the wall\n+ inside, − outside; the wall is where it is 0", fontsize=11)
    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
    ax.text(0.5, 0.62, "no blur step", ha="center", va="center", fontsize=20, color="#999999", weight="bold")
    ax.text(0.5, 0.42, "a distance is already smooth\nand already exact — there is\nnothing to clean up", ha="center", va="center", fontsize=11, color="#777777")
    ax = fig.add_subplot(gs[1, 2]); draw_cells(ax, X, Y, sdf, "RdBu", -2.4, 2.4); true_circle(ax, r)
    for p in c2:
        ax.plot(p[:, 0], p[:, 1], color=C_V2, lw=3)
    ax.set_title("v2 step 2 — trace the 0 contour\nmeshed radius %.2f mm: within %.2f mm of the truth" % (r2, r - r2), fontsize=11, color=C_V2)

    p = os.path.join(OUT, "slide_tube_v1_vs_v2.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig); print("wrote", p, "| v1 %.2f  v2 %.2f" % (r1, r2))


def slide_vanish(radii=(1.6, 1.2, 1.0, 0.8)):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.955, "Why thin vessels vanish in v1 — and stay in v2", ha="center", fontsize=18, weight="bold", color="#1f4e79")
    gs = fig.add_gridspec(2, len(radii), left=0.06, right=0.98, top=0.83, bottom=0.15, wspace=0.08, hspace=0.36)
    for j, r in enumerate(radii):
        X, Y, binary, blurred, sdf = fields(r)
        c1 = contour(blurred, 0.5); c2 = contour(sdf, 0.0)
        r1, r2 = inscribed(c1), inscribed(c2)
        ax = fig.add_subplot(gs[0, j]); draw_cells(ax, X, Y, blurred, "Reds", 0, 1, txt_thr=0.005); true_circle(ax, r)
        for p in c1:
            ax.plot(p[:, 0], p[:, 1], color=C_V1, lw=3)
        peak = blurred.max()
        if r1 is None:
            ax.set_title("r = %.1f mm\npeak %.2f < 0.5: no contour\n→ vessel ABSENT" % (r, peak), fontsize=10.5, color=C_V1, weight="bold")
            ax.text(0, 0, "absent", ha="center", va="center", fontsize=16, color=C_V1, weight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec=C_V1, alpha=0.9))
        else:
            ax.set_title("r = %.1f mm\npeak %.2f → meshed %.2f mm\n(%.2f mm lost)" % (r, peak, r1, r - r1), fontsize=10.5, color=C_V1)
        ax = fig.add_subplot(gs[1, j]); draw_cells(ax, X, Y, sdf, "RdBu", -2.4, 2.4); true_circle(ax, r)
        for p in c2:
            ax.plot(p[:, 0], p[:, 1], color=C_V2, lw=3)
        ax.set_title("meshed %.2f mm (−%.2f)" % (r2, r - r2), fontsize=10.5, color=C_V2)
    fig.text(0.025, 0.66, "v1  (blurred 0/1)", fontsize=12, weight="bold", color=C_V1, ha="center", va="center", rotation=90)
    fig.text(0.025, 0.31, "v2  (signed distance)", fontsize=12, weight="bold", color=C_V2, ha="center", va="center", rotation=90)
    p = os.path.join(OUT, "slide_tube_vanishing.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    slide_main(1.6)
    slide_vanish((2.0, 1.6, 1.2, 1.0))
