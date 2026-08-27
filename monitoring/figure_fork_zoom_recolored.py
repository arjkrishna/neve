#!/usr/bin/env python3
"""The FORK ZOOM panel of varied_rcca_overlay_zoom.png, recolored.

Everything is kept identical to the original second panel of
figure_varied_rcca.py FIG 3 — same 8 anatomies (seeds 12345..12352), same
60 mm arclength window from the shared ostium, same view angle
(elev 18, azim -72), same per-axis focus limits with pad=3, same black star
at the insertion point, same title/label styling.

ONLY the colors change:
    RCCA takeoff  -> different SHADES OF RED, one per anatomy
    RVA entry     -> DOTTED BLUE (the varied proximal window)
    RVA distal    -> a single SOLID GREEN line: beyond ~38 mm every variant
                     is numerically identical (measured max deviation
                     0.00 mm), so the dotted blue entries merge into one RVA

Writes fork_zoom_recolored.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_fork_zoom_recolored.py /results/figs
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve.intervention.vesseltree import RCCAVariedFromMesh
from eve_bench.dualdevicenav import load_branches, DATA_DIR

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

RCCA_NAME = "Centerline curve - RCCA.mrk"
BRANCHES = load_branches(os.path.join(DATA_DIR, "Centrelines_comb"))
ZOOM_MM = 60.0
RVA_MERGE_MM = 38.0     # measured merge point of the RVA variants


def make_tree(seed, gens=0):
    t = RCCAVariedFromMesh(branch_list=BRANCHES, rcca_name=RCCA_NAME,
                           episodes_between_change=10, seed=seed)
    for _ in range(gens):
        t._generate()
    return t


def get(tree, key):
    for b in tree.branches:
        if key in str(b.name).upper():
            return np.asarray(b.coordinates, dtype=float)
    return None


def focus_limits(ax, arrs, pad=3):
    """Verbatim from figure_varied_rcca.py."""
    P = np.vstack(arrs)
    for setter, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(P[:, i].min() - pad, P[:, i].max() + pad)


def style_ax(ax, title, elev=18, azim=-72):
    """Verbatim from figure_varied_rcca.py."""
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-4)
    ax.set_ylabel("y (mm)", fontsize=7, labelpad=-4)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)


pool = [make_tree(12345 + i) for i in range(8)]
ins = np.asarray(pool[0].insertion.position, dtype=float)

REDS = plt.cm.Reds(np.linspace(0.45, 1.0, len(pool)))   # RCCA takeoff shades
BLUE = "#1f6fd0"

fig = plt.figure(figsize=(6.8, 6.2))
ax2 = fig.add_subplot(111, projection="3d")
zarrs = []

for k, t in enumerate(pool):
    # --- RCCA takeoff: shades of red (was: one plasma color, solid)
    a = get(t, "RCCA")
    if a is not None:
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a, axis=0),
                                                          axis=1))])
        m = d <= ZOOM_MM
        ax2.plot(a[m, 0], a[m, 1], a[m, 2], color=REDS[k], lw=2.0,
                 label="RCCA takeoff (shade per anatomy)" if k == 0 else None)
        zarrs.append(a[m])
    # --- RVA entry: dotted blue over the varied window (was: dashed plasma)
    a = get(t, "RVA")
    if a is not None:
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a, axis=0),
                                                          axis=1))])
        m = d <= ZOOM_MM
        mv = d <= RVA_MERGE_MM
        ax2.plot(a[mv, 0], a[mv, 1], a[mv, 2], color=BLUE, lw=2.0, ls=":",
                 label="RVA entry — varied (dotted)" if k == 0 else None)
        zarrs.append(a[m])
        if k == 0:                      # merged distal RVA — identical for all
            md = (d >= RVA_MERGE_MM) & (d <= ZOOM_MM)
            ax2.plot(a[md, 0], a[md, 1], a[md, 2], color="#2ca02c", lw=3.0,
                     solid_capstyle="round",
                     label="RVA distal — identical in all")

ax2.scatter(*ins, s=110, marker="*", color="k", zorder=6)
focus_limits(ax2, zarrs, pad=3)
style_ax(ax2, "FORK ZOOM — first 60 mm: RCCA takeoff and RVA entry vary\n"
              "(shared ostium point pinned; both daughters move)")
ax2.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fork_zoom_recolored.png"), dpi=170)
plt.close(fig)
print("wrote fork_zoom_recolored.png", flush=True)
