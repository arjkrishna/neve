#!/usr/bin/env python3
"""VMR 0248_H_AOCERE_CAS rendered in the mesh_plus_centerline_hero style.

Kept deliberately separate from the TopBrain anatomies and from the shipped
host: this is a DIFFERENT patient tree, the only public anatomy found that
spans aortic arch to circle of Willis in one piece, and it carries 40% left /
85% right carotid stenosis. Mixing it into either existing folder would imply
it is interchangeable with them, which it is not.

Two figures:
  vmr_0248_hero.png     the tree on its own, mesh plus centerlines
  vmr_0248_vs_host.png  beside the shipped host at IDENTICAL scale, which is
                        the comparison that matters: it shows how much of the
                        corridor each one actually covers

The centerlines here are VMTK root-to-tip paths rather than the segmented
branches the shipped tree has, so they overlap along the aorta. That is a
property of the conversion, not of the anatomy, and it is why they are drawn
in one colour rather than grouped by vessel.

    docker run --rm <mounts> eve-training-fixed
      python3 monitoring/figure_vmr_0248.py <anatomy_dir> <out_dir>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

ANAT = sys.argv[1] if len(sys.argv) > 1 else \
    "/opt/eve_training/vmr_data/anatomies/0248_H_AOCERE_CAS"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/opt/eve_training/results/figs/vmr_0248"
HOST = "/opt/eve_training/eve_bench/data/dualdevicenav"
os.makedirs(OUT, exist_ok=True)

ELEV, AZIM = 16, -68
MESH_C, LINE_C = "#d62728", "#1f77b4"
HOST_MESH_C, HOST_LINE_C = "#7f7f7f", "#2ca02c"


# FromMesh rotates the .obj by this at load time with rotate_branches=False,
# so a raw read leaves mesh and centerlines in different frames. Verified by
# bounding-box overlap: 0.97 on the shipped host and 1.00 on 0248, against
# 0.07 or less for every other candidate rotation including the [-90,0,90]
# constant in create_dualdevicenav_format.py.
ROT_YZX = [90, -90, 0]


def load_mesh(path, target=14000):
    import pyvista as pv
    from eve.intervention.vesseltree.util.meshing import rotate_mesh
    m = pv.read(path)
    m = rotate_mesh(m, ROT_YZX)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    if m.n_cells > target:
        try:
            m = m.triangulate().decimate(1.0 - target / float(m.n_cells))
        except Exception:
            pass
    v = np.asarray(m.points, np.float32)
    ff = np.asarray(m.faces, np.int64).ravel()
    tris, i = [], 0
    while i < len(ff):
        k = int(ff[i])
        if k == 3 and i + 3 < len(ff):
            tris.append(ff[i + 1:i + 4])
        i += k + 1
    return v, (np.asarray(tris, np.int64) if tris else np.zeros((0, 3), np.int64))


def centerlines(folder):
    return [np.asarray(b.coordinates, float) for b in load_branches(folder)]


def draw(ax, verts, faces, lines, mc, lc, alpha=0.22):
    if faces.size:
        ax.add_collection3d(Poly3DCollection(
            verts[faces], facecolor=(*mcolors.to_rgb(mc), alpha),
            edgecolor=(0.25, 0.25, 0.3, 0.05), linewidth=0.1))
    for c in lines:
        if c.size:
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=lc, lw=1.4, alpha=0.9, zorder=6)


def frame(ax, pts, title, rad=None, ctr=None):
    P = np.concatenate(pts, axis=0)
    if ctr is None:
        ctr = (P.min(0) + P.max(0)) / 2.0
    if rad is None:
        rad = float(np.max(P.max(0) - P.min(0))) / 2.0 * 1.05
    for i, s in enumerate("xyz"):
        getattr(ax, "set_%slim" % s)(ctr[i] - rad, ctr[i] + rad)
        getattr(ax, "set_%slabel" % s)("%s (vessel-CS, mm)" % s, fontsize=7, labelpad=-2)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.tick_params(labelsize=6, pad=0)
    ax.set_title(title, fontsize=9)
    return rad, ctr


v0, f0 = load_mesh(os.path.join(ANAT, "vessel_architecture_collision.obj"))
c0 = centerlines(os.path.join(ANAT, "Centrelines_comb"))
tot0 = sum(float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for c in c0)
print("0248: %d tris, %d centerlines, %.0f mm of vessel" % (len(f0), len(c0), tot0), flush=True)

# --- figure 1: on its own
fig = plt.figure(figsize=(8.5, 9.5))
ax = fig.add_subplot(111, projection="3d")
draw(ax, v0, f0, c0, MESH_C, LINE_C)
frame(ax, [v0], "VMR 0248_H_AOCERE_CAS  (40%% left / 85%% right carotid stenosis)\n"
               "aortic arch to circle of Willis, %d vessels" % len(c0))
ax.legend(handles=[Line2D([], [], color=MESH_C, lw=6, alpha=0.6, label="vessel mesh"),
                   Line2D([], [], color=LINE_C, lw=2, label="centerlines (VMTK root-to-tip paths)")],
          loc="upper left", fontsize=7)
fig.tight_layout()
p = os.path.join(OUT, "vmr_0248_hero.png")
fig.savefig(p, dpi=160); plt.close(fig)
print("wrote", os.path.basename(p), flush=True)

# --- figure 2: beside the shipped host at identical scale
vh, fh = load_mesh(os.path.join(HOST, "vessel_architecture_collision.obj"))
ch = centerlines(os.path.join(HOST, "Centrelines_comb"))
allp = np.concatenate([v0, vh], axis=0)
RAD = float(np.max(allp.max(0) - allp.min(0))) / 2.0 * 0.55   # each panel, same size

fig = plt.figure(figsize=(17, 9.5))
for j, (v, f, c, mc, lc, name) in enumerate(
        [(v0, f0, c0, MESH_C, LINE_C, "VMR 0248_H_AOCERE_CAS  (candidate second host)"),
         (vh, fh, ch, HOST_MESH_C, HOST_LINE_C, "shipped host  (all 49 TopBrain anatomies use this)")]):
    ax = fig.add_subplot(1, 2, j + 1, projection="3d")
    draw(ax, v, f, c, mc, lc)
    ext = v.max(0) - v.min(0)
    frame(ax, [v], "%s\nextent %.0f x %.0f x %.0f mm, %d vessels"
          % (name, ext[0], ext[1], ext[2], len(c)), rad=RAD)
fig.suptitle("Identical scale. The shipped host carries a long descending aorta; "
             "0248 starts at the arch.", fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p = os.path.join(OUT, "vmr_0248_vs_host.png")
fig.savefig(p, dpi=160); plt.close(fig)
print("wrote", os.path.basename(p), flush=True)
print("done")
