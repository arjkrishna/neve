#!/usr/bin/env python3
"""Single-pane side-by-side hero: CENTERLINES (left) vs MESH+centerlines
(right) for the SAME generated vasculature, with a SHARED axis cube and
view angle so the correspondence is exact.

The two panels reproduce runstyle_hero.png and mesh_plus_centerline_hero.png
(same seed/generation/target), but here the axis limits are computed ONCE
from the union of both panels' geometry and applied to both — the previous
standalone heroes each framed themselves, which made their scales differ
slightly.

Writes hero_centerline_vs_mesh.png (new file; overwrites nothing else).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_hero_side_by_side.py /results/figs
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/opt/eve_training/eve_bench")

import eve
from eve.util.coordtransform import tracking3d_to_vessel_cs
from eve.intervention.vesseltree.util.meshing import generate_temp_mesh
from eve_bench import DualDeviceNavRCCAVaried

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
SEED, GEN, TARGET_SEED = 12345, 0, 7
ELEV, AZIM = 16, -68

COL = {
    "onpath": ("#1f77b4", "on-path branches (RCCA route)"),
    "rva":    ("#2ca02c", "RVA (varied takeoff)"),
    "other":  ("#d62728", "other branches"),
}
MESH_ALPHA = {"onpath": 0.45, "rva": 0.45, "other": 0.14}

# ---------------------------------------------------------------- build once
itv = DualDeviceNavRCCAVaried(seed=SEED, episodes_between_change=10)
for _ in range(GEN):
    itv.vessel_tree._generate()
itv.vessel_tree.reset(episode_nr=0)
itv.target.reset(episode_nr=0, seed=TARGET_SEED)
pf = eve.pathfinder.FixedPathfinder(intervention=itv)
try:
    pf.reset(episode_nr=0)
except Exception:
    pass

on_path = set()
try:
    for br in pf.path_branch_set:
        n = getattr(br, "name", None)
        if n:
            on_path.add(n)
except Exception:
    pass

groups = {"onpath": [], "rva": [], "other": []}
for br in itv.vessel_tree.branches:
    name = str(getattr(br, "name", ""))
    if name in on_path:
        groups["onpath"].append(br)
    elif "RVA" in name.upper():
        groups["rva"].append(br)
    else:
        groups["other"].append(br)


def load_mesh(path, target_faces=9000):
    import pyvista as pv
    m = pv.read(path)
    if not hasattr(m, "faces") or m.faces is None:
        m = m.extract_surface()
    n = m.n_cells if hasattr(m, "n_cells") else 0
    if n > target_faces:
        try:
            m = m.triangulate().decimate(1.0 - target_faces / float(n))
        except Exception:
            pass
    v = np.asarray(m.points, dtype=np.float32)
    ff = np.asarray(m.faces, dtype=np.int64).reshape(-1)
    tris, i = [], 0
    while i < len(ff):
        k = int(ff[i])
        if k == 3 and i + 3 < len(ff):
            tris.append(ff[i + 1:i + 4])
        i += k + 1
    return v, (np.asarray(tris, dtype=np.int64) if tris
               else np.zeros((0, 3), dtype=np.int64))


meshes = {}
for g, brs in groups.items():
    if not brs:
        continue
    t0 = time.time()
    p = generate_temp_mesh(brs, f"hero_{g}", 0.99)
    meshes[g] = load_mesh(p)
    try:
        os.remove(p)
    except OSError:
        pass
    print(f"  mesh[{g}] {len(meshes[g][1])} tris {time.time()-t0:.1f}s", flush=True)

# shared extras
try:
    pp = np.asarray(pf.path_points_vessel_cs)
except Exception:
    pp = None
tgt = tracking3d_to_vessel_cs(
    np.asarray(itv.target.coordinates3d, dtype=float),
    itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)

# ------------------------------------------------- ONE shared axis cube
frame_pts = []
for g in ("onpath", "rva"):                    # frame on the navigated vessels
    if g in meshes:
        frame_pts.append(meshes[g][0])         # mesh verts
    for br in groups[g]:                       # and centerline points
        c = np.asarray(br.coordinates)
        if c.size:
            frame_pts.append(c)
if pp is not None and len(pp) > 1:
    frame_pts.append(pp)
frame_pts.append(np.atleast_2d(tgt))
frame_pts.append(np.atleast_2d(ins))
P = np.concatenate(frame_pts, axis=0)
mins, maxs = P.min(axis=0), P.max(axis=0)
ctr = (mins + maxs) / 2.0
RAD = float(np.max(maxs - mins)) / 2.0 * 1.05
LIMS = [(ctr[i] - RAD, ctr[i] + RAD) for i in range(3)]
print(f"  shared cube: center {ctr.round(1)}, half-size {RAD:.1f} mm", flush=True)


def add_extras(ax, legend):
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00", linewidth=2.8,
                zorder=6, label="planned path" if legend else None)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=110, marker="X",
               edgecolors="black", linewidths=0.7, zorder=8,
               label="target" if legend else None)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=70, marker="*",
               zorder=8, label="insertion (fixed)" if legend else None)


def style(ax, title):
    ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1]); ax.set_zlim(*LIMS[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=8, labelpad=0)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=8, labelpad=0)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=8, labelpad=0)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, pad=8)


fig = plt.figure(figsize=(16, 8.5))

# ---------------- LEFT: centerlines only
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
for g in ("other", "rva", "onpath"):
    color, label = COL[g]
    for k, br in enumerate(groups[g]):
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        ax1.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                 alpha=0.55 if g == "other" else 0.92,
                 linewidth=1.0 if g == "other" else 2.0,
                 label=(label if k == 0 else None))
add_extras(ax1, legend=True)
style(ax1, "Centerlines\n(what the pathfinder and observations use)")
ax1.legend(loc="upper left", fontsize=8)

# ---------------- RIGHT: mesh + centerlines overlaid
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for g in ("other", "rva", "onpath"):
    if g not in meshes:
        continue
    v, f = meshes[g]
    color, label = COL[g]
    if f.size:
        ax2.add_collection3d(Poly3DCollection(
            v[f], facecolor=(*mcolors.to_rgb(color), MESH_ALPHA[g]),
            edgecolor=(0.25, 0.25, 0.3, 0.06), linewidth=0.15))
    ax2.plot([], [], color=color, lw=6, alpha=0.75, label=label)
for g in ("other", "rva", "onpath"):
    color, _ = COL[g]
    for br in groups[g]:
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        ax2.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                 alpha=0.95 if g != "other" else 0.5,
                 linewidth=1.7 if g != "other" else 0.9, zorder=5)
ax2.plot([], [], color="#555555", lw=1.7, label="centerlines (overlaid)")
add_extras(ax2, legend=True)
style(ax2, "Vessel mesh + centerlines\n(what SOFA simulates and collides against)")
ax2.legend(loc="upper left", fontsize=8)

fig.suptitle(
    f"Same generated vasculature, two representations — seed {SEED} "
    f"({itv.vessel_tree.mesh_fingerprint}), identical scale and view\n"
    "blue = on-path RCCA route,  green = RVA,  red = other branches,  "
    "orange = planned path,  X = target,  star = fixed insertion",
    fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(OUT, "hero_centerline_vs_mesh.png"), dpi=160)
plt.close(fig)
print("wrote hero_centerline_vs_mesh.png", flush=True)
