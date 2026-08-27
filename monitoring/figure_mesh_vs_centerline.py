#!/usr/bin/env python3
"""Paired MESH vs CENTERLINE anatomy figure.

Same vessels rendered both ways with IDENTICAL view angle and axis limits so
the two representations are directly comparable:

    row 1 : marching-cubes vessel SURFACE (what SOFA collides against)
    row 2 : the CENTERLINE counterpart (what the pathfinder/obs use)

Palette (both rows): blue = on-path (RCCA route), green = RVA,
red = other branches, orange = planned path, yellow X = target,
black star = fixed insertion. Devices omitted.

Writes mesh_vs_centerline_workers.png and mesh_plus_centerline_hero.png —
never overwrites the existing runstyle_*.png / mesh_*.png sets.

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_mesh_vs_centerline.py /results/figs
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
BASE_SEED = 12345
N_WORKERS = 4
ELEV, AZIM = 16, -68

COL = {
    "onpath": ("#1f77b4", "on-path branches (RCCA route)"),
    "rva":    ("#2ca02c", "RVA (varied takeoff)"),
    "other":  ("#d62728", "other branches"),
}
MESH_ALPHA = {"onpath": 0.45, "rva": 0.45, "other": 0.14}


def build(seed, gens=0, target_seed=7):
    itv = DualDeviceNavRCCAVaried(seed=seed, episodes_between_change=10)
    for _ in range(gens):
        itv.vessel_tree._generate()
    itv.vessel_tree.reset(episode_nr=0)
    itv.target.reset(episode_nr=0, seed=target_seed)
    pf = eve.pathfinder.FixedPathfinder(intervention=itv)
    try:
        pf.reset(episode_nr=0)
    except Exception:
        pass
    return itv, pf


def path_names(pf):
    out = set()
    try:
        for br in pf.path_branch_set:
            n = getattr(br, "name", None)
            if n:
                out.add(n)
    except Exception:
        pass
    return out


def split_groups(itv, pf):
    on_path = path_names(pf)
    g = {"onpath": [], "rva": [], "other": []}
    for br in itv.vessel_tree.branches:
        name = str(getattr(br, "name", ""))
        if name in on_path:
            g["onpath"].append(br)
        elif "RVA" in name.upper():
            g["rva"].append(br)
        else:
            g["other"].append(br)
    return g


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


def meshes_for(groups):
    out = {}
    for g, brs in groups.items():
        if not brs:
            continue
        t0 = time.time()
        try:
            p = generate_temp_mesh(brs, f"pair_{g}", 0.99)
            out[g] = load_mesh(p)
            try:
                os.remove(p)
            except OSError:
                pass
            print(f"    mesh[{g}] {len(out[g][1])} tris {time.time()-t0:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"    mesh[{g}] FAILED {e}", flush=True)
    return out


def common_extras(ax, itv, pf, legend):
    """Planned path + target + insertion; returns points for bounds."""
    pts = []
    try:
        pp = np.asarray(pf.path_points_vessel_cs)
        if len(pp) > 1:
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00",
                    linewidth=2.6, zorder=6,
                    label="planned path" if legend else None)
            pts.append(pp)
    except Exception:
        pass
    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=95, marker="X",
               edgecolors="black", linewidths=0.6, zorder=8,
               label="target" if legend else None)
    pts.append(np.atleast_2d(tgt))
    ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=60, marker="*",
               zorder=8, label="insertion (fixed)" if legend else None)
    return pts


def finish(ax, title, bounds_pts):
    P = np.concatenate(bounds_pts, axis=0)
    mins, maxs = P.min(axis=0), P.max(axis=0)
    ctr = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    ax.set_title(title, fontsize=9)


def draw_mesh(ax, meshes, legend=False):
    frame = []
    for g in ("other", "rva", "onpath"):
        if g not in meshes:
            continue
        v, f = meshes[g]
        color, label = COL[g]
        if f.size:
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=(*mcolors.to_rgb(color), MESH_ALPHA[g]),
                edgecolor=(0.25, 0.25, 0.3, 0.06), linewidth=0.15))
        if g != "other":
            frame.append(v)
        if legend:
            ax.plot([], [], color=color, lw=6, alpha=0.75, label=label)
    return frame


def draw_centerlines(ax, groups, legend=False):
    frame = []
    for g in ("other", "rva", "onpath"):
        color, label = COL[g]
        for k, br in enumerate(groups[g]):
            c = np.asarray(br.coordinates)
            if c.size == 0:
                continue
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                    alpha=0.55 if g == "other" else 0.9,
                    linewidth=1.0 if g == "other" else 1.8,
                    label=(label if (legend and k == 0) else None))
            if g != "other":
                frame.append(c)
    return frame


# ================================================================ FIG 1
print("paired mesh / centerline figure", flush=True)
fig = plt.figure(figsize=(4.6 * N_WORKERS, 9.5))
for i in range(N_WORKERS):
    seed = BASE_SEED + i
    print(f"  worker {i} (seed {seed})", flush=True)
    itv, pf = build(seed)
    groups = split_groups(itv, pf)
    meshes = meshes_for(groups)
    fp = itv.vessel_tree.mesh_fingerprint

    # --- row 1: mesh
    ax1 = fig.add_subplot(2, N_WORKERS, i + 1, projection="3d")
    frame = draw_mesh(ax1, meshes, legend=(i == 0))
    frame += common_extras(ax1, itv, pf, legend=(i == 0))
    finish(ax1, f"MESH — worker {i}, seed {seed} ({fp})", frame)
    if i == 0:
        ax1.legend(loc="upper left", fontsize=7)
    lims = (ax1.get_xlim(), ax1.get_ylim(), ax1.get_zlim())

    # --- row 2: centerlines, IDENTICAL framing
    ax2 = fig.add_subplot(2, N_WORKERS, N_WORKERS + i + 1, projection="3d")
    draw_centerlines(ax2, groups, legend=(i == 0))
    common_extras(ax2, itv, pf, legend=False)
    finish(ax2, f"CENTERLINES — worker {i}, seed {seed} ({fp})",
           [np.array([[lims[0][0], lims[1][0], lims[2][0]],
                      [lims[0][1], lims[1][1], lims[2][1]]])])
    ax2.set_xlim(*lims[0]); ax2.set_ylim(*lims[1]); ax2.set_zlim(*lims[2])
    if i == 0:
        ax2.legend(loc="upper left", fontsize=7)

fig.suptitle(
    "Varied RCCA anatomy per worker — vessel MESH (top) and its CENTERLINE "
    "counterpart (bottom), identical view and scale\n"
    "blue = on-path RCCA route,  green = RVA,  red = other branches,  "
    "orange = planned path,  X = target,  star = fixed insertion",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "mesh_vs_centerline_workers.png"), dpi=150)
plt.close(fig)
print("wrote mesh_vs_centerline_workers.png", flush=True)

# ================================================================ FIG 2
# Single hero: mesh AND centerlines superimposed in one panel.
print("overlay hero", flush=True)
itv, pf = build(BASE_SEED)
groups = split_groups(itv, pf)
meshes = meshes_for(groups)
fig = plt.figure(figsize=(8.5, 9.5))
ax = fig.add_subplot(111, projection="3d")
frame = draw_mesh(ax, meshes, legend=True)
# centerlines on top of the translucent surface
for g in ("other", "rva", "onpath"):
    color, _ = COL[g]
    for br in groups[g]:
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                alpha=0.95 if g != "other" else 0.5,
                linewidth=1.7 if g != "other" else 0.9, zorder=5)
ax.plot([], [], color="#555555", lw=1.7, label="centerlines (overlaid)")
frame += common_extras(ax, itv, pf, legend=True)
finish(ax, f"Vessel mesh + centerlines overlaid — seed {BASE_SEED} "
           f"({itv.vessel_tree.mesh_fingerprint})", frame)
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mesh_plus_centerline_hero.png"), dpi=160)
plt.close(fig)
print("wrote mesh_plus_centerline_hero.png", flush=True)
