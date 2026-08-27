#!/usr/bin/env python3
"""MESH version of the run-style anatomy figures (companion to
figure_run_style_anatomy.py, which draws centerlines).

Renders the actual marching-cubes vessel SURFACE — the same mesh geometry
SOFA collides against — instead of centerlines, keeping the run-snapshot
conventions and the requested palette:

    blue   = on-path branches (the RCCA route the policy must navigate)
    green  = RVA (the varied wrong-daughter takeoff)
    red    = every other branch
    orange = planned path      yellow X = target      black star = insertion

Devices (guidewire/catheter) are deliberately omitted.

Each branch GROUP is meshed separately (generate_temp_mesh accepts any
branch subset) so the surfaces can be colored independently. Meshes are
cached per (seed, generation, group) within a run.

Writes mesh_*.png — it never overwrites the centerline runstyle_*.png set.

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_mesh_anatomy.py /results/figs
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
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
EVAL_SEED = BASE_SEED - 1
TARGET_FACES = 9000        # per group, keeps matplotlib responsive

COL = {
    "onpath": ("#1f77b4", 0.42, "on-path branches (RCCA route)"),
    "rva":    ("#2ca02c", 0.42, "RVA (varied takeoff)"),
    "other":  ("#d62728", 0.16, "other branches"),
}

_MESH_CACHE = {}


def build(seed, gens=0, target_seed=0):
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


def path_branch_names(pf):
    names = set()
    try:
        for br in pf.path_branch_set:
            n = getattr(br, "name", None)
            if n:
                names.add(n)
    except Exception:
        pass
    return names


def load_mesh(path, target_faces=TARGET_FACES):
    """(verts, tri_faces) from an .obj, decimated — mirrors snapshot.py."""
    import pyvista as pv
    mesh = pv.read(path)
    if not hasattr(mesh, "faces") or mesh.faces is None:
        mesh = mesh.extract_surface()
    n = mesh.n_cells if hasattr(mesh, "n_cells") else 0
    if n > target_faces:
        try:
            mesh = mesh.triangulate().decimate(1.0 - target_faces / float(n))
        except Exception:
            pass
    verts = np.asarray(mesh.points, dtype=np.float32)
    ff = np.asarray(mesh.faces, dtype=np.int64).reshape(-1)
    tris, i = [], 0
    while i < len(ff):
        k = int(ff[i])
        if k == 3 and i + 3 < len(ff):
            tris.append(ff[i + 1:i + 4])
        i += k + 1
    return verts, (np.asarray(tris, dtype=np.int64) if tris
                   else np.zeros((0, 3), dtype=np.int64))


def group_meshes(itv, pf, key):
    """{group: (verts, faces)} for on-path / RVA / other, cached."""
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    on_path = path_branch_names(pf)
    groups = {"onpath": [], "rva": [], "other": []}
    for br in itv.vessel_tree.branches:
        name = str(getattr(br, "name", ""))
        if name in on_path:
            groups["onpath"].append(br)
        elif "RVA" in name.upper():
            groups["rva"].append(br)
        else:
            groups["other"].append(br)
    out = {}
    for g, brs in groups.items():
        if not brs:
            continue
        t0 = time.time()
        try:
            p = generate_temp_mesh(brs, f"figmesh_{g}", 0.99)
            out[g] = load_mesh(p)
            try:
                os.remove(p)
            except OSError:
                pass
            print(f"    mesh[{g}] {len(brs)} branches, "
                  f"{len(out[g][1])} tris, {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"    mesh[{g}] FAILED: {e}", flush=True)
    _MESH_CACHE[key] = out
    return out


def set_equal_aspect(ax, pts):
    pts = np.asarray(pts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ctr = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


def render(ax, itv, pf, title, key, legend=False, elev=16, azim=-68):
    meshes = group_meshes(itv, pf, key)
    bounds = []
    for g in ("other", "rva", "onpath"):     # far-to-near for alpha blending
        if g not in meshes:
            continue
        verts, faces = meshes[g]
        color, alpha, label = COL[g]
        if faces.size:
            coll = Poly3DCollection(
                verts[faces], facecolor=(*matplotlib.colors.to_rgb(color), alpha),
                edgecolor=(0.25, 0.25, 0.3, 0.06), linewidth=0.15,
            )
            ax.add_collection3d(coll)
        if g != "other":                      # frame on the navigated vessels
            bounds.append(verts)
        if legend:
            ax.plot([], [], color=color, lw=6, alpha=min(1.0, alpha + 0.35),
                    label=label)

    try:
        pp = np.asarray(pf.path_points_vessel_cs)
        if len(pp) > 1:
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00",
                    linewidth=2.6, label="planned path" if legend else None,
                    zorder=6)
            bounds.append(pp)
    except Exception:
        pass

    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=95, marker="X",
               edgecolors="black", linewidths=0.6, zorder=8,
               label="target" if legend else None)
    bounds.append(np.atleast_2d(tgt))
    ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=60, marker="*",
               zorder=8, label="insertion (fixed)" if legend else None)

    if bounds:
        set_equal_aspect(ax, np.concatenate(bounds, axis=0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="upper left", fontsize=7)


# ---------------------------------------------------------------- FIG A
print("FIG A — per-worker anatomies", flush=True)
fig = plt.figure(figsize=(16, 10))
for i in range(6):
    print(f"  worker {i} (seed {BASE_SEED+i})", flush=True)
    itv, pf = build(BASE_SEED + i, target_seed=7)
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    render(ax, itv, pf,
           f"worker {i}  seed {BASE_SEED+i}  ({itv.vessel_tree.mesh_fingerprint})",
           key=("w", BASE_SEED + i, 0), legend=(i == 0))
fig.suptitle("Per-worker varied RCCA anatomy — VESSEL MESH view "
             "(planned path + target; devices omitted)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "mesh_workers.png"), dpi=150)
plt.close(fig)
print("wrote mesh_workers.png", flush=True)

# ---------------------------------------------------------------- FIG B
print("FIG B — within-worker regenerations", flush=True)
fig = plt.figure(figsize=(16, 10))
for g in range(6):
    print(f"  generation {g}", flush=True)
    itv, pf = build(BASE_SEED, gens=g, target_seed=7)
    ax = fig.add_subplot(2, 3, g + 1, projection="3d")
    render(ax, itv, pf,
           f"episodes {g*10}-{g*10+9}   ({itv.vessel_tree.mesh_fingerprint})",
           key=("g", BASE_SEED, g), legend=(g == 0))
fig.suptitle(f"Within-worker regeneration (seed {BASE_SEED}) — "
             "a new RCCA mesh every 10 episodes", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "mesh_generations.png"), dpi=150)
plt.close(fig)
print("wrote mesh_generations.png", flush=True)

# ---------------------------------------------------------------- FIG C
print("FIG C — held-out eval anatomy, 3 target draws", flush=True)
fig = plt.figure(figsize=(16, 6))
for j, ts in enumerate((3, 17, 55)):
    itv, pf = build(EVAL_SEED, target_seed=ts)
    try:
        pp = np.asarray(pf.path_points_vessel_cs)
        plen = float(np.sum(np.linalg.norm(np.diff(pp, axis=0), axis=1)))
    except Exception:
        plen = float("nan")
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    render(ax, itv, pf,
           f"HELD-OUT eval anatomy (seed {EVAL_SEED})\n"
           f"target draw {ts} — planned path {plen:.0f} mm",
           key=("e", EVAL_SEED, 0), legend=(j == 0))
fig.suptitle("Held-out evaluation anatomy — fixed vessel, 98 target draws",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "mesh_eval_targets.png"), dpi=150)
plt.close(fig)
print("wrote mesh_eval_targets.png", flush=True)

# ---------------------------------------------------------------- FIG D
print("FIG D — hero", flush=True)
itv, pf = build(BASE_SEED, target_seed=7)
fig = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(111, projection="3d")
render(ax, itv, pf,
       f"RCCA-varied vessel mesh (seed {BASE_SEED}, "
       f"{itv.vessel_tree.mesh_fingerprint})\n"
       "blue = on-path (RCCA route), green = RVA, red = other branches",
       key=("w", BASE_SEED, 0), legend=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mesh_hero.png"), dpi=160)
plt.close(fig)
print("wrote mesh_hero.png", flush=True)
