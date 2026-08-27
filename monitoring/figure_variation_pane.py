#!/usr/bin/env python3
"""One pane, four hand-picked varied vasculatures (CENTERLINE view), shared scale.

Selected anatomies (mixed across workers and within-worker regenerations —
deliberately not distinguishing eval vs train here; the point is structural
variety):

    s12344g1, s12349g1, s12347g1

A fingerprint sNNNNNgK means seed NNNNN after K calls to _generate(); the
constructor already performs one, so K-1 extra regenerations are applied.
All four panels share ONE axis cube and view angle so differences in course,
tortuosity and calibre are read directly rather than through re-framing.

Writes variation_three_centerlines.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_variation_pane.py /results/figs
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
ELEV, AZIM = 16, -68
TARGET_SEED = 7

# (seed, generation-in-fingerprint)
PICKS = [(12344, 1), (12349, 1), (12347, 1)]

COL = {
    "onpath": ("#1f77b4", "on-path branches (RCCA route)"),
    "rva":    ("#2ca02c", "RVA (varied takeoff)"),
    "other":  ("#d62728", "other branches"),
}
MESH_ALPHA = {"onpath": 0.75, "rva": 0.75, "other": 0.10}
# Keep more triangles on the navigated vessels so the tube surface (and the
# calibre variation) is legible; the arch can stay coarse.
FACE_BUDGET = {"onpath": 40000, "rva": 20000, "other": 6000}


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


def build(seed, gen):
    """Tree whose fingerprint is s{seed}g{gen} (constructor does gen 1)."""
    itv = DualDeviceNavRCCAVaried(seed=seed, episodes_between_change=10)
    for _ in range(max(0, gen - 1)):
        itv.vessel_tree._generate()
    itv.vessel_tree.reset(episode_nr=0)
    itv.target.reset(episode_nr=0, seed=TARGET_SEED)
    pf = eve.pathfinder.FixedPathfinder(intervention=itv)
    try:
        pf.reset(episode_nr=0)
    except Exception:
        pass
    return itv, pf


panels = []
for seed, gen in PICKS:
    print(f"building s{seed}g{gen}", flush=True)
    itv, pf = build(seed, gen)
    fp = itv.vessel_tree.mesh_fingerprint
    if fp != f"s{seed}g{gen}":
        print(f"  WARNING fingerprint is {fp}, expected s{seed}g{gen}", flush=True)

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

    # CENTERLINE mode — no meshing (the pane shows centerlines, per request).
    meshes = {}

    try:
        pp = np.asarray(pf.path_points_vessel_cs)
        plen = float(np.sum(np.linalg.norm(np.diff(pp, axis=0), axis=1)))
    except Exception:
        pp, plen = None, float("nan")
    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
    ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)
    # RCCA centerline length — a one-number summary of the variation
    rc = None
    for br in itv.vessel_tree.branches:
        if "RCCA" in str(getattr(br, "name", "")).upper():
            rc = np.asarray(br.coordinates)
            break
    rlen = (float(np.sum(np.linalg.norm(np.diff(rc, axis=0), axis=1)))
            if rc is not None else float("nan"))
    panels.append(dict(fp=fp, meshes=meshes, groups=groups, pp=pp, tgt=tgt,
                       ins=ins, plen=plen, rlen=rlen))

# ------------------------------------------------- one shared axis cube
pts = []
for p in panels:
    for g in ("onpath", "rva"):
        for br in p["groups"][g]:
            c = np.asarray(br.coordinates)
            if c.size:
                pts.append(c)
    if p["pp"] is not None and len(p["pp"]) > 1:
        pts.append(p["pp"])
    pts.append(np.atleast_2d(p["tgt"]))
    pts.append(np.atleast_2d(p["ins"]))
P = np.concatenate(pts, axis=0)
mins, maxs = P.min(axis=0), P.max(axis=0)
ctr = (mins + maxs) / 2.0
# Per-axis limits with a small margin — NOT a cube. The RCCA is tall and
# thin (~200 mm in z, ~60 mm in x/y); forcing a cube left most of the frame
# empty and rendered the 4-6 mm lumen only a few pixels wide, so the mesh
# read as a centerline. Equal SCALE is preserved by setting box_aspect to
# the data ranges, so geometry is still undistorted.
# Same framing as runstyle_eval_targets.png: an equal-sided cube around the
# navigated vessels, with matplotlib's default 3d box aspect (no
# set_box_aspect call) — that combination gave the proportions we want.
RAD = float(np.max(maxs - mins)) / 2.0 * 1.05
LIMS = [(ctr[i] - RAD, ctr[i] + RAD) for i in range(3)]
BOX_ASPECT = None
print(f"shared cube center {ctr.round(1)} half-size {RAD:.1f} mm", flush=True)

fig = plt.figure(figsize=(16, 6))
for i, p in enumerate(panels):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    for g in ("other", "rva", "onpath"):
        color, label = COL[g]
        for k, br in enumerate(p["groups"][g]):
            c = np.asarray(br.coordinates)
            if c.size == 0:
                continue
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                    alpha=0.5 if g == "other" else 0.95,
                    linewidth=1.0 if g == "other" else 2.4,
                    label=(label if (i == 0 and k == 0) else None))
    if p["pp"] is not None and len(p["pp"]) > 1:
        ax.plot(p["pp"][:, 0], p["pp"][:, 1], p["pp"][:, 2], color="#ffaa00",
                lw=2.8, zorder=6, label="planned path" if i == 0 else None)
    ax.scatter(*p["tgt"], color="#ffd400", s=110, marker="X",
               edgecolors="black", linewidths=0.7, zorder=8,
               label="target" if i == 0 else None)
    ax.scatter(*p["ins"], color="black", s=70, marker="*", zorder=8,
               label="insertion (fixed)" if i == 0 else None)

    ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1]); ax.set_zlim(*LIMS[2])
    if BOX_ASPECT is not None:
        try:
            ax.set_box_aspect(BOX_ASPECT)
        except Exception:
            pass
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=8, labelpad=-1)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=8, labelpad=-1)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=8, labelpad=-1)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{p['fp']}\nRCCA centerline {p['rlen']:.0f} mm   "
                 f"planned path {p['plen']:.0f} mm", fontsize=11, pad=6)
    if i == 0:
        ax.legend(loc="upper left", fontsize=8)

fig.suptitle(
    "Procedurally varied RCCA vasculature — three generated anatomies at identical scale and view\n"
    "blue = on-path RCCA route,  green = RVA,  red = other branches,  "
    "orange = planned path,  X = target,  star = fixed insertion",
    fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(OUT, "variation_three_centerlines.png"), dpi=160)
plt.close(fig)
print("wrote variation_three_centerlines.png", flush=True)
