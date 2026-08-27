#!/usr/bin/env python3
"""Visual QC of the grafted TopBrain anatomies, two per image.

Same look as mesh_plus_centerline_hero.png: translucent vessel mesh with the
centerlines overlaid. Two anatomies side by side per image, so consecutive
TopBrain patients can be compared directly.

The point of the figure is the graft, so the RCCA centerline is drawn in two
colours: BLUE for the host portion every anatomy shares (0-130 mm) and ORANGE
for the real TopBrain siphon that replaces the old sinusoidal perturbation. A
dot marks the junction. Any kink, doubling back or scale mismatch shows up
there immediately.

ONE shared axis cube is computed across ALL anatomies, so every image in the
series is at identical scale and the differences between patients are real
rather than an artifact of re-framing. Matplotlib 3-D does NOT clip to the
axis limits, so geometry outside the cube (the aortic arch, the contralateral
cerebral vessels) is culled by hand instead of sprawling across the panel.

Only the RCCA branch differs between anatomies (the graft replaces it and
copies every other branch unchanged), so the RVA and neighbour meshes are
generated ONCE and reused: N + 2 meshing calls instead of N x 3.

    docker run --rm <mounts> eve-training-fixed
      python3 /opt/eve_training/monitoring/figure_topbrain_pairs.py
              /opt/eve_training/topbrain_data/anatomies /results/figs/topbrain
"""
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve_bench.dualdevicenav import load_branches
from eve.intervention.vesseltree.util.meshing import generate_temp_mesh

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/eve_training/topbrain_data/anatomies"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/opt/eve_training/results/figs/topbrain"
os.makedirs(OUT, exist_ok=True)

ELEV, AZIM = 16, -68
GRAFT_MM = 130.0          # host arclength where the real TopBrain siphon starts
COL = {"onpath": ("#1f77b4", "host RCCA, 0-%d mm (shared)" % GRAFT_MM),
       "rva":    ("#2ca02c", "RVA"),
       "other":  ("#d62728", "other branches")}
GRAFT_COL = "#ff7f0e"
ALPHA = {"onpath": 0.40, "rva": 0.40, "other": 0.10}


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


def mesh_of(branches, tag):
    p = generate_temp_mesh(branches, tag, 0.99)
    out = load_mesh(p)
    try:
        os.remove(p)
    except OSError:
        pass
    return out


def group(branches):
    g = {"onpath": [], "rva": [], "other": []}
    for b in branches:
        n = str(getattr(b, "name", "")).upper()
        if "RCCA" in n:
            g["onpath"].append(b)
        elif "RVA" in n:
            g["rva"].append(b)
        else:
            g["other"].append(b)
    return g


def arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def inside(p, lims):
    ok = np.ones(len(p), dtype=bool)
    for i, (lo, hi) in enumerate(lims):
        ok &= (p[:, i] >= lo) & (p[:, i] <= hi)
    return ok


def clip_faces(v, f, lims):
    """Drop triangles with any vertex outside the cube: mpl 3-D will not."""
    if f.size == 0:
        return f
    ok = inside(v, lims)
    return f[ok[f].all(axis=1)]


def clip_line(c, lims):
    """NaN out points outside the cube so the polyline breaks there."""
    out = np.asarray(c, float).copy()
    out[~inside(out, lims)] = np.nan
    return out


folders = sorted(glob.glob(os.path.join(ROOT, "*", "Centrelines_comb")))
print("anatomies: %d" % len(folders), flush=True)

cases = []
for f in folders:
    stem = f.split(os.sep)[-2]
    g = group(load_branches(f))
    c = np.asarray(g["onpath"][0].coordinates, float)
    s = arclength(c)
    cases.append({"stem": stem, "g": g, "rcca": c, "s": s, "len": float(s[-1]),
                  "cut": int(np.argmin(np.abs(s - GRAFT_MM)))})

# The RVA is repaired per anatomy in the handful of cases where a siphon would
# otherwise run through it, so it can no longer be meshed once and shared; only
# the untouched neighbour branches can. "other" waits until the box is known,
# so only the branches that survive it get meshed.
REF_RVA = np.asarray(cases[0]["g"]["rva"][0].coordinates, float)


def rva_repaired(case):
    p = np.asarray(case["g"]["rva"][0].coordinates, float)
    return not (len(p) == len(REF_RVA) and np.allclose(p, REF_RVA, atol=1e-6))


shared = {}

# One shared box for the whole series, framed on the navigated route and the
# RVA. Deliberately NOT a cube: everything outside is culled anyway, so a cube
# would just pad the panel with empty space and shrink the siphon. Per-axis
# limits plus a matching box_aspect keep millimetres isotropic.
P = np.concatenate([c["rcca"] for c in cases]
                   + [np.asarray(b.coordinates, float)
                      for c in cases for b in c["g"]["rva"]], axis=0)
lo, hi = P.min(axis=0), P.max(axis=0)
pad = 0.06 * (hi - lo)
LIMS = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(3)]
SPAN = tuple(float(h - l) for l, h in LIMS)
print("shared box %.0f x %.0f x %.0f mm" % SPAN, flush=True)

# Branches that only clip a corner of the box survive as mesh shards floating
# in mid-air, which reads as broken geometry rather than as context. Keep the
# neighbours that actually share the corridor and drop the rest.
NEAR = [b for b in cases[0]["g"]["other"]
        if inside(np.asarray(b.coordinates, float), LIMS).mean() > 0.15]
print("neighbour branches kept: %d of %d"
      % (len(NEAR), len(cases[0]["g"]["other"])), flush=True)
shared["other"] = mesh_of(NEAR, "tb_near") if NEAR else (
    np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))


# A Line3D with no data draws nothing in a legend, so the legend is built from
# plain 2-D proxy artists rather than from the plotted geometry.
HANDLES = [Line2D([], [], color=COL[k][0], lw=6, alpha=0.75, label=COL[k][1])
           for k in ("onpath", "other", "rva")]
HANDLES += [Line2D([], [], color=GRAFT_COL, lw=6, label="real TopBrain siphon"),
            Line2D([], [], color="none", marker="o", markersize=8,
                   markerfacecolor="white", markeredgecolor="k",
                   label="graft junction (130 mm)")]


def draw(ax, case, legend):
    meshes = dict(shared)
    meshes["onpath"] = mesh_of(case["g"]["onpath"], "tb_on_%s" % case["stem"])
    meshes["rva"] = mesh_of(case["g"]["rva"], "tb_rva_%s" % case["stem"])
    for k in ("other", "rva", "onpath"):
        v, f = meshes[k]
        f = clip_faces(v, f, LIMS)
        if f.size:
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=(*mcolors.to_rgb(COL[k][0]), ALPHA[k]),
                edgecolor=(0.25, 0.25, 0.3, 0.05), linewidth=0.12))

    for b in NEAR:
        c = clip_line(b.coordinates, LIMS)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=COL["other"][0],
                alpha=0.45, linewidth=0.9, zorder=4)
    for b in case["g"]["rva"]:
        c = clip_line(b.coordinates, LIMS)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=COL["rva"][0],
                alpha=0.95, linewidth=1.8, zorder=5)

    # the graft itself: host portion vs real siphon, plus the junction
    c, k = case["rcca"], case["cut"]
    ax.plot(c[:k + 1, 0], c[:k + 1, 1], c[:k + 1, 2], color=COL["onpath"][0],
            linewidth=2.2, zorder=6)
    ax.plot(c[k:, 0], c[k:, 1], c[k:, 2], color=GRAFT_COL, linewidth=2.4,
            zorder=7)
    ax.scatter(*c[k], s=42, facecolor="white", edgecolor="k", linewidth=1.2,
               zorder=8)

    ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1]); ax.set_zlim(*LIMS[2])
    # The route is nearly three times taller than it is wide, so the box fits
    # its axes on height alone and leaves the rest empty. zoom fills it back.
    try:
        ax.set_box_aspect(SPAN, zoom=1.45)
    except TypeError:                       # matplotlib without the zoom kwarg
        ax.set_box_aspect(SPAN)
    ax.view_init(elev=ELEV, azim=AZIM)
    # Only z is read off this figure (does the siphon climb, and how far), and
    # the transverse ticks would sit under the legend, so they go.
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_zlabel("z (mm), superior", fontsize=10, labelpad=2)
    ax.tick_params(axis="z", labelsize=8, pad=0)


n = 0
# The box is far taller than it is wide, so the panels are portrait and the
# per-case captions go in the top margin: an axes title would land on top of
# the vessel, since a 3-D axes reserves the same bbox whatever it draws in it.
for i in range(0, len(cases), 2):
    pair = cases[i:i + 2]
    fig = plt.figure(figsize=(12.5, 11))
    for j, case in enumerate(pair):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        draw(ax, case, legend=(j == 0))
        k = case["cut"]
        fig.text(0.30 + 0.44 * j, 0.935,
                 "%s\nroute %.0f mm    siphon %.0f mm%s"
                 % (case["stem"], case["len"], case["len"] - case["s"][k],
                    "\nRVA repaired" if rva_repaired(case) else ""),
                 ha="center", va="top", fontsize=13)
    ids = "_".join(c["stem"].replace("topcow_", "") for c in pair)
    fig.suptitle("Grafted TopBrain anatomies: identical scale, view and host "
                 "tree, bar the RVA where a panel says it was repaired",
                 fontsize=14, y=0.99)
    fig.legend(handles=HANDLES, loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.90, wspace=0.0)
    path = os.path.join(OUT, "topbrain_pair_%02d_%s.png" % (i // 2 + 1, ids))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    n += 1
    print("wrote %s" % os.path.basename(path), flush=True)

print("done: %d images" % n, flush=True)
