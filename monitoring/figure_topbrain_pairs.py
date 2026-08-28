#!/usr/bin/env python3
"""Visual QC of the grafted TopBrain anatomies, two per image.

Framing follows mesh_plus_centerline_hero.png exactly: a CUBE sized from the
on-path route and the RVA only, so the aortic arch and the other branches run
out past the frame the way they do in the original. Nothing is clipped and no
branch is filtered out; an earlier revision of this script tightened the box
and culled geometry to the axis limits, which made the panels tall and cut the
arch off, and that is not the house style.

The one addition to the hero format is the graft itself: the RCCA centerline is
drawn in two colours, BLUE for the host portion every anatomy shares (0-130 mm)
and ORANGE for the real TopBrain siphon that replaces it, with a dot at the
junction. Any kink, doubling back or scale mismatch shows up there.

One cube is shared across ALL anatomies so the series is directly comparable
rather than each panel being re-framed to its own contents.

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
COL = {
    "onpath": ("#1f77b4", "host RCCA, 0-130 mm (shared)"),
    "rva":    ("#2ca02c", "RVA"),
    "other":  ("#d62728", "other branches"),
}
GRAFT_COL = "#ff7f0e"
MESH_ALPHA = {"onpath": 0.45, "rva": 0.45, "other": 0.14}


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

# The RVA is repaired per anatomy in the few cases where a siphon would
# otherwise run through it, so it cannot be meshed once and shared; the other
# branches are identical everywhere and are meshed once.
REF_RVA = np.asarray(cases[0]["g"]["rva"][0].coordinates, float)


def rva_repaired(case):
    p = np.asarray(case["g"]["rva"][0].coordinates, float)
    return not (len(p) == len(REF_RVA) and np.allclose(p, REF_RVA, atol=1e-6))


print("meshing the shared branches once", flush=True)
SHARED_OTHER = mesh_of(cases[0]["g"]["other"], "tb_other")

# Cube sized from the navigated route and the RVA ONLY, as in the hero figure.
# The arch and the other branches deliberately extend past it.
P = np.concatenate([c["rcca"] for c in cases]
                   + [np.asarray(b.coordinates, float)
                      for c in cases for b in c["g"]["rva"]], axis=0)
mins, maxs = P.min(axis=0), P.max(axis=0)
CTR = (mins + maxs) / 2.0
RAD = float(np.max(maxs - mins)) / 2.0 * 1.05
LIMS = [(CTR[i] - RAD, CTR[i] + RAD) for i in range(3)]
print("shared cube half-size %.1f mm" % RAD, flush=True)

# A Line3D with no data draws nothing in a legend, so build it from proxies.
HANDLES = [Line2D([], [], color=COL[k][0], lw=6, alpha=0.75, label=COL[k][1])
           for k in ("onpath", "other", "rva")]
HANDLES += [Line2D([], [], color=GRAFT_COL, lw=6, label="real TopBrain siphon"),
            Line2D([], [], color="none", marker="o", markersize=8,
                   markerfacecolor="white", markeredgecolor="k",
                   label="graft junction (130 mm)")]


def draw(ax, case, legend):
    meshes = {
        "other": SHARED_OTHER,
        "rva": mesh_of(case["g"]["rva"], "tb_rva_%s" % case["stem"]),
        "onpath": mesh_of(case["g"]["onpath"], "tb_on_%s" % case["stem"]),
    }
    for k in ("other", "rva", "onpath"):
        v, f = meshes[k]
        if f.size:
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=(*mcolors.to_rgb(COL[k][0]), MESH_ALPHA[k]),
                edgecolor=(0.25, 0.25, 0.3, 0.06), linewidth=0.15))

    for k in ("other", "rva"):
        for b in case["g"][k]:
            c = np.asarray(b.coordinates)
            if c.size == 0:
                continue
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=COL[k][0],
                    alpha=0.5 if k == "other" else 0.95,
                    linewidth=0.9 if k == "other" else 1.7, zorder=5)

    # the graft: host portion, real siphon, and the junction between them
    c, k = case["rcca"], case["cut"]
    ax.plot(c[:k + 1, 0], c[:k + 1, 1], c[:k + 1, 2], color=COL["onpath"][0],
            linewidth=1.7, zorder=6)
    ax.plot(c[k:, 0], c[k:, 1], c[k:, 2], color=GRAFT_COL, linewidth=2.2,
            zorder=7)
    ax.scatter(*c[k], s=42, facecolor="white", edgecolor="k", linewidth=1.2,
               zorder=8)

    ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1]); ax.set_zlim(*LIMS[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    ax.set_title("%s   route %.0f mm, siphon %.0f mm%s"
                 % (case["stem"], case["len"], case["len"] - case["s"][k],
                    "   (RVA repaired)" if rva_repaired(case) else ""),
                 fontsize=9)
    if legend:
        ax.legend(handles=HANDLES, loc="upper left", fontsize=7)


n = 0
for i in range(0, len(cases), 2):
    pair = cases[i:i + 2]
    fig = plt.figure(figsize=(8.5 * len(pair), 9.5))
    for j, case in enumerate(pair):
        ax = fig.add_subplot(1, len(pair), j + 1, projection="3d")
        draw(ax, case, legend=(j == 0))
    ids = "_".join(c["stem"].replace("topcow_", "") for c in pair)
    fig.tight_layout()
    path = os.path.join(OUT, "topbrain_pair_%02d_%s.png" % (i // 2 + 1, ids))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    n += 1
    print("wrote %s" % os.path.basename(path), flush=True)

print("done: %d images" % n, flush=True)
