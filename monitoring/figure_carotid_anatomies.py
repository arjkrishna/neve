#!/usr/bin/env python3
"""Visual QC of the three-source anatomies, two per image.

Same hero framing as the TopBrain figures, but these have more to show. Each
route is composed of four segments from three donors, so it is drawn in four
colours with a dot at each seam:

    host    grey      the shipped arch, ostium to the first seam
    CCA     blue      real common carotid, from the bifurcation model
    ICA     cyan      real cervical ICA, extended by 10 mm where marked
    siphon  orange    real TopBrain ICA to the terminus

and the ECA is drawn in magenta, because it is the whole point of this set:
a fork the previous 49 anatomies could not represent. Its presence beside the
ICA is what a policy would have to learn to discriminate.

Segment boundaries come from each anatomy's provenance.json rather than being
guessed from calibre, so the seams shown are the ones the graft actually used.

    docker run --rm <mounts> eve-training-fixed
      python3 monitoring/figure_carotid_anatomies.py <anatomies> <out> [--shard i/n]
"""
import argparse
import glob
import json
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

ELEV, AZIM = 16, -68
SEG = [("host", "#7f7f7f"), ("CCA", "#1f77b4"),
       ("ICA", "#17becf"), ("siphon", "#ff7f0e")]
ECA_C, RVA_C, OTHER_C = "#e377c2", "#2ca02c", "#d62728"
MESH_A = {"route": 0.40, "eca": 0.40, "rva": 0.35, "other": 0.12}


def load_mesh(path, target=9000):
    import pyvista as pv
    m = pv.read(path)
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


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def load_case(root):
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    g = {"route": None, "eca": None, "rva": None, "other": []}
    for b in br:
        n = str(getattr(b, "name", "")).upper()
        if "RECA" in n:
            g["eca"] = np.asarray(b.coordinates, float)
        elif "RCCA" in n:
            g["route"] = np.asarray(b.coordinates, float)
        elif "RVA" in n:
            g["rva"] = np.asarray(b.coordinates, float)
        else:
            g["other"].append(np.asarray(b.coordinates, float))
    prov = {}
    p = os.path.join(root, "provenance.json")
    if os.path.exists(p):
        prov = json.load(open(p, encoding="utf-8"))
    return g, prov


def draw(ax, root, case, lims, legend):
    g, prov = case
    v, f = load_mesh(os.path.join(root, "vessel_architecture_collision.obj"))
    if f.size:
        ax.add_collection3d(Poly3DCollection(
            v[f], facecolor=(*mcolors.to_rgb(OTHER_C), MESH_A["other"]),
            edgecolor=(0.25, 0.25, 0.3, 0.05), linewidth=0.1))
    for c in g["other"]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=OTHER_C, lw=0.8, alpha=0.45, zorder=4)
    if g["rva"] is not None:
        c = g["rva"]
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=RVA_C, lw=1.6, alpha=0.9, zorder=5)
    if g["eca"] is not None:
        c = g["eca"]
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=ECA_C, lw=2.4, alpha=0.95, zorder=7)

    # the route, split at the seams the graft actually used
    c = g["route"]
    s = arclen(c)
    hc = prov.get("host_cut_mm", 45.0)
    b1 = hc + prov.get("cca_mm", 28.0)
    b2 = b1 + prov.get("ica_mm", 58.0)
    edges = [0.0, hc, b1, b2, s[-1] + 1]
    for (lab, col), lo, hi in zip(SEG, edges[:-1], edges[1:]):
        m = (s >= lo - 1e-6) & (s <= hi)
        if m.sum() > 1:
            ax.plot(c[m, 0], c[m, 1], c[m, 2], color=col, lw=2.4, zorder=8)
    for cut in (hc, b1, b2):
        i = int(np.argmin(np.abs(s - cut)))
        ax.scatter(*c[i], s=30, facecolor="white", edgecolor="k", lw=1.0, zorder=9)

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    for i, nm in enumerate("xyz"):
        getattr(ax, "set_%slabel" % nm)("%s (vessel-CS, mm)" % nm, fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    low, sip = os.path.basename(root).split("__")
    ext = "  (ICA extended)" if prov.get("ica_mm", 0) and prov.get("trim_cca_mm", 0) == 0 else ""
    ax.set_title("%s  +  %s\nroute %.0f mm   host %.0f | CCA %.0f | ICA %.0f | siphon %.0f%s"
                 % (low, sip, s[-1], hc, prov.get("cca_mm", 0), prov.get("ica_mm", 0),
                    s[-1] - b2, ext), fontsize=8)
    if legend:
        h = [Line2D([], [], color=col, lw=4, label=lab) for lab, col in SEG]
        h += [Line2D([], [], color=ECA_C, lw=4, label="ECA (new fork)"),
              Line2D([], [], color=RVA_C, lw=3, label="RVA"),
              Line2D([], [], color=OTHER_C, lw=3, alpha=0.6, label="other branches"),
              Line2D([], [], color="none", marker="o", markersize=7,
                     markerfacecolor="white", markeredgecolor="k", label="seams")]
        ax.legend(handles=h, loc="upper left", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomies")
    ap.add_argument("out")
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    roots = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    roots = [os.path.dirname(r) for r in roots]
    # One shared cube across the WHOLE set, computed before sharding, so every
    # image in the series is at identical scale no matter which worker made it.
    P = []
    for r in roots:
        g, _ = load_case(r)
        P.append(g["route"])
        if g["rva"] is not None:
            P.append(g["rva"])
    P = np.concatenate(P, axis=0)
    ctr = (P.min(0) + P.max(0)) / 2.0
    rad = float(np.max(P.max(0) - P.min(0))) / 2.0 * 1.05
    LIMS = [(ctr[i] - rad, ctr[i] + rad) for i in range(3)]
    print("%d anatomies, shared cube half-size %.1f mm" % (len(roots), rad), flush=True)

    pairs = [roots[i:i + 2] for i in range(0, len(roots), 2)]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        pairs = pairs[i::n]
        print("shard %d/%d -> %d images" % (i, n, len(pairs)), flush=True)

    for grp in pairs:
        fig = plt.figure(figsize=(8.5 * len(grp), 9.5))
        for j, root in enumerate(grp):
            ax = fig.add_subplot(1, len(grp), j + 1, projection="3d")
            draw(ax, root, load_case(root), LIMS, legend=(j == 0))
        fig.tight_layout()
        nm = "__".join(os.path.basename(r) for r in grp)
        path = os.path.join(a.out, "carotid_%s.png" % nm[:110])
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("wrote %s" % os.path.basename(path), flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
