#!/usr/bin/env python3
"""Visual QC of the v4 anatomies (the whole neck grafted), two per image.

The house style of figure_carotid_anatomies.py (Set B) and
figure_topbrain_pairs_v3.py (baked mesh): same view, same colours, one cube
shared by the whole set so every image is at the same scale, and segment
boundaries taken from provenance.json, so the seams shown are the ones the
graft used. v4 grafts six vessels, so:

    RCCA / LCCA  host grey | CCA blue | ICA cyan | siphon orange, a dot per seam
    RECA / LECA  fork ECA magenta, TopBrain ECA tip purple (dot at the join)
    RVA / LVA    host VA green, TopBrain VA olive (dot at the cut)
    other        host branches, thin red

The baked collision mesh (vessel_architecture_collision.obj) is drawn under the
centerlines, shaded by the group of its nearest centerline: right carotid, left
carotid, vertebrals, host. The cube is sized from all six grafted vessels, so
the left side is in frame; the arch runs out past it as in the hero figures.

    docker run --rm <mounts> eve-training-fixed
      python3 monitoring/figure_carotid_anatomies_v4.py <anatomies_v4> <out> [--only a,b,...] [--shard i/n]
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
SEG = [("host", "#7f7f7f"), ("CCA", "#1f77b4"), ("ICA", "#17becf"), ("siphon", "#ff7f0e")]
ECA_C, ECAEXT_C = "#e377c2", "#9467bd"
VA_C, VAEXT_C = "#2ca02c", "#8c564b"
OTHER_C = "#d62728"
MESH = {"right": ("#1f77b4", 0.13), "left": ("#17becf", 0.13), "va": ("#2ca02c", 0.12), "other": ("#d62728", 0.10)}


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def load_case(root):
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    g = {"other": []}
    for b in br:
        n = str(getattr(b, "name", "")).upper()
        c = np.asarray(b.coordinates, float)
        for key in ("RCCA", "LCCA", "RECA", "LECA", "RVA", "LVA"):
            if key in n:
                g[key] = c
                break
        else:
            g["other"].append(c)
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    return g, prov


def mesh_groups(root, g, target=12000):
    """The baked mesh, decimated for drawing, split by nearest centerline group."""
    import pyvista as pv
    from scipy.spatial import cKDTree
    m = pv.read(os.path.join(root, "vessel_architecture_collision.obj")).extract_surface().triangulate()
    if m.n_cells > target:
        m = m.decimate(1.0 - target / float(m.n_cells), volume_preservation=True)
    v = np.asarray(m.points, np.float32)
    f = np.asarray(m.faces, np.int64).reshape(-1, 4)[:, 1:]
    keys, pts = [], []
    for grp, names in (("right", ("RCCA", "RECA")), ("left", ("LCCA", "LECA")), ("va", ("RVA", "LVA"))):
        for n in names:
            keys += [grp] * len(g[n]); pts.append(g[n])
    for c in g["other"]:
        keys += ["other"] * len(c); pts.append(c)
    _, j = cKDTree(np.concatenate(pts)).query(v[f].mean(axis=1))
    keys = np.asarray(keys)
    return {k: (v, f[keys[j] == k]) for k in MESH}


def dot(ax, p):
    ax.scatter(*p, s=26, facecolor="white", edgecolor="k", lw=1.0, zorder=9)


def route(ax, c, cut, cca, ica):
    """A carotid route in its four segments, with the seams the graft used."""
    s = arclen(c)
    edges = [0.0, cut, cut + cca, cut + cca + ica, s[-1] + 1]
    for (lab, col), lo, hi in zip(SEG, edges[:-1], edges[1:]):
        m = (s >= lo - 1e-6) & (s <= hi)
        if m.sum() > 1:
            ax.plot(c[m, 0], c[m, 1], c[m, 2], color=col, lw=2.3, zorder=8)
    for e in edges[1:4]:
        dot(ax, c[int(np.argmin(np.abs(s - e)))])
    return s[-1] - edges[3]


def split_at(ax, c, at_mm, col_a, col_b, lw):
    """A branch drawn in two colours, split at an arclength, with a dot there."""
    s = arclen(c)
    k = int(np.argmin(np.abs(s - at_mm)))
    ax.plot(c[:k + 1, 0], c[:k + 1, 1], c[:k + 1, 2], color=col_a, lw=lw, zorder=7)
    if k < len(c) - 1:
        ax.plot(c[k:, 0], c[k:, 1], c[k:, 2], color=col_b, lw=lw, zorder=7)
        dot(ax, c[k])


def draw(ax, root, case, lims, legend):
    g, prov = case
    # draw in zorder, not mplot3d's per-artist depth estimate: the translucent
    # mesh would otherwise land on top of the centerlines and turn the orange
    # siphon brown under a blue shell
    ax.computed_zorder = False
    for k, (v, f) in mesh_groups(root, g).items():
        if f.size:
            col, a = MESH[k]
            ax.add_collection3d(Poly3DCollection(v[f], facecolor=(*mcolors.to_rgb(col), a),
                                                 edgecolor=(0.25, 0.25, 0.3, 0.04), linewidth=0.1, zorder=1))
    for c in g["other"]:
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=OTHER_C, lw=0.8, alpha=0.45, zorder=4)

    L, X, V = prov["left"], prov["xform"], prov["vertebrals"]
    sip_r = route(ax, g["RCCA"], prov["host_cut_mm"], prov["cca_mm"], prov["ica_mm"])
    sip_l = route(ax, g["LCCA"], L["host_cut_mm"], L["cca_mm"], L["ica_mm"])
    for side, key, ext in (("right", "RECA", "ecaext_R"), ("left", "LECA", "ecaext_L")):
        if ext in X:
            split_at(ax, g[key], X[ext]["join_at_mm"], ECA_C, ECAEXT_C, 2.2)
        else:
            c = g[key]
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=ECA_C, lw=2.2, zorder=7)
    for key, xk in (("RVA", "va_R"), ("LVA", "va_L")):
        split_at(ax, g[key], X[xk]["cut_mm"], VA_C, VAEXT_C, 1.8)

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    for i, nm in enumerate("xyz"):
        getattr(ax, "set_%slabel" % nm)("%s (vessel-CS, mm)" % nm, fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    tips = [s[0].upper() for s in ("right", "left") if prov["eca_extension"][s].get("extended")]
    ax.set_title("%s  (%s)\nleft fork %s%s   siphons R %.0f / L %.0f mm   VA donors %.0f / %.0f mm   ECA tips %s"
                 % (os.path.basename(root), prov["handedness"], L["lower"],
                    " (substitute)" if L.get("lower_fallback") else "", sip_r, sip_l,
                    V["right"]["donor_len_mm"], V["left"]["donor_len_mm"],
                    " + ".join(tips) or "none"),
                 fontsize=8)
    if legend:
        h = [Line2D([], [], color=col, lw=4, label=lab) for lab, col in SEG]
        h += [Line2D([], [], color=ECA_C, lw=4, label="fork ECA"),
              Line2D([], [], color=ECAEXT_C, lw=4, label="TopBrain ECA tip"),
              Line2D([], [], color=VA_C, lw=3, label="host VA"),
              Line2D([], [], color=VAEXT_C, lw=3, label="TopBrain VA"),
              Line2D([], [], color=OTHER_C, lw=3, alpha=0.6, label="other host branches"),
              Line2D([], [], color="none", marker="o", markersize=7,
                     markerfacecolor="white", markeredgecolor="k", label="seams / joins / cuts")]
        ax.legend(handles=h, loc="upper left", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomies")
    ap.add_argument("out")
    ap.add_argument("--only", default=None, help="comma list, paired in the order given")
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    roots = [os.path.dirname(r) for r in sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))]
    # One shared cube across the WHOLE set, computed before selecting, so every
    # image is at identical scale no matter which subset or worker made it.
    P = []
    for r in roots:
        g, _ = load_case(r)
        P += [g[k] for k in ("RCCA", "LCCA", "RECA", "LECA", "RVA", "LVA")]
    P = np.concatenate(P, axis=0)
    ctr = (P.min(0) + P.max(0)) / 2.0
    rad = float(np.max(P.max(0) - P.min(0))) / 2.0 * 1.05
    lims = [(ctr[i] - rad, ctr[i] + rad) for i in range(3)]
    print("%d anatomies, shared cube half-size %.1f mm" % (len(roots), rad), flush=True)

    if a.only:
        by = {os.path.basename(r): r for r in roots}
        roots = [by[n] for n in a.only.split(",")]
    pairs = [roots[i:i + 2] for i in range(0, len(roots), 2)]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        pairs = pairs[i::n]

    for grp in pairs:
        fig = plt.figure(figsize=(8.5 * len(grp), 9.5))
        for j, root in enumerate(grp):
            ax = fig.add_subplot(1, len(grp), j + 1, projection="3d")
            draw(ax, root, load_case(root), lims, legend=(j == 0))
        fig.tight_layout()
        nm = "__".join(os.path.basename(r) for r in grp)
        path = os.path.join(a.out, "carotid_v4_%s.png" % nm[:110])
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("wrote %s" % os.path.basename(path), flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
