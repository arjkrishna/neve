#!/usr/bin/env python3
"""Three varied anatomies (centerlines) at the EXACT scale/layout of
runstyle_eval_targets.png.

The framing logic is copied verbatim from figure_run_style_anatomy.py's
render(): bounds come from branches whose NAME contains RCCA or RVA, plus
the planned path, plus the target — the insertion point is deliberately NOT
in the bounds, and there is no padding factor. Equal-aspect cube per panel,
matplotlib's default 3d box aspect, 1x3 at figsize (16, 6). Reproducing
those details is what makes the vessels render at the same size as in
runstyle_eval_targets.png.

Anatomies: s12344g1, s12349g1, s12347g1.

Writes variation_three_matched_scale.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_variation_three.py /results/figs
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve_bench")

import eve
from eve.util.coordtransform import tracking3d_to_vessel_cs
from eve_bench import DualDeviceNavRCCAVaried

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
TARGET_SEED = 7
PICKS = [(12344, 1), (12349, 1), (12347, 1)]   # (seed, fingerprint generation)


def build(seed, gen):
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


def draw_centerlines(ax, tree, on_path, labels=False):
    """Palette: on-path BLUE, RVA GREEN, everything else RED."""
    seen = set()
    for br in tree.branches:
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        name = str(getattr(br, "name", "?"))
        if name in on_path:
            color, alpha, lw, key = "#1f77b4", 0.90, 1.8, "on-path branches"
        elif "RVA" in name.upper():
            color, alpha, lw, key = "#2ca02c", 0.90, 1.8, "RVA (varied takeoff)"
        else:
            color, alpha, lw, key = "#d62728", 0.55, 1.0, "other branches"
        lab = None
        if labels and key not in seen:
            seen.add(key)
            lab = key
        ax.plot(c[:, 0], c[:, 1], c[:, 2],
                color=color, alpha=alpha, linewidth=lw, label=lab)


def set_equal_aspect(ax, pts):
    """Verbatim from figure_run_style_anatomy.py — no padding factor."""
    pts = np.asarray(pts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ctr = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


def render(ax, itv, pf, title, legend=False, elev=16, azim=-68):
    """Verbatim structure of figure_run_style_anatomy.py::render()."""
    tree = itv.vessel_tree
    on_path = path_branch_names(pf)
    draw_centerlines(ax, tree, on_path, labels=legend)

    # Bounds: NAME-matched RCCA/RVA branches only (NOT the on-path group —
    # that pulls in the (11) bridge and enlarges the cube).
    bounds = []
    for b in tree.branches:
        n = str(getattr(b, "name", "")).upper()
        c = np.asarray(b.coordinates)
        if c.size and ("RCCA" in n or "RVA" in n):
            bounds.append(c)

    pp = None
    try:
        pp = np.asarray(pf.path_points_vessel_cs)
    except Exception:
        pass
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                color="#ffaa00", linewidth=2.4, label="planned path")
        bounds.append(pp)
    if not bounds:
        bounds = [np.asarray(b.coordinates) for b in tree.branches
                  if np.asarray(b.coordinates).size]

    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=90, marker="X",
               edgecolors="black", linewidths=0.6, label="target", zorder=7)
    bounds.append(np.atleast_2d(tgt))          # target IS in bounds
    ins = np.asarray(tree.insertion.position, dtype=float)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=55, marker="*",
               label="insertion (fixed)")      # insertion is NOT in bounds

    set_equal_aspect(ax, np.concatenate(bounds, axis=0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=-2)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=-2)
    ax.set_zlabel("z (mm)", fontsize=8, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    # Heading sits INSIDE the axes (lower than a normal title) and, on the
    # panel that carries the legend, beside it rather than above it.
    if legend:
        ax.legend(loc="upper left", fontsize=7,
                  bbox_to_anchor=(0.0, 0.88))
        ax.text2D(0.30, 0.86, title, transform=ax.transAxes, fontsize=9,
                  va="top", ha="left")
    else:
        ax.text2D(0.18, 0.86, title, transform=ax.transAxes, fontsize=9,
                  va="top", ha="left")


fig = plt.figure(figsize=(16, 6.4))
PANEL_W, PANEL_H, PANEL_Y = 0.325, 0.90, 0.01
for j, (seed, gen) in enumerate(PICKS):
    itv, pf = build(seed, gen)
    fp = itv.vessel_tree.mesh_fingerprint
    try:
        pp = np.asarray(pf.path_points_vessel_cs)
        plen = float(np.sum(np.linalg.norm(np.diff(pp, axis=0), axis=1)))
    except Exception:
        plen = float("nan")
    rlen = float("nan")
    for br in itv.vessel_tree.branches:
        if "RCCA" in str(getattr(br, "name", "")).upper():
            c = np.asarray(br.coordinates)
            rlen = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            break
    ax = fig.add_axes([0.005 + j * PANEL_W, PANEL_Y,
                       PANEL_W, PANEL_H], projection="3d")
    render(ax, itv, pf,
           f"RCCA centerline {rlen:.0f} mm — planned path {plen:.0f} mm",
           legend=(j == 0))
    print(f"  {fp}: xlim {np.round(ax.get_xlim(),1)} zlim {np.round(ax.get_zlim(),1)}",
          flush=True)

fig.suptitle("Procedurally varied RCCA vasculature — three generated anatomies "
             "(centerlines)", fontsize=12, y=0.995)
fig.savefig(os.path.join(OUT, "variation_three_matched_scale_clean.png"), dpi=150)
plt.close(fig)
print("wrote variation_three_matched_scale_clean.png", flush=True)
