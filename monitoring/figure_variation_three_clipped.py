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

Writes variation_three_matched_scale_clipped.png — as the original, but
centerlines are clipped to the axis box so the aorta does not run off the
figure below the axes.

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


def draw_centerlines(ax, tree, on_path, labels=False, lims=None):
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
        segs = clipped_segments(c, lims) if lims is not None else [c]
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color, alpha=alpha, linewidth=lw, label=lab)
            lab = None




def _inside_mask(pts, lims):
    """Boolean mask of points inside the axis box."""
    pts = np.asarray(pts)
    m = np.ones(len(pts), dtype=bool)
    for i, (lo, hi) in enumerate(lims):
        m &= (pts[:, i] >= lo) & (pts[:, i] <= hi)
    return m


def clipped_segments(coords, lims):
    """Split a polyline into runs that lie inside the axis box.

    matplotlib does not clip 3-D lines to the axes limits, so a long branch
    (the aorta) is drawn running off the figure. Masking to the frame and
    plotting each contiguous run keeps the geometry honest without any
    trailing lines outside the box.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return []
    m = _inside_mask(coords, lims)
    out, start = [], None
    for i, ok in enumerate(m):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                out.append(coords[start:i])
            start = None
    if start is not None and len(coords) - start >= 2:
        out.append(coords[start:])
    return out


def clipped_mesh(verts, faces, lims):
    """Keep triangles whose centroid lies inside the axis box."""
    if faces.size == 0:
        return faces
    cent = verts[faces].mean(axis=1)
    return faces[_inside_mask(cent, lims)]


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
    """Same as the unclipped version, with two differences: the axis limits
    are established BEFORE drawing so branches can be clipped to them, and
    the planned path / target are drawn AFTER the centerlines so they stay
    on top (the draw order of the original)."""
    tree = itv.vessel_tree
    on_path = path_branch_names(pf)

    # ---- bounds first (pure computation, nothing drawn yet) --------------
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
        bounds.append(pp)
    if not bounds:
        bounds = [np.asarray(b.coordinates) for b in tree.branches
                  if np.asarray(b.coordinates).size]
    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
    bounds.append(np.atleast_2d(tgt))
    set_equal_aspect(ax, np.concatenate(bounds, axis=0))
    lims = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())

    # ---- centerlines, clipped to the frame -------------------------------
    draw_centerlines(ax, tree, on_path, labels=legend, lims=lims)

    # ---- planned path / target / insertion on top ------------------------
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                color="#ffaa00", linewidth=2.4, label="planned path", zorder=6)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=90, marker="X",
               edgecolors="black", linewidths=0.6, label="target", zorder=7)
    ins = np.asarray(tree.insertion.position, dtype=float)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=55, marker="*",
               label="insertion (fixed)", zorder=7)

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="upper left", fontsize=7)


fig = plt.figure(figsize=(16, 6))
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
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    render(ax, itv, pf,
           f"{fp}\nRCCA centerline {rlen:.0f} mm — planned path {plen:.0f} mm",
           legend=(j == 0))
    print(f"  {fp}: xlim {np.round(ax.get_xlim(),1)} zlim {np.round(ax.get_zlim(),1)}",
          flush=True)

fig.suptitle("Procedurally varied RCCA vasculature — three generated anatomies "
             "(centerlines)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "variation_three_matched_scale_clipped.png"), dpi=150)
plt.close(fig)
print("wrote variation_three_matched_scale_clipped.png", flush=True)
