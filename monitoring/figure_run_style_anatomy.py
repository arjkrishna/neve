#!/usr/bin/env python3
"""Run-snapshot-style anatomy figures — SAME rendering as the training/eval
snapshots (grey centerlines, on-path branches highlighted, orange planned
path, yellow target X, same axes/labels/orientation) but WITHOUT the
guidewire/catheter — for showing how the varied RCCA + RVA entry differ
per worker and per regeneration.

Uses the real training objects (DualDeviceNavRCCAVaried + FixedPathfinder +
CenterlineRandom) so the planned path and target are computed exactly as in a
run. SOFA is never started (no intervention.reset()), so this is cheap and
safe alongside a live training run.

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_run_style_anatomy.py /results/figs
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
RCCA = "Centerline curve - RCCA.mrk"
BASE_SEED = 12345          # training workers: BASE_SEED + worker_id
EVAL_SEED = BASE_SEED - 1  # held-out eval anatomy


def build(seed, gens=0, target_seed=0):
    """Intervention + pathfinder + sampled target, no SOFA."""
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


def draw_centerlines(ax, tree, on_path, labels=False):
    """snapshot.py::_draw_centerlines, recolored for the anatomy figure:
    on-path branches BLUE (unchanged), RVA GREEN, every other branch RED."""
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
    pts = np.asarray(pts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ctr = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


def render(ax, itv, pf, title, legend=False, elev=16, azim=-68):
    tree = itv.vessel_tree
    on_path = path_branch_names(pf)
    draw_centerlines(ax, tree, on_path, labels=legend)

    # Frame on the NAVIGATED region (RCCA + RVA + planned path), not the whole
    # arch — otherwise the varied siphon is a few percent of a 600 mm box and
    # the variation is invisible. The rest of the tree still draws in grey and
    # simply clips at the axes.
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

    # BUGFIX — target.coordinates3d is in the TRACKING3D frame; the
    # centerlines/planned path are vessel-CS. snapshot.py::_target_vcs does
    # this conversion; without it the X floats off the vessel.
    tgt = tracking3d_to_vessel_cs(
        np.asarray(itv.target.coordinates3d, dtype=float),
        itv.fluoroscopy.image_rot_zx,
        itv.fluoroscopy.image_center,
    )
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=90, marker="X",
               edgecolors="black", linewidths=0.6, label="target", zorder=7)
    bounds.append(np.atleast_2d(tgt))
    ins = np.asarray(tree.insertion.position, dtype=float)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=55, marker="*",
               label="insertion (fixed)")

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
# Six training-worker anatomies, run-snapshot style.
fig = plt.figure(figsize=(16, 10))
for i in range(6):
    itv, pf = build(BASE_SEED + i, target_seed=7)
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    render(ax, itv, pf,
           f"worker {i}  seed {BASE_SEED + i}  ({itv.vessel_tree.mesh_fingerprint})",
           legend=(i == 0))
fig.suptitle("Per-worker varied RCCA anatomy — run-snapshot view "
             "(centerlines + planned path + target; devices omitted)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "runstyle_workers.png"), dpi=150)
plt.close(fig)
print("wrote runstyle_workers.png")

# ---------------------------------------------------------------- FIG B
# One worker, successive regenerations (every 10 episodes).
fig = plt.figure(figsize=(16, 10))
for g in range(6):
    itv, pf = build(BASE_SEED, gens=g, target_seed=7)
    ax = fig.add_subplot(2, 3, g + 1, projection="3d")
    render(ax, itv, pf,
           f"episodes {g*10}-{g*10+9}   ({itv.vessel_tree.mesh_fingerprint})",
           legend=(g == 0))
fig.suptitle(f"Within-worker regeneration (seed {BASE_SEED}) — new RCCA every 10 episodes",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "runstyle_generations.png"), dpi=150)
plt.close(fig)
print("wrote runstyle_generations.png")

# ---------------------------------------------------------------- FIG C
# Held-out eval anatomy, with several sampled targets (the 98-seed eval set).
fig = plt.figure(figsize=(16, 6))
for j, ts in enumerate((3, 17, 55)):
    itv, pf = build(EVAL_SEED, target_seed=ts)
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    d = np.linalg.norm(np.asarray(pf.path_points_vessel_cs)[-1]
                       - np.asarray(pf.path_points_vessel_cs)[0])
    try:
        plen = float(np.sum(np.linalg.norm(
            np.diff(np.asarray(pf.path_points_vessel_cs), axis=0), axis=1)))
    except Exception:
        plen = float("nan")
    render(ax, itv, pf,
           f"HELD-OUT eval anatomy (seed {EVAL_SEED})\n"
           f"target draw {ts} — planned path {plen:.0f} mm", legend=(j == 0))
fig.suptitle("Held-out evaluation anatomy — fixed vessel, 98 target draws "
             "(path length drives the reachability cliff)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "runstyle_eval_targets.png"), dpi=150)
plt.close(fig)
print("wrote runstyle_eval_targets.png")

# ---------------------------------------------------------------- FIG D
# Single large hero panel (one anatomy, publication size).
itv, pf = build(BASE_SEED, target_seed=7)
fig = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(111, projection="3d")
render(ax, itv, pf,
       f"RCCA-varied anatomy (seed {BASE_SEED}, {itv.vessel_tree.mesh_fingerprint})\n"
       "blue = on-path (RCCA route), green = RVA, red = other branches, "
       "orange = planned path", legend=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "runstyle_hero.png"), dpi=160)
plt.close(fig)
print("wrote runstyle_hero.png")
