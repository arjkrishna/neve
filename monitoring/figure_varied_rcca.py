#!/usr/bin/env python3
"""Presentation figures: how the varied RCCA / RVA anatomy differs per worker
and per generation.

Uses the REAL training code path (eve.intervention.vesseltree.
RCCAVariedFromMesh via eve_bench.DualDeviceNavRCCAVaried's own construction
args), so what is drawn is exactly the geometry the workers navigate. Only
CENTERLINES are drawn — no SOFA, no meshing (mesh_path is lazy and is never
touched), so this is cheap and safe to run alongside a live training run.

Run inside the training image:
    docker run --rm <mounts> eve-training-fixed \
        python3 /opt/eve_training/monitoring/figure_varied_rcca.py /results/figs

Outputs (into the given dir):
    varied_rcca_workers.png   — 6 worker anatomies (seeds 12345..), overlaid + panels
    varied_rcca_generations.png — one worker, 6 successive regenerations
    varied_rcca_overlay_zoom.png — RCCA+RVA only, all draws overlaid, fork zoom
    varied_rcca_eval_vs_train.png — held-out eval mesh (12344) vs train workers
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve.intervention.vesseltree import RCCAVariedFromMesh
from eve_bench.dualdevicenav import load_branches, DATA_DIR

RCCA = "Centerline curve - RCCA.mrk"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

# Same construction args DualDeviceNavRCCAVaried uses (defaults).
BRANCHES = load_branches(os.path.join(DATA_DIR, "Centrelines_comb"))


def make_tree(seed, gens=0):
    """A tree at generation `gens` for `seed` (gens=0 = as constructed)."""
    t = RCCAVariedFromMesh(
        branch_list=BRANCHES, rcca_name=RCCA, episodes_between_change=10,
        seed=seed,
    )
    for _ in range(gens):
        t._generate()
    return t


def get(tree, key):
    for b in tree.branches:
        if key in str(b.name).upper():
            return np.asarray(b.coordinates, dtype=float)
    return None


def others(tree):
    out = []
    for b in tree.branches:
        n = str(b.name).upper()
        if "RCCA" in n or "RVA" in n:
            continue
        out.append(np.asarray(b.coordinates, dtype=float))
    return out


def style_ax(ax, title, elev=18, azim=-72):
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=7, labelpad=-4)
    ax.set_ylabel("y (mm)", fontsize=7, labelpad=-4)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)


def draw_context(ax, tree, alpha=0.25):
    for c in others(tree):
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="#bdbdbd", lw=0.7, alpha=alpha)


def focus_limits(ax, arrs, pad=8):
    P = np.vstack(arrs)
    for setter, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(P[:, i].min() - pad, P[:, i].max() + pad)


COLORS = plt.cm.viridis(np.linspace(0.05, 0.9, 6))

# ---------------------------------------------------------------- FIG 1
# Six worker anatomies (the seeds the 16 training workers actually use).
seeds = [12345, 12346, 12347, 12348, 12349, 12350]
trees = [make_tree(s) for s in seeds]

fig = plt.figure(figsize=(15, 9))
for i, (s, t) in enumerate(zip(seeds, trees)):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    draw_context(ax, t)
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    ax.plot(rc[:, 0], rc[:, 1], rc[:, 2], color="#d62728", lw=2.0, label="RCCA (varied)")
    if rv is not None:
        ax.plot(rv[:, 0], rv[:, 1], rv[:, 2], color="#2ca02c", lw=2.0, label="RVA (varied takeoff)")
    ins = np.asarray(t.insertion.position, dtype=float)
    ax.scatter(*ins, s=45, marker="*", color="k", zorder=6, label="insertion (fixed)")
    focus_limits(ax, [rc] + ([rv] if rv is not None else []))
    style_ax(ax, f"worker seed {s}  (fingerprint {t.mesh_fingerprint})")
    if i == 0:
        ax.legend(loc="upper left", fontsize=6)

# overlay panel spanning the right column
axo = fig.add_subplot(1, 4, 4, projection="3d")
draw_context(axo, trees[0], alpha=0.18)
arrs = []
for c, (s, t) in zip(COLORS, zip(seeds, trees)):
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    axo.plot(rc[:, 0], rc[:, 1], rc[:, 2], color=c, lw=1.8, label=f"seed {s}")
    if rv is not None:
        axo.plot(rv[:, 0], rv[:, 1], rv[:, 2], color=c, lw=1.2, ls="--")
    arrs += [rc] + ([rv] if rv is not None else [])
ins = np.asarray(trees[0].insertion.position, dtype=float)
axo.scatter(*ins, s=70, marker="*", color="k", zorder=6)
focus_limits(axo, arrs)
style_ax(axo, "all 6 workers overlaid\n(solid = RCCA, dashed = RVA)")
axo.legend(loc="upper left", fontsize=6)
fig.suptitle(
    "Per-worker RCCA/RVA anatomy variation — loaded arch fixed (grey), "
    "RCCA+proximal RVA perturbed per worker seed; insertion point identical",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "varied_rcca_workers.png"), dpi=150)
plt.close(fig)
print("wrote varied_rcca_workers.png")

# ---------------------------------------------------------------- FIG 2
# One worker, six successive regenerations (what it sees every 10 episodes).
gens = [make_tree(12345, gens=g) for g in range(6)]
fig = plt.figure(figsize=(15, 9))
for i, t in enumerate(gens):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    draw_context(ax, t)
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    ax.plot(rc[:, 0], rc[:, 1], rc[:, 2], color="#d62728", lw=2.0)
    if rv is not None:
        ax.plot(rv[:, 0], rv[:, 1], rv[:, 2], color="#2ca02c", lw=2.0)
    ins = np.asarray(t.insertion.position, dtype=float)
    ax.scatter(*ins, s=45, marker="*", color="k", zorder=6)
    focus_limits(ax, [rc] + ([rv] if rv is not None else []))
    style_ax(ax, f"episode {i*10}-{i*10+9}   ({t.mesh_fingerprint})")

axo = fig.add_subplot(1, 4, 4, projection="3d")
draw_context(axo, gens[0], alpha=0.18)
arrs = []
for c, (g, t) in zip(COLORS, enumerate(gens)):
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    axo.plot(rc[:, 0], rc[:, 1], rc[:, 2], color=c, lw=1.8, label=f"gen {g}")
    if rv is not None:
        axo.plot(rv[:, 0], rv[:, 1], rv[:, 2], color=c, lw=1.2, ls="--")
    arrs += [rc] + ([rv] if rv is not None else [])
focus_limits(axo, arrs)
style_ax(axo, "6 generations overlaid")
axo.legend(loc="upper left", fontsize=6)
fig.suptitle(
    "Within-worker regeneration (seed 12345) — a new RCCA every 10 episodes",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "varied_rcca_generations.png"), dpi=150)
plt.close(fig)
print("wrote varied_rcca_generations.png")

# ---------------------------------------------------------------- FIG 3
# Fork zoom: how much the RVA entry / RCCA takeoff actually move.
pool = [make_tree(12345 + i) for i in range(8)]
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
arrs = []
for c, t in zip(plt.cm.plasma(np.linspace(0.05, 0.9, len(pool))), pool):
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    ax1.plot(rc[:, 0], rc[:, 1], rc[:, 2], color=c, lw=1.6)
    if rv is not None:
        ax1.plot(rv[:, 0], rv[:, 1], rv[:, 2], color=c, lw=1.6, ls="--")
    arrs += [rc] + ([rv] if rv is not None else [])
ins = np.asarray(pool[0].insertion.position, dtype=float)
ax1.scatter(*ins, s=80, marker="*", color="k", zorder=6, label="insertion (identical)")
focus_limits(ax1, arrs)
style_ax(ax1, "8 worker anatomies — full RCCA (solid) + RVA (dashed)")
ax1.legend(loc="upper left", fontsize=7)

# zoom on the fork region (first 60mm of arclength from the shared ostium)
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
zarrs = []
for c, t in zip(plt.cm.plasma(np.linspace(0.05, 0.9, len(pool))), pool):
    for key, ls in (("RCCA", "-"), ("RVA", "--")):
        a = get(t, key)
        if a is None:
            continue
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
        m = d <= 60.0
        ax2.plot(a[m, 0], a[m, 1], a[m, 2], color=c, lw=2.0, ls=ls)
        zarrs.append(a[m])
ax2.scatter(*ins, s=110, marker="*", color="k", zorder=6)
focus_limits(ax2, zarrs, pad=3)
style_ax(ax2, "FORK ZOOM — first 60 mm: RCCA takeoff and RVA entry vary\n"
              "(shared ostium point pinned; both daughters move)")
fig.suptitle("Bifurcation variation — the RCCA-vs-RVA deflection the policy cannot memorize",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "varied_rcca_overlay_zoom.png"), dpi=150)
plt.close(fig)
print("wrote varied_rcca_overlay_zoom.png")

# ---------------------------------------------------------------- FIG 4
# Held-out eval anatomy vs the training workers.
ev = make_tree(12344)
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
draw_context(ax, ev, alpha=0.2)
arrs = []
for t in pool[:6]:
    rc, rv = get(t, "RCCA"), get(t, "RVA")
    ax.plot(rc[:, 0], rc[:, 1], rc[:, 2], color="#9e9e9e", lw=1.2, alpha=0.85)
    if rv is not None:
        ax.plot(rv[:, 0], rv[:, 1], rv[:, 2], color="#9e9e9e", lw=1.0, ls="--", alpha=0.7)
    arrs += [rc]
rc, rv = get(ev, "RCCA"), get(ev, "RVA")
ax.plot(rc[:, 0], rc[:, 1], rc[:, 2], color="#d62728", lw=2.6, label="HELD-OUT eval (seed 12344)")
if rv is not None:
    ax.plot(rv[:, 0], rv[:, 1], rv[:, 2], color="#2ca02c", lw=2.2, ls="--", label="held-out RVA")
ax.plot([], [], color="#9e9e9e", lw=1.2, label="training workers (12345+)")
focus_limits(ax, arrs + [rc])
style_ax(ax, "Held-out evaluation anatomy vs training anatomies")
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "varied_rcca_eval_vs_train.png"), dpi=150)
plt.close(fig)
print("wrote varied_rcca_eval_vs_train.png")

# ---------------------------------------------------------------- stats
print("\n=== quantitative spread (for the caption) ===")
rows = []
for t in pool:
    rc = get(t, "RCCA")
    d = float(np.sum(np.linalg.norm(np.diff(rc, axis=0), axis=1)))
    rows.append((t.mesh_fingerprint, d))
    print(f"  {t.mesh_fingerprint}: RCCA centerline length {d:.1f} mm")
L = np.array([r[1] for r in rows])
print(f"  across {len(rows)} workers: mean {L.mean():.1f} mm, sd {L.std():.1f} mm, "
      f"range {L.min():.1f}-{L.max():.1f} mm")
base = get(make_tree(12345), "RCCA")
for s in (12346, 12350, 12344):
    o = get(make_tree(s), "RCCA")
    n = min(len(base), len(o))
    dev = np.linalg.norm(base[:n] - o[:n], axis=1)
    print(f"  seed 12345 vs {s}: mean point deviation {dev.mean():.2f} mm, max {dev.max():.2f} mm")
