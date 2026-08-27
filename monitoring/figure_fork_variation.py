#!/usr/bin/env python3
"""Fork-zoom variation figure — RCCA takeoffs vs RVA entries, rotated for 3-D.

Shows the first ~60 mm of arclength from the shared ostium for N worker
anatomies:

    RCCA takeoff      : one line per seed, different SHADES OF RED
                        (measured spread: 0 mm at the pinned ostium ->
                         ~6.7 mm by 30 mm arclength, still ~5.4 mm at 60 mm)
    RVA proximal      : one DOTTED BLUE line per seed (varied window;
                        spread peaks ~4.4 mm near 20 mm arclength)
    RVA distal        : a SINGLE GREEN line — beyond ~38 mm every variant is
                        numerically identical (max deviation 0.00 mm), i.e.
                        the dotted-blue takeoffs all merge into one RVA.

Two viewing angles in one image: the azimuth that maximizes the projected
RVA fan-out, and an orthogonal one, so the variation reads as 3-D rather
than as a flat spread.

Writes fork_variation_rotated.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_fork_variation.py /results/figs
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve_bench import DualDeviceNavRCCAVaried

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 8
BASE_SEED = 12345
ZOOM_MM = 60.0          # arclength window from the shared ostium
RVA_MERGE_MM = 38.0     # measured: variants coincide beyond this
ELEV = 20


def vt(seed):
    return DualDeviceNavRCCAVaried(seed=seed,
                                   episodes_between_change=10).vessel_tree


def branch(tree, key):
    for b in tree.branches:
        if key in str(b.name).upper():
            return np.asarray(b.coordinates, dtype=float)
    return None


def arclen(a):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a, axis=0),
                                                           axis=1))])


trees = [vt(BASE_SEED + i) for i in range(N_SEEDS)]
RCCA = [branch(t, "RCCA") for t in trees]
RVA = [branch(t, "RVA") for t in trees]

# ---- pick the azimuth that spreads the RVA variants most on screen --------
# Variation peaks around 20 mm; take the fan of points there and maximize the
# in-plane spread seen by the camera.
d0 = arclen(RVA[0])
i20 = int(np.argmin(np.abs(d0 - 20.0)))
fan = np.array([r[i20] for r in RVA if r is not None and len(r) > i20])
fan = fan - fan.mean(axis=0)


def screen_spread(elev_deg, azim_deg, pts):
    """Spread of pts projected onto the camera plane (matplotlib convention)."""
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    # camera forward vector
    f = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, f)
    u = pts @ right
    v = pts @ true_up
    return float(np.hypot(u.std(), v.std()))


cands = np.arange(-180, 180, 5)
scores = [screen_spread(ELEV, a, fan) for a in cands]
best_azim = float(cands[int(np.argmax(scores))])
alt_azim = best_azim + 90.0
print(f"RVA fan spread: best azim {best_azim:.0f} "
      f"(score {max(scores):.2f}) vs worst {cands[int(np.argmin(scores))]:.0f} "
      f"({min(scores):.2f})", flush=True)

RED = plt.cm.Reds(np.linspace(0.45, 1.0, N_SEEDS))     # RCCA takeoff shades
BLUE = "#1f6fd0"

# common (merged) distal RVA — identical across seeds, drawn once
d_rva = arclen(RVA[0])
merge_mask = (d_rva >= RVA_MERGE_MM) & (d_rva <= ZOOM_MM)
merged = RVA[0][merge_mask]

fig = plt.figure(figsize=(15, 7.5))
for panel, (azim, tag) in enumerate(
        [(best_azim, "view A — maximal RVA fan-out"),
         (alt_azim, "view B — rotated 90 deg")]):
    ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
    pts_all = []

    for k in range(N_SEEDS):
        # --- RCCA takeoff: shades of red
        rc = RCCA[k]
        if rc is not None:
            d = arclen(rc)
            m = d <= ZOOM_MM
            ax.plot(rc[m, 0], rc[m, 1], rc[m, 2], color=RED[k], lw=2.2,
                    label=("RCCA takeoff (one shade per anatomy)"
                           if (k == 0 and panel == 0) else None))
            pts_all.append(rc[m])
        # --- RVA proximal: dotted blue, one per seed
        rv = RVA[k]
        if rv is not None:
            d = arclen(rv)
            m = d <= RVA_MERGE_MM
            ax.plot(rv[m, 0], rv[m, 1], rv[m, 2], color=BLUE, lw=1.9,
                    ls=":", alpha=0.95,
                    label=("RVA entry — varied (dotted, one per anatomy)"
                           if (k == 0 and panel == 0) else None))
            pts_all.append(rv[m])

    # --- merged distal RVA: single green line
    if len(merged) > 1:
        ax.plot(merged[:, 0], merged[:, 1], merged[:, 2], color="#2ca02c",
                lw=3.4, solid_capstyle="round",
                label=("RVA distal — identical in every anatomy"
                       if panel == 0 else None))
        pts_all.append(merged)

    ostium = RCCA[0][0]
    ax.scatter(*ostium, s=130, marker="o", facecolor="white",
               edgecolors="black", linewidths=1.4, zorder=9,
               label="shared ostium (pinned)" if panel == 0 else None)

    P = np.concatenate(pts_all, axis=0)
    mins, maxs = P.min(axis=0), P.max(axis=0)
    ctr = (mins + maxs) / 2.0
    r = float(np.max(maxs - mins)) / 2.0 * 1.05
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.view_init(elev=ELEV, azim=azim)
    ax.set_xlabel("x (vessel-CS, mm)", fontsize=8)
    ax.set_ylabel("y (vessel-CS, mm)", fontsize=8)
    ax.set_zlabel("z (vessel-CS, mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{tag}   (elev {ELEV}, azim {azim:.0f})", fontsize=10)
    if panel == 0:
        ax.legend(loc="upper left", fontsize=8)

fig.suptitle(
    f"Bifurcation variation across {N_SEEDS} anatomies — first {ZOOM_MM:.0f} mm "
    "from the shared ostium\n"
    "RCCA takeoff diverges (up to ~6.7 mm); RVA entries vary (up to ~4.4 mm) "
    f"then merge into one identical distal RVA beyond ~{RVA_MERGE_MM:.0f} mm",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(OUT, "fork_variation_rotated.png"), dpi=160)
plt.close(fig)
print("wrote fork_variation_rotated.png", flush=True)
