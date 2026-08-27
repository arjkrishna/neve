#!/usr/bin/env python3
"""Three panels in ONE image: centerlines | vessel mesh | devices + observation.

Panels 1-2 are exactly hero_centerline_vs_mesh_clean.png (same seed, target,
view, shared axis cube). Panel 3 is devices_obs_region.png (same anatomy,
its own tight narrow frame — the devices figure deliberately does not use the
122 mm cube, which would re-introduce the empty width).

Layout notes (why add_axes and not subplots):
  * Axes3D.apply_aspect shrinks every 3-D axes to the SQUARE inscribed in its
    rect, then centres the projected box inside that square. A 3-D panel is
    therefore mostly transparent margin, and side-by-side subplots stack those
    margins into wide white gutters.
  * So the rects are given in INCHES, sized to keep each panel's inscribed
    square at least as large as it is in the standalone figures (nothing gets
    smaller), and deliberately OVERLAPPED so the empty margins overlap instead
    of adding up.
  * Titles and legends are placed in FIGURE coordinates for the same reason —
    axes-fraction positions would be measured against those oversized rects.

Writes three_panel_overview.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_three_panel.py /results/figs
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

SEED, GEN, TARGET_SEED = 12345, 0, 7
ELEV, AZIM = 16, -68

# ---- devices panel (verbatim from figure_devices_obs_region.py) -----------
GW_FRACTION = 0.50
GW_LEAD_MM = 14.0
PREVIEWS_MM = [10.0, 20.0, 40.0, 80.0]
RING_HALF_W = 14.5
RING_TALL = 0.62
ARC_DEG = (18.0, 162.0)

# ---- layout, in INCHES ---------------------------------------------------
FIG_W, FIG_H = 16.2, 9.4
PANEL_H = 7.55                 # inscribed square of every panel
PANEL_Y = 0.70
SQ = [7.55, 7.55, 7.55]        # rect widths (>= PANEL_H so height binds)
# Content half-widths measured off the render: ~3.0 in for the two cube
# panels (box + tick labels), ~1.6 in for the narrow devices slab. Spaced
# so the boxes clear each other by ~0.35 in and nothing overlaps.
CENTER_X = [3.15, 9.50, 14.45]
TITLE_Y = 0.985
LEGEND_Y = 0.930
LEGEND_FS = 11        # legend font; the band under the titles was empty
AXIS_FS = 12          # x/y/z axis labels
TICK_FS = 9           # tick numbers, kept proportional to AXIS_FS
LABEL_FS = 14         # bold (a)/(b)/(c) panel letters
LABEL_X = [0.35, 6.75, 12.95]   # inches, left edge of each panel
# (c) sits further out than the box because panel 3's title is wider than
# its plot; 12.95 clears panel (b)'s box at 12.8 and the title at 13.3.

COL = {
    "onpath": ("#1f77b4", "on-path branches (RCCA route)"),
    "rva":    ("#2ca02c", "RVA (varied takeoff)"),
    "other":  ("#d62728", "other branches"),
}
MESH_ALPHA = {"onpath": 0.45, "rva": 0.45, "other": 0.14}

# ---------------------------------------------------------------- build once
itv = DualDeviceNavRCCAVaried(seed=SEED, episodes_between_change=10)
for _ in range(GEN):
    itv.vessel_tree._generate()
itv.vessel_tree.reset(episode_nr=0)
itv.target.reset(episode_nr=0, seed=TARGET_SEED)
pf = eve.pathfinder.FixedPathfinder(intervention=itv)
try:
    pf.reset(episode_nr=0)
except Exception:
    pass

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


meshes = {}
for g, brs in groups.items():
    if not brs:
        continue
    t0 = time.time()
    p = generate_temp_mesh(brs, f"tri_{g}", 0.99)
    meshes[g] = load_mesh(p)
    try:
        os.remove(p)
    except OSError:
        pass
    print(f"  mesh[{g}] {len(meshes[g][1])} tris {time.time()-t0:.1f}s", flush=True)

try:
    pp = np.asarray(pf.path_points_vessel_cs, dtype=float)
except Exception:
    pp = None
tgt = tracking3d_to_vessel_cs(
    np.asarray(itv.target.coordinates3d, dtype=float),
    itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)

# ------------------------------------------------- ONE shared cube (1 and 2)
frame_pts = []
for g in ("onpath", "rva"):
    if g in meshes:
        frame_pts.append(meshes[g][0])
    for br in groups[g]:
        c = np.asarray(br.coordinates)
        if c.size:
            frame_pts.append(c)
if pp is not None and len(pp) > 1:
    frame_pts.append(pp)
frame_pts.append(np.atleast_2d(tgt))
frame_pts.append(np.atleast_2d(ins))
P = np.concatenate(frame_pts, axis=0)
mins, maxs = P.min(axis=0), P.max(axis=0)
ctr = (mins + maxs) / 2.0
RAD = float(np.max(maxs - mins)) / 2.0 * 1.05
LIMS = [(ctr[i] - RAD, ctr[i] + RAD) for i in range(3)]
print(f"  shared cube: center {ctr.round(1)}, half-size {RAD:.1f} mm", flush=True)

# ------------------------------------------------- devices geometry (panel 3)
seg = np.linalg.norm(np.diff(pp, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
total = float(s[-1])


def point_at(arc):
    arc = float(np.clip(arc, 0.0, total))
    i = int(np.searchsorted(s, arc, side="right") - 1)
    i = min(max(i, 0), len(pp) - 2)
    d = s[i + 1] - s[i]
    t = 0.0 if d <= 1e-9 else (arc - s[i]) / d
    return pp[i] + t * (pp[i + 1] - pp[i])


def polyline_to(arc):
    m = s <= arc
    out = pp[m]
    tip = point_at(arc)
    if len(out) == 0 or np.linalg.norm(out[-1] - tip) > 1e-9:
        out = np.vstack([out, tip]) if len(out) else np.atleast_2d(tip)
    return out


gw_s = GW_FRACTION * total
cath_s = max(0.0, gw_s - GW_LEAD_MM)
gw_line, cath_line = polyline_to(gw_s), polyline_to(cath_s)
gw_tip, cath_tip = point_at(gw_s), point_at(cath_s)

_e, _a = np.radians(ELEV), np.radians(AZIM)
view_dir = np.array([np.cos(_e) * np.cos(_a), np.cos(_e) * np.sin(_a),
                     np.sin(_e)])
e1 = np.cross(view_dir, np.array([0.0, 0.0, 1.0]))
e1 /= max(np.linalg.norm(e1), 1e-9)
e2 = np.cross(e1, view_dir); e2 /= max(np.linalg.norm(e2), 1e-9)

rcca = None
for br in itv.vessel_tree.branches:
    if "RCCA" in str(getattr(br, "name", "")).upper():
        rcca = np.asarray(br.coordinates, dtype=float)
        break

# ------------------------------------------------------------------- draw
fig = plt.figure(figsize=(FIG_W, FIG_H))


def panel(i):
    """Axes whose inscribed square is PANEL_H and centred on CENTER_X[i]."""
    w = SQ[i]
    x = CENTER_X[i] - w / 2.0
    ax = fig.add_axes([x / FIG_W, PANEL_Y / FIG_H, w / FIG_W,
                       PANEL_H / FIG_H], projection="3d")
    # The rects overlap on purpose (see module docstring). An opaque axes
    # patch would then paint over the neighbour to its left — panel 1 lost
    # its right wall and y-axis to panel 2 that way.
    ax.patch.set_visible(False)
    return ax


def add_extras(ax, legend):
    if pp is not None and len(pp) > 1:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00", linewidth=2.8,
                zorder=6, label="planned path" if legend else None)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=110, marker="X",
               edgecolors="black", linewidths=0.7, zorder=8,
               label="target" if legend else None)
    ax.scatter(ins[0], ins[1], ins[2], color="black", s=70, marker="*",
               zorder=8, label="insertion (fixed)" if legend else None)


def style(ax, cube=True):
    if cube:
        ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1]); ax.set_zlim(*LIMS[2])
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (mm)", fontsize=AXIS_FS, labelpad=0)
    ax.set_ylabel("y (mm)", fontsize=AXIS_FS, labelpad=0)
    ax.set_zlabel("z (mm)", fontsize=AXIS_FS, labelpad=0)
    ax.tick_params(labelsize=TICK_FS)


def place_legend(ax, i, ncol=1, order=None, fontsize=LEGEND_FS, dx_in=0.0,
                 **kw):
    """Legend as a FIGURE artist, centred over its panel.

    An axes legend whose bbox_to_anchor is in figure coords still clips to
    the axes, which chopped the second column off panels 1-2.
    """
    h, l = ax.get_legend_handles_labels()
    if order is not None:
        h = [h[k] for k in order]
        l = [l[k] for k in order]
    fig.legend(h, l, loc="upper center", ncol=ncol, fontsize=fontsize,
               bbox_to_anchor=((CENTER_X[i] + dx_in) / FIG_W, LEGEND_Y),
               **kw)


# ---------------- 1: centerlines only
ax1 = panel(1)
for g in ("other", "rva", "onpath"):
    color, label = COL[g]
    for k, br in enumerate(groups[g]):
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        ax1.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                 alpha=0.55 if g == "other" else 0.92,
                 linewidth=1.0 if g == "other" else 2.0,
                 label=(label if k == 0 else None))
add_extras(ax1, legend=True)
style(ax1)
place_legend(ax1, 1, ncol=2)

# ---------------- 2: mesh + centerlines
ax2 = panel(0)
for g in ("other", "rva", "onpath"):
    if g not in meshes:
        continue
    v, f = meshes[g]
    color, label = COL[g]
    if f.size:
        ax2.add_collection3d(Poly3DCollection(
            v[f], facecolor=(*mcolors.to_rgb(color), MESH_ALPHA[g]),
            edgecolor=(0.25, 0.25, 0.3, 0.06), linewidth=0.15))
    ax2.plot([], [], color=color, lw=6, alpha=0.75, label=label)
for g in ("other", "rva", "onpath"):
    color, _ = COL[g]
    for br in groups[g]:
        c = np.asarray(br.coordinates)
        if c.size == 0:
            continue
        ax2.plot(c[:, 0], c[:, 1], c[:, 2], color=color,
                 alpha=0.95 if g != "other" else 0.5,
                 linewidth=1.7 if g != "other" else 0.9, zorder=5)
ax2.plot([], [], color="#555555", lw=1.7, label="centerlines (overlaid)")
add_extras(ax2, legend=True)
style(ax2)
place_legend(ax2, 0, ncol=2)

# ---------------- 3: devices + observation neighbourhood
ax3 = panel(2)
if rcca is not None:
    ax3.plot(rcca[:, 0], rcca[:, 1], rcca[:, 2], color="#1f77b4", lw=1.8,
             alpha=0.9)
ax3.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00", lw=2.6, zorder=4)
ax3.plot(cath_line[:, 0], cath_line[:, 1], cath_line[:, 2], color="#d62728",
         lw=6.0, solid_capstyle="round", alpha=0.95, zorder=5, label="catheter")
ax3.plot(gw_line[:, 0], gw_line[:, 1], gw_line[:, 2], color="#2ca02c", lw=3.0,
         solid_capstyle="round", zorder=6, label="guidewire")
ax3.scatter(*gw_tip, s=55, color="#2ca02c", edgecolors="black",
            linewidths=0.8, zorder=9)
ax3.scatter(*cath_tip, s=45, color="#d62728", edgecolors="black",
            linewidths=0.8, zorder=9)

theta = np.radians(np.linspace(ARC_DEG[0], ARC_DEG[1], 160))
ring_pts = []
for k, r in enumerate(PREVIEWS_MM):
    ring = (gw_tip[None, :]
            + RING_HALF_W * np.cos(theta)[:, None] * e1[None, :]
            + (RING_TALL * r) * np.sin(theta)[:, None] * e2[None, :])
    ring_pts.append(ring)
    ax3.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#6a3d9a", lw=1.1,
             ls="--", alpha=0.75, zorder=7,
             label=("obs. neighbourhood\n(10/20/40/80 mm)"
                    if k == 0 else None))
ax3.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=110, marker="X",
            edgecolors="black", linewidths=0.7, zorder=9)
ax3.scatter(ins[0], ins[1], ins[2], color="black", s=70, marker="*", zorder=9)

b3 = [pp, np.atleast_2d(tgt), np.atleast_2d(ins)] + ring_pts
if rcca is not None:
    b3.append(rcca)
Q = np.concatenate(b3, axis=0)
q0, q1 = Q.min(axis=0), Q.max(axis=0)
pad = 0.03 * float(np.max(q1 - q0))
lo, hi = q0 - pad, q1 + pad
ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
try:
    ax3.set_box_aspect(tuple(hi - lo))
except Exception:
    pass
style(ax3, cube=False)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("x (mm)", fontsize=AXIS_FS, labelpad=-16)
ax3.set_ylabel("y (mm)", fontsize=AXIS_FS, labelpad=-16)
ax3.set_zlabel("z (mm)", fontsize=AXIS_FS, labelpad=8)
place_legend(ax3, 2, ncol=2, order=[0, 2, 1], dx_in=-0.15,
             handlelength=1.6, columnspacing=1.2, handletextpad=0.6)
print("  devices extent mm  x %.0f  y %.0f  z %.0f" % tuple(hi - lo), flush=True)

TITLES = ["Vessel mesh + centerlines\n(what SOFA simulates and collides against)",
          "Centerlines\n(what the pathfinder and observations use)",
          "Devices at ~50 % insertion\n(guidewire leading, with observation rings)"]
for i, t in enumerate(TITLES):
    fig.text(CENTER_X[i] / FIG_W, TITLE_Y, t, fontsize=11, ha="center",
             va="top")
    # panel letter, left-aligned at the start of the panel rather than
    # inside the centred title, so it reads as a figure-part label
    fig.text(LABEL_X[i] / FIG_W, TITLE_Y, "(%s)" % "abc"[i],
             fontsize=LABEL_FS, fontweight="bold", ha="left", va="top")

fig.savefig(os.path.join(OUT, "three_panel_overview_biglegend.png"), dpi=150)
plt.close(fig)
print("wrote three_panel_overview_biglegend.png", flush=True)
