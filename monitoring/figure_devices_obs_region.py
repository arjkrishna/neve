#!/usr/bin/env python3
"""Devices + observation-neighbourhood schematic.

Like the LEFT (centerline) panel of hero_centerline_vs_mesh.png, but:
  * only the RCCA route centerline is drawn (no other branches) — uncluttered
  * the guidewire and catheter are drawn inserted to ~50 % of the planned
    path, guidewire leading the catheter by a short distance
  * concentric rings around the guidewire tip mark the region the
    observation is derived from, at the ACTUAL lookahead distances used by
    eve.observation.LocalGuidance: path previews at 10 / 20 / 40 / 80 mm
    (features 13-16, 39-42) and the 20 mm curvature / calibre lookahead
  * each arc is labelled with the preview distance it stands for

The device poses are schematic — both devices are drawn along the planned
path (a real SOFA pose hugs the wall and buckles); the figure is about where
the observation comes from, not about a particular simulated state.

Writes devices_obs_region.png (new file; overwrites nothing).

    docker run --rm <mounts> eve-training-fixed \
      python3 /opt/eve_training/monitoring/figure_devices_obs_region.py /results/figs
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

SEED, TARGET_SEED = 12345, 7
ELEV, AZIM = 16, -68
GW_FRACTION = 0.50          # guidewire tip at ~half the planned path
GW_LEAD_MM = 14.0           # guidewire ahead of the catheter tip
PREVIEWS_MM = [10.0, 20.0, 40.0, 80.0]     # LocalGuidance preview distances
# Ring shape: narrow elongated arcs over the tip rather than closed circles —
# the observation looks AHEAD along the path, so the lower half carried no
# meaning and added clutter. All arcs share ONE half-width (that of the +10
# arc) and nest by HEIGHT only, so distance is read vertically.
RING_HALF_W = 14.5      # screen-horizontal semi-axis, mm — same for every arc
RING_TALL = 0.62        # screen-vertical semi-axis, x the preview distance
ARC_DEG = (18.0, 162.0)  # arc span (deg); 0/180 = horizontal, 90 = straight up
LABEL_OUT = 5.0         # label gap to the right of each arc, mm

# ------------------------------------------------------------------ build
itv = DualDeviceNavRCCAVaried(seed=SEED, episodes_between_change=10)
itv.vessel_tree.reset(episode_nr=0)
itv.target.reset(episode_nr=0, seed=TARGET_SEED)
pf = eve.pathfinder.FixedPathfinder(intervention=itv)
try:
    pf.reset(episode_nr=0)
except Exception:
    pass

pp = np.asarray(pf.path_points_vessel_cs, dtype=float)
seg = np.linalg.norm(np.diff(pp, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
total = float(s[-1])


def point_at(arc):
    """Interpolated point on the planned path at arclength `arc`."""
    arc = float(np.clip(arc, 0.0, total))
    i = int(np.searchsorted(s, arc, side="right") - 1)
    i = min(max(i, 0), len(pp) - 2)
    d = s[i + 1] - s[i]
    t = 0.0 if d <= 1e-9 else (arc - s[i]) / d
    return pp[i] + t * (pp[i + 1] - pp[i])


def polyline_to(arc):
    """Path polyline from the insertion up to arclength `arc`."""
    m = s <= arc
    out = pp[m]
    tip = point_at(arc)
    if len(out) == 0 or np.linalg.norm(out[-1] - tip) > 1e-9:
        out = np.vstack([out, tip]) if len(out) else np.atleast_2d(tip)
    return out


gw_s = GW_FRACTION * total
cath_s = max(0.0, gw_s - GW_LEAD_MM)
gw_line = polyline_to(gw_s)
cath_line = polyline_to(cath_s)
gw_tip = point_at(gw_s)

# Ring orientation: FACE THE CAMERA (billboard). Orienting them
# perpendicular to the vessel tangent put them nearly edge-on at this view
# angle, so they rendered as flat lines instead of circles.
_e, _a = np.radians(ELEV), np.radians(AZIM)
view_dir = np.array([np.cos(_e) * np.cos(_a),
                     np.cos(_e) * np.sin(_a),
                     np.sin(_e)])
_up = np.array([0.0, 0.0, 1.0])
e1 = np.cross(view_dir, _up); e1 /= max(np.linalg.norm(e1), 1e-9)
e2 = np.cross(e1, view_dir);  e2 /= max(np.linalg.norm(e2), 1e-9)

tgt = tracking3d_to_vessel_cs(
    np.asarray(itv.target.coordinates3d, dtype=float),
    itv.fluoroscopy.image_rot_zx, itv.fluoroscopy.image_center)
ins = np.asarray(itv.vessel_tree.insertion.position, dtype=float)

# RCCA centerline only
rcca = None
for br in itv.vessel_tree.branches:
    if "RCCA" in str(getattr(br, "name", "")).upper():
        rcca = np.asarray(br.coordinates, dtype=float)
        break

# ------------------------------------------------------------------ draw
fig = plt.figure(figsize=(5.0, 7.6))
ax = fig.add_axes([0.00, 0.035, 0.97, 0.845], projection="3d")

if rcca is not None:
    ax.plot(rcca[:, 0], rcca[:, 1], rcca[:, 2], color="#1f77b4", lw=1.8,
            alpha=0.9, label="RCCA centerline")
ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="#ffaa00", lw=2.6, zorder=4,
        label="planned path")

# devices: catheter first (thicker, behind), guidewire on top
ax.plot(cath_line[:, 0], cath_line[:, 1], cath_line[:, 2], color="#d62728",
        lw=6.0, solid_capstyle="round", alpha=0.95, zorder=5, label="catheter")
ax.plot(gw_line[:, 0], gw_line[:, 1], gw_line[:, 2], color="#2ca02c",
        lw=3.0, solid_capstyle="round", zorder=6, label="guidewire")
ax.scatter(*gw_tip, s=55, color="#2ca02c", edgecolors="black",
           linewidths=0.8, zorder=9)
cath_tip = point_at(cath_s)
ax.scatter(*cath_tip, s=45, color="#d62728", edgecolors="black",
           linewidths=0.8, zorder=9)

# concentric rings around the guidewire tip, at the observation lookaheads
theta = np.radians(np.linspace(ARC_DEG[0], ARC_DEG[1], 160))
# -e1 is screen-right at this ELEV/AZIM (verified against the render); each
# label sits level with its own arc's crown, clear of the path on the right.
e_right = -e1
arc_label_xyz = []
ring_pts = []
for k, r in enumerate(PREVIEWS_MM):
    w, h = RING_HALF_W, RING_TALL * r
    ring = (gw_tip[None, :]
            + w * np.cos(theta)[:, None] * e1[None, :]
            + h * np.sin(theta)[:, None] * e2[None, :])
    ring_pts.append(ring)
    arc_label_xyz.append(gw_tip + (w + LABEL_OUT) * e_right + h * e2)
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#6a3d9a", lw=1.1,
            ls="--", alpha=0.75, zorder=7,
            label=("observation neighbourhood\n(10 / 20 / 40 / 80 mm)"
                   if k == 0 else None))


ax.scatter(tgt[0], tgt[1], tgt[2], color="#ffd400", s=110, marker="X",
           edgecolors="black", linewidths=0.7, zorder=9, label="target")
ax.scatter(ins[0], ins[1], ins[2], color="black", s=70, marker="*",
           zorder=9, label="insertion (fixed)")

# Frame TIGHT on what is actually drawn. An equal-sided cube gave x and y
# the RCCA's ~200 mm z-extent, so most of the frame width was empty. Instead
# take per-axis limits and hand the data ranges to set_box_aspect: mm-per-unit
# stays identical on all three axes, so the arcs and the geometry are still
# undistorted, but the box is as narrow as the content.
bounds = [pp, np.atleast_2d(tgt), np.atleast_2d(ins),
          np.atleast_2d(np.asarray(arc_label_xyz))]
bounds.extend(ring_pts)
if rcca is not None:
    bounds.append(rcca)
P = np.concatenate(bounds, axis=0)
mins, maxs = P.min(axis=0), P.max(axis=0)
pad = 0.03 * float(np.max(maxs - mins))
lo, hi = mins - pad, maxs + pad
ax.set_xlim(lo[0], hi[0])
ax.set_ylim(lo[1], hi[1])
ax.set_zlim(lo[2], hi[2])
try:
    ax.set_box_aspect(tuple(hi - lo))       # equal scale, narrow box
except Exception:
    pass
print("extent mm  x %.0f  y %.0f  z %.0f" % tuple(hi - lo), flush=True)
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_xlabel("x (mm)", fontsize=9, labelpad=-16)
ax.set_ylabel("y (mm)", fontsize=9, labelpad=-16)
ax.set_zlabel("z (mm)", fontsize=9)
ax.tick_params(labelsize=7)
# x/y tick VALUES carry nothing here — the figure is about the geometry,
# not about absolute vessel-CS coordinates; z is kept for the depth scale.
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(0.5, 0.945), ncol=2, fontsize=7)

fig.suptitle("Guidewire / catheter pose and the observation neighbourhood",
             fontsize=11, y=0.99)
fig.savefig(os.path.join(OUT, "devices_obs_region.png"), dpi=165)
plt.close(fig)
print(f"planned path {total:.0f} mm | guidewire tip at {gw_s:.0f} mm "
      f"| catheter tip at {cath_s:.0f} mm", flush=True)
print("wrote devices_obs_region.png", flush=True)
