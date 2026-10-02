#!/usr/bin/env python3
"""Zoom slides: the same anatomy under two meshers, at the place where they differ.

Each row is one anatomy; each column one version. The window is a stretch of
the RCCA route centred where the difference lives -- where the v1 mesh pinches
the route shut, or where the real surface bulges past the tube -- and the mesh
is cropped to that window so the vessel wall is what fills the frame.

Runs in the container (pyvista):  python3 /opt/eve_training/saved/mesher_probe/slides_mesh_zoom_v2.py
"""
import json
import os
import sys
import textwrap

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/opt/eve_training/eve_bench")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from eve_bench.dualdevicenav import load_branches
from sdf_mesher import route_lumen

ROOTS = {"A": {"v1": "/opt/eve_training/topbrain_data/anatomies", "v2": "/opt/eve_training/topbrain_data/anatomies_v2", "v3": "/opt/eve_training/topbrain_data/anatomies_v3"},
         "B": {"v1": "/opt/eve_training/carotid_data/anatomies", "v2": "/opt/eve_training/carotid_data/anatomies_v2", "v3": "/opt/eve_training/carotid_data/anatomies_v3"}}
OUT = "/opt/eve_training/saved/mesher_probe/slides_v2"
os.makedirs(OUT, exist_ok=True)
TINT = {"v1": ((0.86, 0.25, 0.25, 0.50), "#c0392b", "v1  —  blurred 0/1 voxels, 0.5 contour, 3.7 k triangles"),
        "v2": ((0.20, 0.45, 0.80, 0.50), "#1f77b4", "v2  —  signed distance, 0 contour, 20 k triangles"),
        "v3": ((0.25, 0.65, 0.30, 0.50), "#2ca02c", "v3  —  v2 ∪ the patient's real segmented surface")}
plt.rcParams.update({"font.family": "DejaVu Sans"})


def load(setkey, name, ver):
    root = os.path.join(ROOTS[setkey][ver], name)
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    mesh = pv.read(os.path.join(root, "vessel_architecture_collision.obj")).extract_surface().triangulate()
    return br, route, rad, s, mesh


def rays_at(mesh, route, s, i, n_dir=16, max_r=12.0):
    t = route[min(i + 1, len(route) - 1)] - route[max(i - 1, 0)]; t /= max(np.linalg.norm(t), 1e-9)
    a = np.cross(t, [1.0, 0, 0]); a = a if np.linalg.norm(a) > 0.2 else np.cross(t, [0, 1.0, 0]); a /= np.linalg.norm(a); b = np.cross(t, a)
    rr = []
    for th in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        u = np.cos(th) * a + np.sin(th) * b
        pts, _ = mesh.ray_trace(route[i], route[i] + u * max_r, first_point=True)
        if len(pts):
            rr.append(float(np.linalg.norm(np.asarray(pts).reshape(-1)[:3] - route[i])))
    return np.asarray(rr)


def bulge_station(mesh, route, rad, s, lo, hi, step=2.0):
    """Arclength in [lo, hi] where the longest wall ray / MISR is largest."""
    best = (0.0, lo)
    for st in np.arange(lo + 3.0, hi - 3.0, step):
        i = int(np.searchsorted(s, st))
        if i < 2 or i >= len(route) - 2:
            continue
        rr = rays_at(mesh, route, s, i)
        if len(rr) >= 12:
            ratio = rr.max() / max(rad[i], 1e-6)
            if ratio > best[0]:
                best = (ratio, st)
    return best[1], best[0]


def window(route, s, s0, half, margin):
    """One frame per row: limits and camera from the v2 route, shared by every version."""
    m = (s >= s0 - half) & (s <= s0 + half)
    seg = route[m]
    lo, hi = seg.min(0) - margin, seg.max(0) + margin
    ext = np.maximum(hi - lo, 6.0)
    i0 = min(int(np.searchsorted(s, s0)), len(route) - 1)
    t = route[min(i0 + 4, len(route) - 1)] - route[max(i0 - 4, 0)]
    az = np.degrees(np.arctan2(t[1], t[0])) + 90.0
    return dict(lo=lo, ext=ext, az=az)


def panel(ax, setkey, name, ver, s0, win, half=10.0, show_ratio=False):
    """Draw one version inside a shared window; return the numbers for the caption."""
    br, route, rad, s, mesh = load(setkey, name, ver)
    s0 = min(s0, s[-1] - 2.0)
    lo, ext = win["lo"], win["ext"]
    hi = lo + ext
    v = np.asarray(mesh.points); f = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    cen = v[f].mean(1)
    keep = np.all((cen >= lo - 1.0) & (cen <= hi + 1.0), axis=1)
    seg = route[(s >= s0 - half) & (s <= s0 + half)]
    face, line, _ = TINT[ver]
    ax.add_collection3d(Poly3DCollection(v[f[keep]], facecolor=face[:3] + (0.62,), edgecolor=(0.12, 0.12, 0.18, 0.28), linewidth=0.25))
    ax.plot(*seg.T, color="#111111", lw=1.8, zorder=5)
    i0 = min(int(np.searchsorted(s, s0)), len(route) - 1)
    ax.scatter(*route[i0], s=70, color="#ffd700", edgecolor="k", zorder=6, depthshade=False)
    for k in range(3):
        getattr(ax, "set_%slim" % "xyz"[k])(lo[k], hi[k])
    try:
        ax.set_box_aspect(tuple(ext), zoom=1.3)
    except TypeError:
        try:
            ax.set_box_aspect(tuple(ext))
        except Exception:
            pass
    ax.view_init(elev=18, azim=win["az"])
    ax.set_axis_off()
    d, ins, body, _ = route_lumen(mesh, route)
    w = (s >= s0 - half) & (s <= s0 + half) & body
    lum = d[w & ins]
    out = int((w & ~ins).sum())
    lmin = float(lum.min()) if len(lum) else 0.0
    txt = "min lumen in window  %.2f mm   (declared %.2f)" % (lmin, rad[i0])
    if out:
        txt += "\n%d route points outside the mesh" % out
    if lmin < 0.3:
        txt += "\nwall collapsed onto the centerline"
    if show_ratio and ver in ("v2", "v3"):
        rr = rays_at(mesh, route, s, i0)
        if len(rr) >= 12:
            txt += "\nlongest wall ray / MISR at the dot  %.2f" % (rr.max() / max(rad[i0], 1e-6))
    return txt, line


def slide(fname, title, subtitle, rows, cols, foot, show_ratio=False):
    n, m = len(rows), len(cols)
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.962, title, ha="center", fontsize=19, weight="bold", color="#1f4e79")
    x0, w_all = 0.175, 0.815
    for j, ver in enumerate(cols):
        fig.text(x0 + (j + 0.5) * w_all / m, 0.878, TINT[ver][2], ha="center", fontsize=12, weight="bold", color=TINT[ver][1])
    top, bot = 0.865, 0.075
    h = (top - bot) / n
    cap_h = 0.055
    for i, (setkey, name, s0, half, label) in enumerate(rows):
        fig.text(0.012, top - i * h - h / 2, ("set %s\n%s\n%s" % (setkey, name, label)).replace("__", "__\n"),
                 ha="left", va="center", fontsize=9.6, color="#333333", linespacing=1.35)
        _, route, _, s, _ = load(setkey, name, "v2")
        win = window(route, s, s0, half, 4.0)
        for j, ver in enumerate(cols):
            x = x0 + j * w_all / m
            ax = fig.add_axes([x, top - (i + 1) * h + cap_h, w_all / m - 0.006, h - cap_h - 0.004], projection="3d")
            ax.set_facecolor("white")
            txt, line = panel(ax, setkey, name, ver, s0, win, half, show_ratio)
            fig.text(x + w_all / (2 * m), top - (i + 1) * h + 0.006, txt, ha="center", va="bottom", fontsize=9.2, color=line, linespacing=1.25,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=line, alpha=0.95))
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=140, facecolor="white"); plt.close(fig); print("wrote", p, flush=True)


def main():
    # ----- where the bulges are, found on the v3 meshes
    def bulge(setkey, name, section):
        br, route, rad, s, mesh = load(setkey, name, "v3")
        if section == "siphon":
            lo, hi = 130.0, s[-1]
        else:
            prov = json.load(open(os.path.join(ROOTS[setkey]["v3"], name, "provenance.json")))
            lo, hi = prov["host_cut_mm"], prov["host_cut_mm"] + prov["cca_mm"] + prov["ica_mm"]
        st, ratio = bulge_station(mesh, route, rad, s, lo, hi)
        print("  %-40s %-6s bulge at %.0f mm, ratio %.2f" % (name, section, st, ratio), flush=True)
        return st

    b_003L = bulge("A", "topcow_mr_003_L", "siphon")
    b_008 = bulge("A", "topcow_mr_008", "siphon")
    b_m030 = bulge("B", "case_m_030_right__topcow_mr_016", "lower")
    b_w021 = bulge("B", "case_w_021_right__topcow_mr_008_L", "siphon")

    slide("slide_zoom_v1_vs_v2.png",
          "v1 → v2:  the vessel that was not there",
          "Three anatomies at the station where the v1 mesh pinched the route shut.  Black line = centerline, ● = station, box = 24 mm of route.  "
          "Declared radius at these points is 1.4–1.7 mm — healthy calibre; the label is fine, the blur is not.",
          [("A", "topcow_mr_023_L", 206.0, 12.0, "v1 lumen 0.06 mm\nat 206 mm"),
           ("A", "topcow_mr_024", 195.0, 12.0, "v1 lumen 0.08 mm\nat 195 mm"),
           ("B", "case_w_008_right__topcow_mr_023_L", 205.0, 12.0, "v1 lumen 0.01 mm\nat 205 mm")],
          ["v1", "v2"],
          "v1 paints 0/1 voxels, blurs twice and traces 0.5; anything under ~1.2 mm never reaches 0.5 and drops out, and everything else loses ~0.65 mm of radius.  "
          "v2 stores the signed distance and traces 0: the same centerline, the same radii, and the wall lands where the data says.")

    slide("slide_zoom_v2_vs_v3.png",
          "v2 → v3:  the shape that was not there",
          "Same centerlines, same radii, same triangle budget.  Window centred on the station where the real segmented surface reaches furthest beyond the inscribed radius.",
          [("A", "topcow_mr_003_L", b_003L, 12.0, "siphon\nreal ICA label surface"),
           ("A", "topcow_mr_008", b_008, 12.0, "siphon\nreal ICA label surface"),
           ("B", "case_m_030_right__topcow_mr_016", b_m030, 12.0, "carotid bulb\nreal lumen STL")],
          ["v2", "v3"],
          "v2 is a circular tube of radius MISR everywhere — the largest circle that fits, never the lumen.  v3 unions the patient's own surface in where the graft kept it: "
          "the far wall of a bulge, the eccentric sections, the bulb.  The tube stays underneath as a floor, so nothing gets narrower.")

    slide("slide_zoom_v1_vs_v3.png",
          "v1 → v3:  shipped mesh against the rebuilt one",
          "Both defects on the same anatomies: v1 loses the vessel where it is thin and the shape where it is real; v3 has the wall at the data's calibre and the patient's own cross-section.",
          [("A", "topcow_mr_024", 195.0, 12.0, "severed in v1\nat 195 mm"),
           ("B", "case_w_012_right__topcow_mr_023_L", 207.0, 12.0, "severed in v1\nat 207 mm"),
           ("B", "case_w_021_right__topcow_mr_008_L", b_w021, 12.0, "siphon bulge\nreal surface in v3")],
          ["v1", "v3"],
          "Navigable anatomies: v1 93 of 264, v3 272 of 272.  Median radius deficit 0.65 mm → 0.10 mm.  Real sections carry cross-sections 1.13–1.25× their inscribed radius; a tube carries 1.0.")


if __name__ == "__main__":
    main()
