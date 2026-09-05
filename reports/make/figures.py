#!/usr/bin/env python3
"""Figures for the two meshing reports. Everything is drawn from the data the
analysis produced (saved/mesher_probe/*, the per-anatomy mesh_v2/v3.json
reports) or from the numbers measured in the analysis, restated here.

    python reports/make/figures.py        -> reports/figs/*.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"v1": "#c0392b", "v2": "#1f77b4", "v3": "#2ca02c", "host": "#7f7f7f", "tube": "#1f77b4", "real": "#ff7f0e"}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. pipeline flow
def pipeline_flow():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.set_axis_off()
    boxes = [("A  label map\n(voxels, 0.3 x 0.3 x 0.6 mm)", 0.2, "#e8f1fb"),
             ("A  surface\n(marching cubes + smoothing)", 2.35, "#e8f1fb"),
             ("B  centerline + MISR\n(VMTK)", 4.5, "#e8f1fb"),
             ("C  graft onto host\n(frames, ramps, floors)", 6.65, "#fdf0e0"),
             ("D  collision mesh\n(v1 / v2 / v3 mesher)", 8.8, "#e9f7e9")]
    for txt, x, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.95, 1.25, boxstyle="round,pad=0.04", fc=col, ec="#555555", lw=1))
        ax.text(x + 0.975, 1.62, txt, ha="center", va="center", fontsize=8.6)
    for x in (2.15, 4.3, 6.45, 8.6):
        ax.add_patch(FancyArrowPatch((x, 1.62), (x + 0.2, 1.62), arrowstyle="-|>", mutation_scale=14, color="#333333"))
    ax.add_patch(FancyBboxPatch((8.8, 0.05), 1.95, 0.6, boxstyle="round,pad=0.04", fc="#f4f4f4", ec="#888888", lw=0.8))
    ax.text(9.775, 0.35, "E  SOFA: triangle + line\ncollision, contact 0.3 mm", ha="center", va="center", fontsize=7.8)
    ax.add_patch(FancyArrowPatch((9.775, 1.0), (9.775, 0.68), arrowstyle="-|>", mutation_scale=12, color="#333333"))
    ax.text(4.5, 2.75, "the segmented SURFACE is used only to find the centerline; the mesh SOFA loads is rebuilt from centerline + radii",
            ha="center", fontsize=8.5, color="#a33", style="italic")
    ax.text(1.2, 0.45, "Zenodo carotids arrive at B already:\nlumen STL + VMTK centerline tree", ha="center", fontsize=7.8, color="#555")
    save(fig, "pipeline_flow.png")


# ---------------------------------------------------------------- 2. what a blur does to a disc
def disc_profiles(sigma_mm=0.85, px=0.02):
    """Radial profile of a binary disc after an isotropic Gaussian blur."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, r in zip(axes, (1.2, 2.0, 4.0)):
        n = int(2 * (r + 4 * sigma_mm) / px) + 1
        xs = (np.arange(n) - n // 2) * px
        X, Y = np.meshgrid(xs, xs)
        disc = ((X ** 2 + Y ** 2) < r * r).astype(float)
        blur = ndimage.gaussian_filter(disc, sigma_mm / px)
        rho = np.abs(xs)
        prof = blur[n // 2]
        ax.plot(rho, disc[n // 2], color="#999999", lw=1.2, label="binary tube (true wall at r)")
        ax.plot(rho, prof, color=C["v1"], lw=2, label="after Gaussian blur, sigma 0.85 mm")
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        cross = rho[(prof >= 0.5)].max() if (prof >= 0.5).any() else 0.0
        ax.axvline(r, color="#999999", lw=0.8, ls=":")
        ax.axvline(cross, color=C["v1"], lw=0.8, ls=":")
        ax.annotate("", xy=(cross, 0.62), xytext=(r, 0.62), arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
        ax.text((r + cross) / 2, 0.66, "%.2f mm lost" % (r - cross), ha="center", fontsize=8.5)
        ax.set_title("declared r = %.1f mm\nmeshed wall at %.2f mm" % (r, cross), fontsize=9)
        ax.set_xlabel("distance from centerline (mm)"); ax.set_xlim(0, r + 2.2)
        ax.text(0.15, 0.53, "iso-level 0.5", fontsize=8)
    axes[0].set_ylabel("field value"); axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("v1 mesher: blurring a binary tube and cutting at 0.5 moves the wall inward, by more the thinner the vessel", fontsize=10, y=1.04)
    save(fig, "v1_blur_profiles.png")

    # SDF comparison at r = 1.2
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    r = 1.2
    rho = np.linspace(0, r + 2.0, 400)
    ax.plot(rho, r - rho, color=C["v2"], lw=2, label="v2: signed distance  f = r - distance")
    n = int(2 * (r + 4 * sigma_mm) / px) + 1
    xs = (np.arange(n) - n // 2) * px; X, Y = np.meshgrid(xs, xs)
    blur = ndimage.gaussian_filter(((X ** 2 + Y ** 2) < r * r).astype(float), sigma_mm / px)[n // 2]
    ax.plot(np.abs(xs), blur, color=C["v1"], lw=2, label="v1: blurred binary tube")
    ax.axhline(0, color=C["v2"], lw=0.8, ls="--"); ax.axhline(0.5, color=C["v1"], lw=0.8, ls="--")
    ax.axvline(r, color="#999999", lw=0.9, ls=":"); ax.text(r + 0.03, -0.9, "true wall\nr = 1.2 mm", fontsize=8, color="#555")
    ax.set_xlabel("distance from centerline (mm)"); ax.set_ylabel("field value"); ax.set_ylim(-1.2, 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Where each mesher puts the wall: the zero of the distance field is at r exactly;\nthe 0.5 level of the blurred field is inside it", fontsize=9.5)
    save(fig, "v2_sdf_vs_blur.png")


# ---------------------------------------------------------------- 3. measured tube erosion
def tube_erosion():
    r = np.array([1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0])
    v1z = np.array([np.nan, 0.66, 0.78, 1.11, 1.62, 2.22, 2.83, 3.69])
    v1x = np.array([np.nan, np.nan, 0.47, 0.72, 1.41, 2.05, 2.66, 3.64])
    sdf = np.array([0.92, 1.14, 1.35, 1.54, 1.96, 2.47, 2.97, 3.98])
    sdf3 = np.array([0.97, 1.17, 1.37, 1.57, 1.97, 2.47, 2.96, 3.96])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot([0, 4.3], [0, 4.3], color="#aaaaaa", lw=1, ls="--", label="perfect (meshed = declared)")
    ax.plot(r, v1z, "o-", color=C["v1"], label="v1 mesher, vessel running superior (z)")
    ax.plot(r, v1x, "s--", color="#e07b39", label="v1 mesher, vessel running in the axial plane (x)")
    ax.plot(r, sdf, "^-", color=C["v2"], label="signed distance, same 0.6/0.6/0.9 mm grid")
    ax.plot(r, sdf3, "v-", color=C["v3"], label="signed distance, 0.3 mm grid")
    ax.text(1.05, 0.15, "v1: absent below\n~1.2 mm (z) / ~1.4 mm (x)", fontsize=8, color=C["v1"])
    ax.set_xlabel("declared radius (mm)"); ax.set_ylabel("meshed inscribed radius (mm)"); ax.set_xlim(0.8, 4.3); ax.set_ylim(0, 4.3)
    ax.legend(fontsize=7.5, loc="upper left"); ax.set_title("Straight test tubes through each mesher", fontsize=10)
    ax = axes[1]
    ax.plot(r, r - v1z, "o-", color=C["v1"], label="v1, z"); ax.plot(r, r - v1x, "s--", color="#e07b39", label="v1, x")
    ax.plot(r, r - sdf, "^-", color=C["v2"], label="SDF, 0.6/0.9 grid"); ax.plot(r, r - sdf3, "v-", color=C["v3"], label="SDF, 0.3 grid")
    ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
    ax.set_xlabel("declared radius (mm)"); ax.set_ylabel("radius lost (mm)"); ax.legend(fontsize=7.5); ax.set_title("Radius the mesh loses", fontsize=10)
    save(fig, "tube_erosion_measured.png")


# ---------------------------------------------------------------- 4. SOFA cost vs budget
def sofa_budget():
    tris = np.array([3709, 6000, 9000, 12000, 16000, 20000, 37628])
    adv = np.array([129, 136, 159, 205, 203, 202, 219]); con = np.array([140, 163, 193, 230, 229, 209, 234])
    dtris = np.array([3708, 6000, 9000, 12000, 20000]); deficit = np.array([0.49, 0.33, 0.23, 0.19, 0.12]); lmin = np.array([0.24, 0.36, 0.59, 0.62, 0.75])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(tris, adv, "o-", color="#1f77b4", label="free advance"); ax.plot(tris, con, "s-", color="#d62728", label="pushing / twisting at the wall")
    ax.axvline(3709, color="#999", ls=":", lw=1); ax.text(3900, 245, "v1 budget", fontsize=8, color="#666")
    ax.axvline(20000, color="#999", ls=":", lw=1); ax.text(20500, 245, "v2/v3 budget", fontsize=8, color="#666")
    ax.set_xscale("log"); ax.set_xlabel("collision-mesh triangles"); ax.set_ylabel("SOFA step time (ms)"); ax.set_ylim(100, 260)
    ax.legend(fontsize=8); ax.set_title("Simulator cost vs triangle count (topcow_mr_001)", fontsize=10)
    ax = axes[1]
    ax.plot(dtris, deficit, "o-", color=C["v2"], label="median radius deficit on the route")
    ax.plot(dtris, lmin, "s--", color="#ff7f0e", label="minimum lumen radius on the route")
    ax.axhline(0.35 + 0.3, color="#999", ls="--", lw=1); ax.text(3800, 0.67, "catheter radius + contact distance = 0.65 mm", fontsize=7.5, color="#666")
    ax.set_xscale("log"); ax.set_xlabel("collision-mesh triangles (SDF mesh, quadric decimation)"); ax.set_ylabel("mm"); ax.set_ylim(0, 0.9)
    ax.legend(fontsize=8); ax.set_title("Lumen fidelity vs triangle count (same anatomy)", fontsize=10)
    save(fig, "sofa_cost_vs_budget.png")


# ---------------------------------------------------------------- 5. per-anatomy distributions
def load_set(kind, prefix):
    v1 = json.load(open(os.path.join(ROOT, "saved/mesher_probe/lumen_v1.json")))
    v1 = {k: v for k, v in v1.items() if k.startswith(prefix)}
    root2 = os.path.join(ROOT, "topbrain_data/anatomies_v2" if kind == "A" else "carotid_data/anatomies_v2")
    root3 = os.path.join(ROOT, "topbrain_data/anatomies_v3" if kind == "A" else "carotid_data/anatomies_v3")
    v2 = {os.path.basename(os.path.dirname(f)): json.load(open(f))["obj"] for f in glob.glob(os.path.join(root2, "*", "mesh_v2.json"))}
    v3 = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in glob.glob(os.path.join(root3, "*", "mesh_v3.json"))}
    return v1, v2, v3


def lumen_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (kind, prefix, title) in zip(axes, (("A", "topcow", "Set A - TopBrain siphons (49)"), ("B", "case_", "Set B - carotid three-source (215 / 223)"))):
        v1, v2, v3 = load_set(kind, prefix)
        data = [[r.get("lumen_min_mm", 0) for r in v1.values()], [r.get("lumen_min_mm", 0) for r in v2.values()], [r["obj"].get("lumen_min_mm", 0) for r in v3.values()]]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.5))
        for patch, col in zip(bp["boxes"], (C["v1"], C["v2"], C["v3"])):
            patch.set_facecolor(col); patch.set_alpha(0.35)
        for i, d in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.06, len(d)), d, s=6, color=(C["v1"], C["v2"], C["v3"])[i], alpha=0.5)
        ax.axhline(0.65, color="#999", ls="--", lw=1); ax.text(0.55, 0.68, "0.65 mm: catheter radius + contact distance", fontsize=7.5, color="#666")
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["v1", "v2", "v3"]); ax.set_ylabel("minimum lumen radius on the route (mm)")
        nav = [sum(1 for r in v1.values() if r.get("navigable")), sum(1 for r in v2.values() if r.get("navigable")), sum(1 for r in v3.values() if r["obj"].get("navigable"))]
        ax.set_title("%s\nnavigable: v1 %d / v2 %d / v3 %d" % (title, *nav), fontsize=9.5)
        ax.set_ylim(0, max(max(d) for d in data) * 1.1)
    save(fig, "lumen_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (kind, prefix, title) in zip(axes, (("A", "topcow", "Set A"), ("B", "case_", "Set B"))):
        v1, v2, v3 = load_set(kind, prefix)
        data = [[r.get("median_deficit_mm", 0) for r in v1.values()], [r.get("median_deficit_mm", 0) for r in v2.values()], [r["obj"].get("median_deficit_mm", 0) for r in v3.values()]]
        for i, (d, col) in enumerate(zip(data, (C["v1"], C["v2"], C["v3"]))):
            ax.hist(d, bins=np.linspace(-0.1, 1.0, 45), color=col, alpha=0.55, label=("v1", "v2", "v3")[i] + "  median %.2f" % np.median(d))
        ax.set_xlabel("median radius deficit along the route (mm)"); ax.set_ylabel("anatomies"); ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    save(fig, "deficit_distributions.png")


def shape_ratios():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (kind, prefix, title) in zip(axes, (("A", "topcow", "Set A v3"), ("B", "case_", "Set B v3"))):
        _, _, v3 = load_set(kind, prefix)
        groups = [("tube", "host tube\nsections")] + ([("lower", "real carotid\nCCA-ICA")] if kind == "B" else []) + [("siphon", "real ICA\nsiphon")]
        data, labels, cols = [], [], []
        for key, lab in groups:
            vals = [r["shape"][key]["max_over_misr"]["median"] for r in v3.values() if key in r.get("shape", {})]
            data.append(vals); labels.append(lab); cols.append(C["tube"] if key == "tube" else C["real"])
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.4)
        for i, d in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.05, len(d)), d, s=6, color=cols[i], alpha=0.5)
        ax.axhline(1.0, color="#999", ls="--", lw=1); ax.text(0.55, 1.01, "perfect circle of radius MISR", fontsize=7.5, color="#666")
        ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("longest wall ray / MISR (median per anatomy)"); ax.set_title(title + ": how non-circular the meshed lumen is", fontsize=9.5)
    save(fig, "shape_ratios.png")


# ---------------------------------------------------------------- 6. schematics
def cross_section_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    th = np.linspace(0, 2 * np.pi, 400)
    # a real, slightly eccentric lumen with a bulge
    rx, ry = 2.6, 1.9
    bul = 1 + 0.35 * np.exp(-((th - 0.9) ** 2) / 0.12)
    x, y = rx * np.cos(th) * bul, ry * np.sin(th) * bul
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.6, 3.6); ax.set_axis_off()
    ax = axes[0]
    ax.fill(x, y, color="#f7d5d5"); ax.plot(x, y, color="#c0392b", lw=1.6)
    ax.plot(0.15, -0.05, "ko", ms=4); ax.text(0.3, -0.35, "centerline\npoint", fontsize=8)
    ax.text(0, 3.2, "a real vessel cross-section (the LUMEN)", ha="center", fontsize=9.5)
    ax.text(0, -3.3, "calibre = how wide it is;\nnot one number for a shape like this", ha="center", fontsize=8, color="#555")
    ax = axes[1]
    ax.fill(x, y, color="#f7d5d5"); ax.plot(x, y, color="#c0392b", lw=1.6)
    # inscribed circle centred at the centerline point: radius = min distance to the wall
    d = np.sqrt((x - 0.15) ** 2 + (y + 0.05) ** 2); misr = d.min()
    ax.add_patch(Circle((0.15, -0.05), misr, fc="none", ec="#1f77b4", lw=2))
    ax.plot(0.15, -0.05, "ko", ms=4)
    j = d.argmin(); ax.plot([0.15, x[j]], [-0.05, y[j]], color="#1f77b4", lw=1.2)
    ax.text(0, 3.2, "MISR: the largest sphere that fits\n(radius = distance to the NEAREST wall)", ha="center", fontsize=9.5)
    ax.text(0, -3.3, "MISR = %.2f mm here; this is what the\ncenterline file stores as the 'radius'" % misr, ha="center", fontsize=8, color="#555")
    ax = axes[2]
    ax.fill(x, y, color="#f7d5d5"); ax.plot(x, y, color="#c0392b", lw=1.6)
    ax.add_patch(Circle((0.15, -0.05), misr, fc="none", ec="#1f77b4", lw=2, ls="--"))
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))); r_eq = np.sqrt(area / np.pi)
    ax.add_patch(Circle((0.15, -0.05), r_eq, fc="none", ec="#2ca02c", lw=2))
    ax.text(0, 3.2, "what a TUBE mesh keeps vs what is real", ha="center", fontsize=9.5)
    ax.text(0, -3.3, "blue dashed: the tube built from MISR (%.2f)\ngreen: a circle of the same AREA (%.2f) -- the far wall is lost" % (misr, r_eq), ha="center", fontsize=8, color="#555")
    save(fig, "cross_section_schematic.png")


def v3_union_schematic():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    th = np.linspace(0, 2 * np.pi, 400)
    rx, ry = 2.4, 1.7; bul = 1 + 0.4 * np.exp(-((th - 1.0) ** 2) / 0.1); neck = 1 - 0.55 * np.exp(-((th - 4.2) ** 2) / 0.08)
    x, y = rx * np.cos(th) * bul * neck, ry * np.sin(th) * bul * neck
    r_tube = 1.5; r_cap = 1.8 * r_tube + 1.0
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_axis_off()
    axes[0].add_patch(Circle((0, 0), r_tube, fc="#d6e6f5", ec="#1f77b4", lw=2)); axes[0].set_title("1. floored tube (v2)\nradius = max(MISR, 1.0 mm)", fontsize=8.5)
    axes[1].fill(x, y, color="#fde4cc"); axes[1].plot(x, y, color="#ff7f0e", lw=2); axes[1].set_title("2. real segmented lumen\n(a bulge, and a neck)", fontsize=8.5)
    axes[2].add_patch(Circle((0, 0), r_cap, fc="none", ec="#7f7f7f", lw=1.6, ls="--")); axes[2].fill(x, y, color="#fde4cc"); axes[2].plot(x, y, color="#ff7f0e", lw=2)
    axes[2].set_title("3. capsule (dashed): 1.8 x MISR + 1 mm\nclips what the graft did not keep", fontsize=8.5)
    ax = axes[3]
    # union = points inside tube OR inside (real AND capsule)
    g = np.linspace(-5, 5, 400); X, Y = np.meshgrid(g, g)
    from matplotlib.path import Path
    inside_real = Path(np.c_[x, y]).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    f = np.maximum(r_tube - np.hypot(X, Y), np.minimum(np.where(inside_real, 1.0, -1.0), r_cap - np.hypot(X, Y)))
    ax.contourf(X, Y, f, levels=[0, 10], colors=["#e5f5e0"]); ax.contour(X, Y, f, levels=[0], colors=["#2ca02c"], linewidths=2)
    ax.add_patch(Circle((0, 0), r_tube, fc="none", ec="#1f77b4", lw=1, ls=":"))
    ax.set_title("4. v3 union: max(tube, min(real, capsule))\nreal where wider, tube where it pinches", fontsize=8.5)
    save(fig, "v3_union_schematic.png")


def construction_versions():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    th = np.linspace(0, 2 * np.pi, 300)
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_axis_off()
    r = 2.3
    ax = axes[0]
    ax.add_patch(Circle((0, 0), r, fc="none", ec="#bbbbbb", lw=1.2, ls="--"))
    ax.add_patch(Circle((0, 0), r - 0.55, fc="#f7d5d5", ec=C["v1"], lw=2)); ax.plot(0, 0, "ko", ms=3)
    ax.text(0, 3.4, "v1: blurred binary tube, cut at 0.5", ha="center", fontsize=9.5); ax.text(0, -3.5, "circular, ~0.65 mm narrower than declared,\nthin vessels vanish", ha="center", fontsize=8, color="#555")
    ax = axes[1]
    ax.add_patch(Circle((0, 0), r, fc="#d6e6f5", ec=C["v2"], lw=2)); ax.plot(0, 0, "ko", ms=3)
    ax.text(0, 3.4, "v2: signed-distance tube, cut at 0", ha="center", fontsize=9.5); ax.text(0, -3.5, "circular, radius = MISR (within ~0.1 mm)", ha="center", fontsize=8, color="#555")
    ax = axes[2]
    bul = 1 + 0.3 * np.exp(-((th - 0.8) ** 2) / 0.15)
    x, y = 2.6 * np.cos(th) * bul, 2.0 * np.sin(th) * bul
    ax.fill(x, y, color="#e5f5e0"); ax.plot(x, y, color=C["v3"], lw=2); ax.add_patch(Circle((0, 0), r, fc="none", ec=C["v2"], lw=1, ls=":")); ax.plot(0, 0, "ko", ms=3)
    ax.text(0, 3.4, "v3: tube + the patient's real surface", ha="center", fontsize=9.5); ax.text(0, -3.5, "real cross-section where segmented;\nnever narrower than the floored tube", ha="center", fontsize=8, color="#555")
    save(fig, "construction_versions.png")


def host_vs_sets():
    labels = ["host test mesh\n(real surface, v1 ship)", "v1 training", "v2 training", "v3 training"]
    deficit = [-0.15, 0.65, 0.12, 0.10]; lo = [-0.94, 0.53, 0.09, 0.08]; hi = [0.86, 1.08, 0.15, 0.13]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    cols = [C["host"], C["v1"], C["v2"], C["v3"]]
    ax.bar(range(4), deficit, color=cols, alpha=0.6, yerr=[np.array(deficit) - np.array(lo), np.array(hi) - np.array(deficit)], capsize=4)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("declared MISR - meshed wall distance (mm)\nmedian, bars = p10..p90")
    ax.set_title("Calibre relative to the declared radius: training sets vs the host test mesh", fontsize=10)
    ax.set_ylim(-1.1, 1.3)
    ax.text(0.42, -0.6, "host: wider than MISR on average,\nreal +/- 0.9 mm spread", ha="left", fontsize=7.5, color="#555")
    ax.text(1.42, 0.9, "v1: 0.65 mm narrower\nthan declared", ha="left", fontsize=7.5, color=C["v1"])
    save(fig, "host_vs_sets.png")


def label_necks():
    d = json.load(open(os.path.join(ROOT, "topbrain_data/label_necks.json")))
    names = sorted(d, key=lambda k: -d[k]["necks"]); names = [n for n in names if d[n]["necks"] > 0]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(len(names)), [d[n]["necks"] for n in names], color=["#c0392b" if d[n]["reject"] else "#e07b39" for n in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("topcow_", "") for n in names], rotation=60, fontsize=7.5, ha="right")
    ax.set_ylabel("skeleton voxels with\ninscribed radius < 0.6 mm"); ax.set_title("Label necks: where the segmentation is one or two voxels thin (20 of 50 vessels; red = rejected fragment)", fontsize=9.5)
    save(fig, "label_necks.png")


if __name__ == "__main__":
    np.random.seed(0)
    pipeline_flow(); disc_profiles(); tube_erosion(); sofa_budget(); lumen_distributions(); shape_ratios()
    cross_section_schematic(); v3_union_schematic(); construction_versions(); host_vs_sets(); label_necks()
