"""Render a labelled snapshot of the aortic-arch / bif2 branch topology
relevant to the RCCA/RVA/LCCA/LVA wedge debugging.

Highlights:
  - Trunk(2)  : femoral -> trunk-top
  - Branch(0) : two physical sub-arcs (lower / upper) joined at the
                LCCA-junction; only the upper part is on-path for non-LCCA
                targets
  - Branch(11): bridge from LCCA-junction to RCCA/RVA-junction (sharp
                ~100 deg turn at i=0->1)
  - Branch(18): trunk-top -> LVA-junction bridge
  - Daughters : LCCA / LVA / RCCA / RVA
  - Junction nodes annotated with their vessel-CS coordinate
  - Empirical wedge-cluster centroid (~(18, 37, 375)) marked

Run:
    python plot_bif2_topology.py [out_path.png]

Output PNG defaults to ./bif2_topology.png in the current dir.
"""
import os
import sys
import json
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CENTERLINE_DIR = "eve_bench/data/dualdevicenav/Centrelines_comb"


def _get_rot_matrix(rzx):
    rz = -rzx[0] * np.pi / 180
    rx = -rzx[1] * np.pi / 180
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])
    return Rz @ Rx


_ROT = _get_rot_matrix((20, 5))


def _load(tag):
    fp = os.path.join(CENTERLINE_DIR, f"Centerline curve {tag}.mrk.json")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = []
    for m in data["markups"]:
        if m["type"] == "Curve":
            for cp in m["controlPoints"]:
                x, y, z = cp["position"]
                pts.append((y, -z, -x))
    return np.array(pts, dtype=float)


def _load_all_others():
    """All centerlines except the highlighted ones (drawn faintly)."""
    highlighted = {"(2)", "(0)", "(11)", "(18)", "- LCCA", "- LVA",
                   "- RCCA", "- RVA"}
    out = []
    for fp in sorted(glob.glob(os.path.join(CENTERLINE_DIR, "*.json"))):
        base = os.path.basename(fp).replace(".mrk.json", "").replace(
            "Centerline curve ", ""
        )
        if base in highlighted:
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = []
        for m in data["markups"]:
            if m["type"] == "Curve":
                for cp in m["controlPoints"]:
                    x, y, z = cp["position"]
                    pts.append((y, -z, -x))
        out.append((base, np.array(pts, dtype=float)))
    return out


def _plot_branch(ax, pts, color, label, lw=2.4):
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color=color, linewidth=lw, label=label)


def _annotate_node(ax, p, text, color="#000000", offset=(8, 8, 8),
                   marker_size=90):
    ax.scatter(p[0], p[1], p[2], color=color, s=marker_size, marker="o",
               edgecolors="white", linewidths=1.4, zorder=10)
    ax.text(p[0] + offset[0], p[1] + offset[1], p[2] + offset[2], text,
            fontsize=9, color=color, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.85))


def main(out_path: str = "bif2_topology.png"):
    trunk = _load("(2)")
    b0 = _load("(0)")
    b11 = _load("(11)")
    b18 = _load("(18)")
    lcca = _load("- LCCA")
    lva = _load("- LVA")
    rcca = _load("- RCCA")
    rva = _load("- RVA")
    others = _load_all_others()

    # Branch (0) splits at indices 21,22 (duplicate at LCCA-junction).
    # Lower sub-arc: i=0..21  (off-path lower bif2 cavity bottom)
    # Upper sub-arc: i=22..30 (on-path bridge to trunk-top)
    b0_lower = b0[:22]
    b0_upper = b0[22:]

    # Empirical wedge centroid from forensic analysis (vessel-CS).
    wedge = np.array([18.0, 37.0, 375.0])

    # ---- Figure 1: full arch topology ----
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Faint background: other centerlines
    for name, pts in others:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="#cccccc", linewidth=0.8, alpha=0.6)

    _plot_branch(ax, trunk, "#1f77b4", "trunk(2)")
    _plot_branch(ax, b0_lower, "#9467bd",
                 "(0) lower subarc (OFF-path for non-LCCA targets)",
                 lw=2.6)
    _plot_branch(ax, b0_upper, "#2ca02c",
                 "(0) upper subarc (on-path: trunk-top->LCCA-jn)")
    _plot_branch(ax, b11, "#d62728",
                 "(11) bridge LCCA-jn -> RCCA/RVA-jn  *SHARP TURN at i=0->1*")
    _plot_branch(ax, b18, "#17becf", "(18) bridge trunk-top -> LVA-jn")
    _plot_branch(ax, lcca, "#8c564b", "LCCA daughter")
    _plot_branch(ax, lva, "#e377c2", "LVA daughter")
    _plot_branch(ax, rcca, "#ff7f0e", "RCCA daughter")
    _plot_branch(ax, rva, "#bcbd22", "RVA daughter")

    # Junctions — offsets chosen to keep labels readable in the default view
    junctions = [
        (np.array([46.66, 33.91, 391.95]),
         "Trunk-top jn (47,34,392)\n{trunk(2),(0),(18)}",
         "#000000", (15, 15, 12)),
        (np.array([23.21, 15.75, 384.70]),
         "LCCA-jn (23,16,385)\n~100 deg bend!\n{(0),(11),LCCA}",
         "#aa0000", (-50, -25, -30)),
        (np.array([-0.35, 24.14, 416.22]),
         "RCCA/RVA-jn (-0.4,24,416)\n{(11),RCCA,RVA}",
         "#aa6600", (-45, -15, 15)),
        (np.array([47.54, 34.47, 430.11]),
         "LVA-jn (48,34,430)\n{(18),(19),LVA}",
         "#0066aa", (15, 15, 15)),
    ]
    for p, txt, c, off in junctions:
        _annotate_node(ax, p, txt, c, offset=off)

    # Wedge centroid
    ax.scatter(wedge[0], wedge[1], wedge[2],
               color="#ffd400", s=240, marker="*",
               edgecolors="black", linewidths=1.2, zorder=20,
               label="empirical wedge centroid (~18,37,375)")
    ax.text(wedge[0] + 4, wedge[1] + 4, wedge[2] - 6,
            "wedge zone\n(~22mm above LCCA-jn,\n10mm below it in z)",
            fontsize=8, color="#aa8800", weight="bold")

    # Daughter-ostia labels (start of each daughter)
    for tag, arr, c in [("LCCA[0]", lcca, "#8c564b"),
                       ("RCCA[0]", rcca, "#ff7f0e"),
                       ("RVA[0]", rva, "#bcbd22"),
                       ("LVA[0]", lva, "#e377c2")]:
        ax.text(arr[0, 0] + 2, arr[0, 1] + 2, arr[0, 2] + 2, tag,
                fontsize=7, color=c)

    ax.set_xlabel("x (vessel-CS, mm)")
    ax.set_ylabel("y (vessel-CS, mm)")
    ax.set_zlabel("z (vessel-CS, mm)")
    ax.set_title(
        "Bif2 region topology (full arch view)\n"
        "Branch (0) is on-path ONLY in indices 22..30 (upper subarc).\n"
        "Wire wedges at z~375 sit on (0)'s OFF-path lower indices 0..20 -- this is the bug.",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    # Looking down on the arch from the +y side; daughter ostia in the
    # upper portion of the frame.
    ax.view_init(elev=22, azim=-72)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ---- Figure 2: bif2 close-up zoom ----
    out_zoom = out_path.replace(".png", "_zoom.png")
    if out_zoom == out_path:
        out_zoom = "bif2_topology_zoom.png"
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Only paint the upper arch (z >= 360)
    def _crop(pts, zmin=360):
        return pts[pts[:, 2] >= zmin]

    for name, pts in others:
        c = _crop(pts)
        if len(c) > 1:
            ax.plot(c[:, 0], c[:, 1], c[:, 2],
                    color="#cccccc", linewidth=0.8, alpha=0.6)

    trunk_c = _crop(trunk, zmin=360)
    if len(trunk_c) > 1:
        _plot_branch(ax, trunk_c, "#1f77b4", "trunk(2)")
    _plot_branch(ax, b0_lower, "#9467bd", "(0) lower subarc OFF-path")
    _plot_branch(ax, b0_upper, "#2ca02c", "(0) upper subarc on-path")
    _plot_branch(ax, b11, "#d62728", "(11) bridge -> RCCA/RVA")
    _plot_branch(ax, b18, "#17becf", "(18) bridge -> LVA")
    _plot_branch(ax, _crop(lcca), "#8c564b", "LCCA daughter")
    _plot_branch(ax, _crop(lva), "#e377c2", "LVA daughter")
    _plot_branch(ax, _crop(rcca), "#ff7f0e", "RCCA daughter")
    _plot_branch(ax, _crop(rva), "#bcbd22", "RVA daughter")

    zoom_junction_offsets = {
        "Trunk-top": (12, 8, 6),
        "LCCA-jn": (-45, -25, -22),
        "RCCA/RVA-jn": (-50, -8, 12),
        "LVA-jn": (10, 12, 8),
    }
    for p, txt, c, _ in junctions:
        # Pick zoom-specific offset by matching the prefix of the label
        for k, off in zoom_junction_offsets.items():
            if txt.startswith(k):
                _annotate_node(ax, p, txt, c, offset=off)
                break

    # Highlight the sharp turn at (11)[0]->(11)[1]
    ax.plot([b11[0, 0], b11[1, 0]],
            [b11[0, 1], b11[1, 1]],
            [b11[0, 2], b11[1, 2]],
            color="#ff0000", linewidth=4.5, alpha=0.9,
            label="(11) i=0->1 (Δz=+12.5 — the sharp upturn)")

    ax.scatter(wedge[0], wedge[1], wedge[2],
               color="#ffd400", s=300, marker="*",
               edgecolors="black", linewidths=1.5, zorder=20,
               label="empirical wedge centroid")
    ax.text(wedge[0] + 3, wedge[1] + 3, wedge[2] - 4,
            "wedge\n(18,37,375)", fontsize=9,
            color="#aa8800", weight="bold")

    # Vector from wedge to the LCCA-junction (showing what the wire missed)
    junc_lcca = np.array([23.21, 15.75, 384.70])
    ax.plot([wedge[0], junc_lcca[0]],
            [wedge[1], junc_lcca[1]],
            [wedge[2], junc_lcca[2]],
            color="#888888", linestyle="--", linewidth=1.2, alpha=0.8,
            label=f"wedge -> LCCA-jn (~{np.linalg.norm(wedge-junc_lcca):.1f} mm)")

    ax.set_xlabel("x (vessel-CS, mm)")
    ax.set_ylabel("y (vessel-CS, mm)")
    ax.set_zlabel("z (vessel-CS, mm)")
    ax.set_title(
        "Bif2 close-up (z >= 360 mm)\n"
        "RCCA/RVA path: trunk(2) -> (0) upper subarc -> LCCA-jn (sharp turn) -> (11) -> RCCA/RVA ostium",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax.view_init(elev=20, azim=-55)
    # Tighter z-bounds — focus on z=360..440 where bif2 action happens
    z_top = 445
    ax.set_zlim(360, z_top)
    fig.tight_layout()
    fig.savefig(out_zoom, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_zoom}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "bif2_topology.png"
    main(out)
