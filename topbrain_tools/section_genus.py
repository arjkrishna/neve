#!/usr/bin/env python3
"""Where are LOCAL handles? Genus of the mesh piece around each section.

find_bridges.py finds handles that join two vessels or two distant stretches of
one vessel. A handle can also be local -- a tunnel through the wall of a single
real surface, or a loop where a real lumen and its own tube cross twice. This
clips the mesh to the neighbourhood of each real-surface section, and of each
10 mm stretch of it, and reports the genus of each piece (boundary-aware:
chi = 2 - 2g - b per component). Also reports the genus of each SOURCE surface
as transformed, which says whether the handle was already in the segmentation.

    python3 topbrain_tools/section_genus.py <anatomy_dir> --margin 1.0
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/eve_bench")


def piece_genus(mesh, keep_cells):
    from sdf_union_v4 import genus
    if not keep_cells.any():
        return []
    sub = mesh.extract_cells(np.nonzero(keep_cells)[0]).extract_surface().clean()
    return [(round(c["genus"], 1), c["boundary_loops"], c["cells"]) for c in genus(sub) if c["cells"] > 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomy")
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()
    from eve_bench.dualdevicenav import load_branches
    from bake_meshes_v4 import sections_for, keep_largest
    from sdf_mesher import tube_field, iso_surface, SPACING_MM
    from sdf_union_v4 import Samples, add_real_sections_owned, genus
    br = load_branches(os.path.join(a.anatomy, "Centrelines_comb"))
    samp = Samples(br)
    secs, _ = sections_for(a.anatomy, br)
    field = tube_field(br, spacing=SPACING_MM)
    add_real_sections_owned(field, secs, samp, SPACING_MM, a.margin)
    mesh, _, _ = keep_largest(iso_surface(field))
    mesh = mesh.clean().triangulate()
    print("whole mesh genus:", [c["genus"] for c in genus(mesh)])
    cen = np.asarray(mesh.cell_centers().points)
    for sec in secs:
        src = [(round(c["genus"], 1), c["boundary_loops"]) for c in genus(sec.surface)]
        p, r = sec.points, sec.radii
        reach = 1.8 * r + 2.0
        d, i = cKDTree(p).query(cen)
        near = d < reach[i]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        whole = piece_genus(mesh, near)
        line = "%-9s source-surface genus %s | mesh around section %s" % (sec.name, src, whole)
        bad = []
        for lo in np.arange(0, s[-1], 10.0):
            m = near & (s[i] >= lo) & (s[i] < lo + 10.0)
            gg = piece_genus(mesh, m)
            if any(g[0] > 0 for g in gg):
                bad.append("%.0f-%.0f mm: %s" % (sec.s0 + lo, sec.s0 + lo + 10, gg))
        print(line + ("\n      handles in: " + " | ".join(bad) if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
