#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\diag_tubes2.py (= /sp/v5_va/bake_full/diag_tubes2.py in the build containers)
"""v5_bake_full diagnosis (read-only): localise the handle of a tube-only union along one VA branch (10 mm bins,
bake_v5b.localise_bins on a pseudo section = the whole branch), and test the tube-only genus with that VA truncated
at several arclengths.    python3 diag_tubes2.py <name> <RVA|LVA>
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
from sdf_mesher import iso_surface, tube_field  # noqa: E402
from sdf_union_v4 import genus  # noqa: E402


def main():
    from eve_bench.dualdevicenav import load_branches
    name, va = sys.argv[1], sys.argv[2]
    root = os.path.join("/repo/carotid_data/anatomies_v5", name)
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    b = B.branch(br, va)
    P = np.asarray(b.coordinates, float)
    S = B.arclen(P)
    iso, _, _ = B.keep_largest(iso_surface(tube_field(br, spacing=B.SPACING_MM)))
    sec = SimpleNamespace(name=va, branch=va, s0=0.0, points=P)
    for bm in (10.0, 20.0):
        print("bins %.0f mm with a handle:" % bm, V.localise_bins(iso, br, [sec], bm), flush=True)
    for cut in (15, 20, 30, 40, 60, 80, 100, 120, 150):
        sub = []
        for x in br:
            if x is b:
                m = S <= cut
                sub.append(SimpleNamespace(name=x.name, coordinates=P[m], radii=np.asarray(x.radii)[m]))
            else:
                sub.append(x)
        g = genus(B.keep_largest(iso_surface(tube_field(sub, spacing=B.SPACING_MM)))[0])
        print("%s truncated at s=%d mm: genus %s" % (va, cut, [c["genus"] for c in g]), flush=True)
    print("radii along %s (s, r) every 5 mm to 60:" % va, [(round(float(S[i]), 1), round(float(b.radii[i]), 2))
                                                         for i in range(0, len(S)) if S[i] <= 60 and i % 5 == 0], flush=True)


if __name__ == "__main__":
    main()
