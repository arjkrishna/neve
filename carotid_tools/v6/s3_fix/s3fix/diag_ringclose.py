#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_ringclose.py (= /sp/v6/s3fix/diag_ringclose.py in the build containers)
"""s3_fix diagnosis: can a genus>0 route island (a perforated scan sliver) be made genus 0 by a binary closing of ITS OWN
voxels (fills holes narrower than ~2 k voxels), so that one bridge can then attach it without a handle?
Rebuilds the anatomy's final union state (its mesh_v6.json), finds the route islands, and for each genus>0 island tries
closings with k = 1, 2, 3 iterations: genus of the closed island alone, voxels added, then (with the closing kept) the
whole union with ONE bridge to that island: genus.
    python3 diag_ringclose.py <name> <tree>      (neve-build-meshlab)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6_1p2_dev as BV  # noqa: E402
V, B = BV.V, BV.B
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sdf_mesher import iso_surface  # noqa: E402
from sdf_union_v4 import Samples, genus  # noqa: E402

name, tree = sys.argv[1], sys.argv[2]
root = os.path.join(tree, name)
rep6 = json.load(open(os.path.join(root, "mesh_v6.json")))
from eve_bench.dualdevicenav import load_branches  # noqa: E402
import pyvista as pv  # noqa: E402
BV.setup(name, root, "/repo/carotid_data/anatomies_v4", "/s3/pin/diag")
br = load_branches(os.path.join(root, "Centrelines_comb"))
samp = Samples(br)
secs_all, _ = B.sections_for(root, br)
pin = json.load(open(os.path.join("/s3/pin/diag", name, B.REPORT)))
v4m = pin["margin_mm"]
tube_only = set(rep6.get("sections_tube_only") or [])
secs = [s for s in secs_all if s.name not in tube_only]
vam = rep6["ladder_v5b"]["va_margin_mm"]
sup = {k: [tuple(x) for x in v] for k, v in (rep6.get("suppressed_stretches_mm") or {}).items()}
cache = V.UnionCache(br, samp, secs, B.SPACING_MM)
cache.build([(s.name, (vam if s.branch in BV.VA_BRANCHES else v4m), tuple(tuple(x) for x in sup.get(s.name, []))) for s in secs])
f = cache.field
sp = f.spacing


def gmax(m):
    g = genus(m)
    return max(c["genus"] for c in g if c["cells"] > 0) if g else -1


raw = iso_surface(f)
conn = raw.connectivity()
rid = np.asarray(conn.cell_data["RegionId"])
sizes = np.bincount(rid)
keep = int(np.argmax(sizes))
Vm = np.asarray(conn.extract_cells(np.nonzero(rid == keep)[0]).extract_surface().points, float)
tm = cKDTree(Vm)
rc = B.branch(br, "RCCA")
P, R = np.asarray(rc.coordinates, float), np.asarray(rc.radii, float)
S = B.arclen(P)
tP = cKDTree(P)
for k in [int(x) for x in np.argsort(-sizes) if x != keep][:6]:
    pc = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface().clean().triangulate()
    X = np.asarray(pc.points, float)
    d, i = tP.query(X)
    if np.median(d - R[i]) > 6 or sizes[k] < 30:
        continue
    g0 = gmax(pc)
    print("island cells %d vol %.2f s %.1f-%.1f genus %s" % (sizes[k], abs(pc.volume), S[i].min(), S[i].max(), g0), flush=True)
    if g0 <= 0:
        continue
    # the island's voxels: positive voxels in a box around it, labelled; the label nearest the island's points
    lo = np.maximum(np.floor((X.min(0) - 2 - f.offset) / sp).astype(int), 0)
    hi = np.minimum(np.ceil((X.max(0) + 2 - f.offset) / sp).astype(int) + 1, np.array(f.values.shape))
    sl = tuple(slice(lo[j], hi[j]) for j in range(3))
    sub = f.values[sl]
    pos = sub > 0
    lab, nl = ndimage.label(pos)
    G = np.argwhere(pos)
    Xg = G * sp + f.offset + lo * sp
    dd, jj = cKDTree(Xg).query(X)
    labs = lab[tuple(G[jj].T)]
    il = np.bincount(labs).argmax()
    isl = lab == il
    print("   island voxels %d (label %d of %d in box)" % (isl.sum(), il, nl), flush=True)
    for it in (1, 2, 3):
        cl = ndimage.binary_closing(isl, structure=ndimage.generate_binary_structure(3, 1), iterations=it)
        add = cl & ~pos
        old = sub.copy()
        sub2 = sub.copy()
        sub2[add] = 0.5 * sp
        f.values[sl] = sub2
        m2 = iso_surface(f)
        c2 = m2.connectivity()
        r2 = np.asarray(c2.cell_data["RegionId"])
        # the component holding the island now
        cen = X.mean(0)
        pts2 = np.asarray(c2.points)
        # component of the nearest point to the island's first vertex
        j0 = int(cKDTree(pts2).query(X[0])[1])
        pid = np.asarray(c2.point_data["RegionId"])[j0]
        ic = c2.extract_cells(np.nonzero(r2 == pid)[0]).extract_surface().clean().triangulate()
        print("   closing it=%d: voxels added %d, closed island cells %d genus %s" % (it, add.sum(), ic.n_cells, gmax(ic)), flush=True)
        f.values[sl] = old
