#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_island.py (= /sp/v6/s3fix/diag_island.py in the build containers)
"""s3_fix: genus of each dropped route island in an anatomy's final bake state (tree given), with optional lattice shift.
    python3 diag_island.py <name> <tree> [sx,sy,sz]"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/sp/v6/bake")
import bake_v6_1p2_dev as BV
V, B = BV.V, BV.B
from sdf_mesher import iso_surface
from sdf_union_v4 import Samples, genus
from scipy.spatial import cKDTree
name, tree = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    BV.LATTICE_SHIFT[:] = [float(v) for v in sys.argv[3].split(",")]
root = os.path.join(tree, name)
rep6 = json.load(open(os.path.join(root, "mesh_v6.json")))
from eve_bench.dualdevicenav import load_branches
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
raw = iso_surface(cache.field)
conn = raw.connectivity()
rid = np.asarray(conn.cell_data["RegionId"]); sizes = np.bincount(rid); keep = int(np.argmax(sizes))
rc = B.branch(br, "RCCA"); P, R = np.asarray(rc.coordinates, float), np.asarray(rc.radii, float); S = B.arclen(P); tP = cKDTree(P)
for k in [int(x) for x in np.argsort(-sizes) if x != keep][:12]:
    pc = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface().clean().triangulate()
    X = np.asarray(pc.points); d, i = tP.query(X)
    if np.median(d - R[i]) > 6: continue
    print("cells", int(sizes[k]), "vol", round(abs(pc.volume), 2), "s", round(S[i].min(), 1), round(S[i].max(), 1), "genus", genus(pc), "open_edges", pc.n_open_edges)
