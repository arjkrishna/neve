#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_shift.py (= /sp/v6/s3fix/diag_shift.py in the build containers)
"""s3_fix screen: does a sub-voxel shift of the bake lattice keep the route piece's thin scanned slivers attached?
For each shift (mm, applied to the lattice origin of the field AND every section capsule, so they stay aligned), build
the bake's union at the rung-1 state (bake_v5b: carotid sections pinned to v4's margin / stretches, VA sections whole at
VA margin 0.5) and report genus, components, and the dropped components that lie on lower_R / siphon_R's own scanned
surface near the RCCA (the lost-wall class found by diag_frag).
    python3 diag_shift.py <name> <out.json> <sx,sy,sz> [<sx,sy,sz> ...]   (neve-build-meshlab)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402
import sdf_mesher as SM  # noqa: E402

V = BV.V
B = BV.B
from scipy.spatial import cKDTree  # noqa: E402
from sdf_mesher import iso_surface  # noqa: E402
from sdf_union_v4 import Samples, genus  # noqa: E402

_GRID = SM._grid
SHIFT = np.zeros(3)


def grid_shifted(branches, spacing):
    lo = np.min([np.min(np.asarray(b.coordinates) - np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    hi = np.max([np.max(np.asarray(b.coordinates) + np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    lo = np.floor((lo - SHIFT) / spacing) * spacing - SM.PAD_VOX * spacing + SHIFT
    shape = np.ceil((hi - lo) / spacing).astype(int) + 2 * SM.PAD_VOX
    return lo, shape


SM._grid = grid_shifted


def main():
    name, out = sys.argv[1], sys.argv[2]
    shifts = [np.array([float(v) for v in a.split(",")]) for a in sys.argv[3:]]
    tree = os.environ.get("TREE", "/repo/carotid_data/anatomies_v6")
    root = os.path.join(tree, name)
    from eve_bench.dualdevicenav import load_branches
    import pyvista as pv
    BV.setup(name, root, "/repo/carotid_data/anatomies_v4", "/s3/pin")
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs_all, _ = B.sections_for(root, br)
    pin = json.load(open(os.path.join("/s3/pin", name, B.REPORT)))
    v4_margin = pin["margin_mm"]
    drop = set(pin.get("sections_tube_only") or [])
    car = {s.name for s in secs_all if s.branch in V.CAROTID}
    pin_sup = {k: [tuple(x) for x in v] for k, v in (pin.get("suppressed_stretches_mm") or {}).items() if k in car}
    secs = [s for s in secs_all if s.name not in drop]
    rc = B.branch(br, "RCCA")
    P, R = np.asarray(rc.coordinates, float), np.asarray(rc.radii, float)
    S = B.arclen(P)
    trR = cKDTree(P)
    surf = {s.name: s.surface for s in secs_all if s.name in ("lower_R", "siphon_R")}
    res = {"name": name, "v4_margin": v4_margin, "pinned": {k: [list(x) for x in v] for k, v in pin_sup.items()},
           "tube_only": sorted(drop), "shifts": []}
    if os.path.exists(out):
        res = json.load(open(out))
    done = {tuple(r["shift"]) for r in res["shifts"]}
    for sh in shifts:
        if tuple(np.round(sh, 4)) in done:
            continue
        t0 = time.time()
        SHIFT[:] = sh
        cache = V.UnionCache(br, samp, secs, B.SPACING_MM)
        seq = [(s.name, (0.5 if s.branch in BV.VA_BRANCHES else v4_margin), tuple(tuple(x) for x in pin_sup.get(s.name, [])))
               for s in secs]
        cache.build(seq)
        iso = iso_surface(cache.field)
        conn = iso.connectivity()
        rid = np.asarray(conn.cell_data["RegionId"])
        sizes = np.bincount(rid)
        keep = int(np.argmax(sizes))
        kept = conn.extract_cells(np.nonzero(rid == keep)[0]).extract_surface().triangulate()
        g = genus(kept)
        gmax = max(c["genus"] for c in g) if g else -1
        nme = B.edge_defects(kept)
        frs = []
        for k in [int(x) for x in np.argsort(-sizes) if x != keep][:40]:
            pc = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface()
            X = np.asarray(pc.points, float)
            d, i = trR.query(X)
            if np.median(d - R[i]) > 6.0:
                continue
            row = {"cells": int(sizes[k]), "volume_mm3": round(abs(float(pc.volume)), 2),
                   "s": [round(float(S[i].min()), 1), round(float(S[i].max()), 1)],
                   "tube_dist_med": round(float(np.median(d - R[i])), 2)}
            for nm, sf in surf.items():
                ds = np.abs(np.asarray(pv.PolyData(X).compute_implicit_distance(sf)["implicit_distance"]))
                row[nm] = round(float((ds <= 0.3).mean()), 2)
            frs.append(row)
        r = {"shift": [round(float(v), 4) for v in sh], "genus": gmax, "nonmanifold_edges": int(nme), "comps": int(len(sizes)),
             "route_fragments": frs,
             "route_fragment_volume_mm3": round(sum(f["volume_mm3"] for f in frs if max(f.get("lower_R", 0), f.get("siphon_R", 0)) >= 0.5), 2),
             "seconds": round(time.time() - t0, 1)}
        res["shifts"].append(r)
        json.dump(res, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)
        print(json.dumps({k: r[k] for k in ("shift", "genus", "comps", "route_fragment_volume_mm3", "seconds")}), flush=True)
        del cache, iso, conn


if __name__ == "__main__":
    main()
