#!/usr/bin/env python3
"""Bake v2 collision meshes -- signed-distance mesher -- in place.

For every <anatomy>/Centrelines_comb under --anatomies, writes

    <anatomy>/collision_full.vtp               SDF surface, quadric-reduced to --full-tris
    <anatomy>/vessel_architecture_collision.obj  interim: quadric to --obj-tris (replaced
                                                 by the route-weighted VMTK remesh later)
    <anatomy>/mesh_v2.json                       what was measured on the way

The .obj is what the env loads, so a folder is usable as soon as this has
run; the .vtp is the input to remesh_vmtk_v2.py, which spends triangles where
the device goes and overwrites the .obj with a better one at the same budget.

Never touches a folder that already has mesh_v2.json unless --force.

    python3 topbrain_tools/bake_meshes_v2.py --anatomies <dir> [--shard i/n]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/eve_bench")
from sdf_mesher import (SPACING_MM, decimate_to, iso_surface, mesh_stats,
                        route_lumen, tube_field)

MESH_NAME = "vessel_architecture_collision.obj"
FULL_NAME = "collision_full.vtp"
REPORT = "mesh_v2.json"
CATH_R = 0.35          # mic_cath OD 0.7
CONTACT = 0.3          # LocalMinDistance contactDistance in sofabeamadapter


def bake_one(root, spacing, full_tris, obj_tris, verbose):
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    t0 = time.time()
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    field = tube_field(br, spacing=spacing, verbose=verbose)
    full = iso_surface(field)
    st_full = mesh_stats(full)
    full = decimate_to(full, full_tris)
    full.save(os.path.join(root, FULL_NAME))
    obj = decimate_to(full, obj_tris)
    pv.save_meshio(os.path.join(root, MESH_NAME), obj)

    rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
    rep = {"spacing_mm": spacing, "seconds": round(time.time() - t0, 1),
           "iso_surface": st_full, "full": mesh_stats(full), "obj": mesh_stats(obj)}
    for tag, m in (("full", full), ("obj", obj)):
        d, ins, body, s = route_lumen(m, route)
        ok = ins & body
        if ok.any():
            i = int(np.argmin(np.where(ok, d, np.inf)))
            rep[tag].update(lumen_min_mm=round(float(d[i]), 3), lumen_min_at_mm=round(float(s[i]), 1),
                            declared_there_mm=round(float(rad[i]), 3),
                            median_deficit_mm=round(float(np.median(rad[ok] - d[ok])), 3),
                            route_pts_outside=int((~ins & body).sum()),
                            navigable=bool(d[ok].min() - CONTACT >= CATH_R))
        else:
            rep[tag].update(route_pts_outside=int(body.sum()), navigable=False)
    with open(os.path.join(root, REPORT), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", required=True)
    ap.add_argument("--spacing", type=float, default=SPACING_MM)
    ap.add_argument("--full-tris", type=int, default=60000)
    ap.add_argument("--obj-tris", type=int, default=20000)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated anatomy names")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        folders = [f for f in folders if os.path.basename(os.path.dirname(f)) in want]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        folders = folders[i::n]
    if not folders:
        print("no anatomies under %s" % a.anatomies)
        return 1
    print("baking %d meshes at %.2f mm (full %d / obj %d tris)\n" % (len(folders), a.spacing, a.full_tris, a.obj_tris))
    print("%-44s %6s %5s %5s %9s %9s %7s %5s" % ("anatomy", "tris", "comps", "open", "lumen_min", "deficit", "navig", "sec"))
    bad = 0
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        if os.path.exists(os.path.join(root, REPORT)) and not a.force:
            print("%-44s kept" % name[:44]); continue
        try:
            rep = bake_one(root, a.spacing, a.full_tris, a.obj_tris, a.verbose)
        except Exception as e:                                       # noqa: BLE001
            bad += 1
            print("%-44s FAILED %s: %s" % (name[:44], type(e).__name__, str(e)[:60])); continue
        o = rep["obj"]
        print("%-44s %6d %5d %5d %9.2f %9.2f %7s %5.0f" % (
            name[:44], o["tris"], o["comps"], o["open_edges"], o.get("lumen_min_mm", -1),
            o.get("median_deficit_mm", -1), "yes" if o.get("navigable") else "NO", rep["seconds"]), flush=True)
    print("\ndone, %d failed" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
