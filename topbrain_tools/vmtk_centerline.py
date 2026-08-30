#!/usr/bin/env python3
"""STAGE B (run in VMTK_ENV): right-ICA surface + seeds -> centerline with radius.

Mirrors vmr_processing_tools/extract_centerlines_vmtk.py: vmtkCenterlines with
explicit SourcePoints/TargetPoints, then the "MaximumInscribedSphereRadius"
point array, so the radii here are the same quantity as in the anatomy the
project already ships. Stage A wrote the .vtp and the two seeds.

    conda run -n vmtk_env python vmtk_centerline.py <surface_dir> --out <out_dir>

Writes {stem}_ica.json: {"points": [[x,y,z],...], "radii": [...], "length_mm": ...}
in world millimetres, ordered proximal -> distal.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def centerline_of(vtp_path, seeds_path):
    from vmtk import vmtkscripts
    import vtk
    from vtk.util import numpy_support as ns

    with open(seeds_path) as f:
        seeds = json.load(f)

    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = vtp_path
    reader.Execute()
    surface = reader.Surface

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(map(float, seeds["source"]))
    cl.TargetPoints = list(map(float, seeds["target"]))
    cl.AppendEndPoints = 1
    cl.Execute()
    centerlines = cl.Centerlines

    pts = ns.vtk_to_numpy(centerlines.GetPoints().GetData())
    arr = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if arr is None:
        return None, "no MaximumInscribedSphereRadius array"
    radii = ns.vtk_to_numpy(arr)

    # ALWAYS walk the cell's point ids, even for a single cell: the polydata's
    # point ORDER is not the path order, and using it directly makes the
    # polyline zig-zag (it inflated arclength ~2x and tortuosity to ~3.7).
    n_cells = centerlines.GetNumberOfCells()
    best, best_len = None, -1.0
    for c in range(n_cells):
        ids = centerlines.GetCell(c).GetPointIds()
        idx = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(idx) < 2:
            continue
        q = pts[idx]
        ln = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        if ln > best_len:
            best, best_len = idx, ln
    if best is None:
        return None, "no traversable cell"
    pts, radii = pts[best], radii[best]

    if pts[0][2] > pts[-1][2]:          # LPS: +z superior, so order proximal -> distal
        pts, radii = pts[::-1], radii[::-1]

    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return {"points": pts.tolist(), "radii": radii.tolist(),
            "length_mm": length, "n": int(len(pts)),
            "diam_prox_mm": float(2 * radii[0]),
            "diam_dist_mm": float(2 * radii[-1])}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("surface_dir")
    ap.add_argument("--out", default="topbrain_data/centerlines")
    ap.add_argument("--suffix", default="rICA")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vtps = sorted(glob.glob(os.path.join(a.surface_dir, "*_%s.vtp" % a.suffix)))
    print("found %d surfaces" % len(vtps))
    ok = 0
    for v in vtps:
        stem = os.path.basename(v).replace("_%s.vtp" % a.suffix, "")
        seeds = v.replace("_%s.vtp" % a.suffix, "_%s_seeds.json" % a.suffix)
        if not os.path.exists(seeds):
            print("  SKIP %-40s no seeds" % stem[:40]); continue
        try:
            res, err = centerline_of(v, seeds)
        except Exception as e:                       # noqa: BLE001
            res, err = None, "%s: %s" % (type(e).__name__, str(e)[:90])
        if res is None:
            print("  SKIP %-40s %s" % (stem[:40], err))
        else:
            ok += 1
            with open(os.path.join(a.out, stem + "_ica.json"), "w") as f:
                json.dump(res, f)
            print("  OK   %-40s %6.1f mm  %4d pts  diam %.1f -> %.1f mm"
                  % (stem[:40], res["length_mm"], res["n"],
                     res["diam_prox_mm"], res["diam_dist_mm"]))
    print("centerlines: %d / %d" % (ok, len(vtps)))


if __name__ == "__main__":
    sys.exit(main())
