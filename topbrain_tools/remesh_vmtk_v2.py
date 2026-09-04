#!/usr/bin/env python3
"""Spend the collision triangles where the device goes.

Reads <anatomy>/collision_full.vtp (the SDF surface bake_meshes_v2.py wrote),
remeshes it with vmtksurfaceremeshing driven by a per-point target edge
length, and overwrites <anatomy>/vessel_architecture_collision.obj.

Edge length: 0.8 x the local route radius (so ~8 segments round the vessel,
inradius within ~0.1 mm) within NEAR_MM of the RCCA route, ramping to FAR_EDGE
by FAR_MM. Everything the device never touches is coarse; the arch is ~12 mm
radius and 6 mm edges are plenty. Measured on topcow_mr_001: 0.11 mm median
deficit at 20 k triangles, one manifold component, and SOFA steps at the same
speed as with 3.7 k.

Runs on the HOST in vmtk_env (no VMTK wheel exists for the container's
Python; no scipy in vmtk_env, so the distance query is plain numpy):

    conda run -n vmtk_env python topbrain_tools/remesh_vmtk_v2.py --anatomies <dir> [--workers 6]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

NEAR_MM = 6.0
FAR_MM = 30.0
FAR_EDGE = 6.0
EDGE_FACTOR = 0.8
EDGE_MIN, EDGE_MAX = 1.0, 2.4
FULL_NAME = "collision_full.vtp"
MESH_NAME = "vessel_architecture_collision.obj"
REPORT = "mesh_v2.json"


def route_of(anat):
    """RCCA route and radii in BRANCH coordinates (the loader's (y, -z, -x))."""
    f = [x for x in glob.glob(os.path.join(anat, "Centrelines_comb", "*.json")) if "RCCA" in x.upper()][0]
    d = json.load(open(f, encoding="utf-8"))
    m = d["markups"][0]
    pos = np.array([cp["position"] for cp in m["controlPoints"]], float)
    rad = None
    for meas in m.get("measurements", []):
        if meas.get("name") == "Radius":
            rad = np.array(meas["controlPointValues"], float)
    return np.c_[pos[:, 1], -pos[:, 2], -pos[:, 0]], rad


def nearest(pts, route, chunk=20000):
    d = np.empty(len(pts)); j = np.empty(len(pts), dtype=int)
    for i in range(0, len(pts), chunk):
        blk = pts[i:i + chunk]
        dd = np.sqrt(((blk[:, None, :] - route[None, :, :]) ** 2).sum(-1))
        j[i:i + chunk] = dd.argmin(1)
        d[i:i + chunk] = dd.min(1)
    return d, j


def remesh_one(anat):
    import vtk
    from vtk.util import numpy_support as ns
    from vmtk import vmtkscripts
    t0 = time.time()
    name = os.path.basename(anat)
    vtp = os.path.join(anat, FULL_NAME)
    if not os.path.exists(vtp):
        return name, "no %s" % FULL_NAME
    route, rad = route_of(anat)
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(vtp); r.Update(); s = r.GetOutput()
    pts = ns.vtk_to_numpy(s.GetPoints().GetData())
    d, j = nearest(pts, route)
    near_edge = np.clip(EDGE_FACTOR * (rad[j] if rad is not None else 2.0), EDGE_MIN, EDGE_MAX)
    w = np.clip((d - NEAR_MM) / (FAR_MM - NEAR_MM), 0.0, 1.0)
    edge = near_edge + (FAR_EDGE - near_edge) * w
    arr = ns.numpy_to_vtk(edge.astype(np.float64), deep=1); arr.SetName("TargetEdgeLength")
    s.GetPointData().AddArray(arr)
    rm = vmtkscripts.vmtkSurfaceRemeshing()
    rm.Surface = s
    rm.ElementSizeMode = "edgelengtharray"
    rm.TargetEdgeLengthArrayName = "TargetEdgeLength"
    rm.NumberOfIterations = 10
    # vmtk prints an "Iteration i/10" line per pass; silence it
    devnull = open(os.devnull, "w"); old = sys.stdout; sys.stdout = devnull
    try:
        rm.Execute()
    finally:
        sys.stdout = old; devnull.close()
    out = rm.Surface
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(out); tri.Update()
    w_ = vtk.vtkOBJWriter(); w_.SetFileName(os.path.join(anat, MESH_NAME)); w_.SetInputData(tri.GetOutput()); w_.Write()
    n = int(tri.GetOutput().GetNumberOfCells())
    rep_p = os.path.join(anat, REPORT)
    rep = json.load(open(rep_p, encoding="utf-8")) if os.path.exists(rep_p) else {}
    rep["remesh"] = {"tris": n, "near_mm": NEAR_MM, "far_mm": FAR_MM, "far_edge": FAR_EDGE,
                     "edge_factor": EDGE_FACTOR, "edge_range": [EDGE_MIN, EDGE_MAX],
                     "seconds": round(time.time() - t0, 1)}
    with open(rep_p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return name, "%d tris  %.0f s" % (n, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="redo folders that already carry a remesh record")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    anats = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.anatomies, "*", FULL_NAME)))
    if a.only:
        want = set(a.only.split(","))
        anats = [x for x in anats if os.path.basename(x) in want]
    if not a.force:
        def done(x):
            p = os.path.join(x, REPORT)
            return os.path.exists(p) and "remesh" in json.load(open(p, encoding="utf-8"))
        anats = [x for x in anats if not done(x)]
    print("remeshing %d anatomies with %d workers" % (len(anats), a.workers), flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for name, msg in pool.imap_unordered(remesh_one, anats):
            print("  %-46s %s" % (name[:46], msg), flush=True)
    print("done in %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
