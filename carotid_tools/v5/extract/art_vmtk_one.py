#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\art_vmtk_one.py (= /sp/v5_va/art_vmtk_one.py in the build containers)
"""art_vmtk_one.py <surface.vtp> <seeds.json> <trace_points.json|-> <out.json>

VMTK centerline with MISR radii exactly as vmr_vmtk.py / topbrain_tools/vmtk_centerline.py (vmtkCenterlines,
pointlist seeds, AppendEndPoints=1, longest cell), with retries when VMTK returns a degenerate line
(< 50% of the seed distance; happens when the Voronoi diagram is split by a pinch):
  attempt 1: seeds as given;
  attempt 2: seeds moved 3 mm inward along the traced path;
  attempt 3: surface remeshed (vmtkSurfaceRemeshing, target edge 0.3 mm) + attempt-2 seeds.
neve-build-vmtk (python 3.10, vmtk 1.5.0)."""
import json
import sys

import numpy as np


def run(surface, src, tgt):
    from vmtk import vmtkscripts
    from vtk.util import numpy_support as ns
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(map(float, src))
    cl.TargetPoints = list(map(float, tgt))
    cl.AppendEndPoints = 1
    cl.Execute()
    c = cl.Centerlines
    pts = ns.vtk_to_numpy(c.GetPoints().GetData())
    rad = ns.vtk_to_numpy(c.GetPointData().GetArray("MaximumInscribedSphereRadius"))
    best, bl = None, -1
    for k in range(c.GetNumberOfCells()):
        ids = c.GetCell(k).GetPointIds()
        idx = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(idx) < 2:
            continue
        ln = float(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1).sum())
        if ln > bl:
            best, bl = idx, ln
    if best is None:
        return None, None, 0.0
    P, R = pts[best], rad[best]
    if np.linalg.norm(P[-1] - src) < np.linalg.norm(P[0] - src):
        P, R = P[::-1], R[::-1]
    return P, R, bl


def main():
    surf_path, seeds_path, trace_path, out_path = sys.argv[1:5]
    from vmtk import vmtkscripts
    seeds = json.load(open(seeds_path))
    src, tgt = np.asarray(seeds["source"], float), np.asarray(seeds["target"], float)
    rd = vmtkscripts.vmtkSurfaceReader(); rd.InputFileName = surf_path; rd.Execute()
    surface = rd.Surface
    dist = float(np.linalg.norm(tgt - src))
    tries = []
    P, R, L = run(surface, src, tgt)
    tries.append({"attempt": 1, "length_mm": L})
    note = "seeds as given"
    if L < 0.5 * dist and trace_path != "-":
        T = np.asarray(json.load(open(trace_path))["points"], float)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T, axis=0), axis=1))]
        src2 = T[int(np.searchsorted(s, 3.0))]
        tgt2 = T[int(np.searchsorted(s, s[-1] - 3.0))]
        P, R, L = run(surface, src2, tgt2)
        tries.append({"attempt": 2, "length_mm": L})
        note = "seeds moved 3 mm inward"
        if L < 0.5 * dist:
            rm = vmtkscripts.vmtkSurfaceRemeshing()
            rm.Surface = surface
            rm.ElementSizeMode = "edgelength"
            rm.TargetEdgeLength = 0.3
            rm.Execute()
            P, R, L = run(rm.Surface, src2, tgt2)
            tries.append({"attempt": 3, "length_mm": L})
            note = "remeshed (edge 0.3 mm) + seeds moved 3 mm inward"
    ok = L >= 0.5 * dist
    out = {"points": [] if P is None else P.tolist(), "radii": [] if R is None else R.tolist(), "length_mm": L,
           "n": 0 if P is None else int(len(P)), "surface": surf_path, "seeds": seeds, "note": note, "tries": tries,
           "ok": bool(ok), "method": "vmtkCenterlines pointlist AppendEndPoints=1 (as topbrain_tools/vmtk_centerline.py)"}
    json.dump(out, open(out_path, "w"))
    print("%s: %d pts, %.1f mm, r median %.2f, %s, ok=%s" % (out_path, out["n"], L,
          float(np.median(R)) if R is not None and len(R) else -1, note, ok))


if __name__ == "__main__":
    main()
