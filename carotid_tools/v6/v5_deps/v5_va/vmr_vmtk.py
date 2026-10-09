#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_vmtk.py (= /sp/v5_va/vmr_vmtk.py in the build containers)
"""VMTK centerlines (MaximumInscribedSphereRadius) for the VMR 0248 VA surfaces.
Same call as topbrain_tools/vmtk_centerline.py (vmtkCenterlines, pointlist seeds,
AppendEndPoints=1, longest cell, walked in cell order); open surfaces (the lofted
SimVascular wall faces) are capped first with vmtkSurfaceCapper (centerpoint).

    python3 /sp/v5_va/vmr_vmtk.py <surface.vtp> <seeds.json> <out.json> [--cap]

neve-build-vmtk (python 3.10, vmtk 1.5.0)."""
import json
import sys

import numpy as np


def main():
    surf_path, seeds_path, out_path = sys.argv[1:4]
    cap = "--cap" in sys.argv
    from vmtk import vmtkscripts
    from vtk.util import numpy_support as ns

    seeds = json.load(open(seeds_path))
    rd = vmtkscripts.vmtkSurfaceReader()
    rd.InputFileName = surf_path
    rd.Execute()
    surface = rd.Surface
    note = []
    if cap:
        cp = vmtkscripts.vmtkSurfaceCapper()
        cp.Surface = surface
        cp.Method = "centerpoint"
        cp.Interactive = 0
        cp.Execute()
        surface = cp.Surface
        note.append("capped (vmtkSurfaceCapper centerpoint)")
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(map(float, seeds["source"]))
    cl.TargetPoints = list(map(float, seeds["target"]))
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
    P, R = pts[best], rad[best]
    src = np.asarray(seeds["source"], float)
    if np.linalg.norm(P[-1] - src) < np.linalg.norm(P[0] - src):
        P, R = P[::-1], R[::-1]
    out = {"points": P.tolist(), "radii": R.tolist(), "length_mm": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
           "n": int(len(P)), "surface": surf_path, "seeds": seeds, "note": "; ".join(note) or "closed surface",
           "method": "vmtkCenterlines pointlist AppendEndPoints=1 (as topbrain_tools/vmtk_centerline.py)"}
    json.dump(out, open(out_path, "w"))
    print("%s: %d pts, %.1f mm, r median %.2f" % (out_path, len(P), out["length_mm"], float(np.median(R))))


if __name__ == "__main__":
    main()
