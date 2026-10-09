#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\vmtk_fix.py (= /sp/v6/wall/vmtk_fix.py in the build containers)
"""v6_wall: diagnose / repair a degenerate VMTK line (vmr_vmtk.py returned a 0-mm line for ia172_015 L and ia172_069 L
on the plateau walls). For each <seg_dir> <vessel>:
  1. surface topology (components, points per component) and where the two seeds sit (distance to the surface, which
     component is nearest);
  2. repair A: the largest connected component only (the bake's sdf_union.load_surface keeps the largest component too),
     same seeds -> <v>_stageA_vmtk_lc.json;
  3. repair B (if A fails): seeds moved 2 mm inward along the label (the station path's point 2 mm from each end,
     snapped into the surface) -> <v>_stageA_vmtk_lc2.json.
A repair that yields a line of >= 90 % of the station length replaces <v>_stageA_vmtk.json (the failed one is kept as
<v>_stageA_vmtk_failed.json), and the surface used is written as <v>_stageA.vtp (the original kept as
<v>_stageA_multi.vtp) so the donor record's surface and line agree.
    python3 vmtk_fix.py <seg_dir> <rva|lva> [--apply]     (neve-build-vmtk, -u 0)"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import vtk
from vtk.util import numpy_support as ns

d, v = sys.argv[1], sys.argv[2]
apply = "--apply" in sys.argv
surf_p = os.path.join(d, "%s_stageA.vtp" % v)
seeds = json.load(open(os.path.join(d, "%s_seeds.json" % v)))
st = json.load(open(os.path.join(d, "stations_rays.json")))[v]
L_st = float(st["s"][-1])

rd = vtk.vtkXMLPolyDataReader()
rd.SetFileName(surf_p)
rd.Update()
pd = rd.GetOutput()
cf = vtk.vtkPolyDataConnectivityFilter()
cf.SetInputData(pd)
cf.SetExtractionModeToAllRegions()
cf.ColorRegionsOn()
cf.Update()
nreg = cf.GetNumberOfExtractedRegions()
sizes = [int(cf.GetRegionSizes().GetValue(i)) for i in range(nreg)]
lab = ns.vtk_to_numpy(cf.GetOutput().GetPointData().GetArray("RegionId"))
P = ns.vtk_to_numpy(cf.GetOutput().GetPoints().GetData())
info = {"components": nreg, "component_points": sorted(sizes, reverse=True)[:8], "station_length_mm": L_st}
for k in ("source", "target"):
    q = np.asarray(seeds[k], float)
    dd = np.linalg.norm(P - q, axis=1)
    j = int(np.argmin(dd))
    info["seed_%s" % k] = {"dist_to_surface_pts_mm": float(dd[j]), "nearest_component": int(lab[j]),
                           "nearest_component_points": sizes[int(lab[j])]}
print("diag", json.dumps(info), flush=True)

cl = vtk.vtkPolyDataConnectivityFilter()
cl.SetInputData(pd)
cl.SetExtractionModeToLargestRegion()
cl.Update()
cc = vtk.vtkCleanPolyData()
cc.SetInputConnection(cl.GetOutputPort())
cc.Update()
lc_p = os.path.join(d, "%s_stageA_lc.vtp" % v)
w = vtk.vtkXMLPolyDataWriter()
w.SetFileName(lc_p)
w.SetInputData(cc.GetOutput())
w.Write()


def run(surface, seeds_p, out):
    subprocess.call(["timeout", "900", "python3", "/sp/v5_va/vmr_vmtk.py", surface, seeds_p, out],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        r = json.load(open(out))
        return float(r["length_mm"]), int(r["n"])
    except Exception:  # noqa: BLE001
        return 0.0, 0


res = {"diag": info}
LA, nA = run(lc_p, os.path.join(d, "%s_seeds.json" % v), os.path.join(d, "%s_stageA_vmtk_lc.json" % v))
res["repair_A_largest_component"] = {"length_mm": LA, "n": nA}
best = ("A", LA, lc_p, os.path.join(d, "%s_stageA_vmtk_lc.json" % v)) if LA >= 0.9 * L_st else None
if best is None:
    # B: seeds 2 mm inward along the station path, snapped to the nearest LARGEST-component point then pushed 0.5 mm in
    Ps = np.asarray(st["P"], float)
    s = np.asarray(st["s"], float)
    a = Ps[int(np.searchsorted(s, 2.0))]
    b = Ps[int(np.searchsorted(s, s[-1] - 2.0)) - 1]
    sp2 = os.path.join(d, "%s_seeds_in2.json" % v)
    json.dump({"source": a.tolist(), "target": b.tolist(), "note": "vmtk_fix: station path points 2 mm from each end"},
              open(sp2, "w"))
    LB, nB = run(lc_p, sp2, os.path.join(d, "%s_stageA_vmtk_lc2.json" % v))
    res["repair_B_seeds_2mm_in"] = {"length_mm": LB, "n": nB}
    if LB >= 0.9 * (L_st - 4.0):
        best = ("B", LB, lc_p, os.path.join(d, "%s_stageA_vmtk_lc2.json" % v))
res["chosen"] = best[0] if best else None
if best and apply:
    vj = os.path.join(d, "%s_stageA_vmtk.json" % v)
    if os.path.exists(vj) and not os.path.exists(os.path.join(d, "%s_stageA_vmtk_failed.json" % v)):
        shutil.copy2(vj, os.path.join(d, "%s_stageA_vmtk_failed.json" % v))
    if not os.path.exists(os.path.join(d, "%s_stageA_multi.vtp" % v)):
        shutil.copy2(surf_p, os.path.join(d, "%s_stageA_multi.vtp" % v))
    shutil.copy2(best[2], surf_p)
    r = json.load(open(best[3]))
    r["note"] = (r.get("note") or "") + "; vmtk_fix repair %s (largest component%s)" % (
        best[0], ", seeds 2 mm inward" if best[0] == "B" else "")
    r["surface"] = surf_p
    json.dump(r, open(vj, "w"))
    res["applied"] = True
json.dump(res, open(os.path.join(d, "%s_vmtk_fix.json" % v), "w"), indent=1)
print("result", json.dumps(res), flush=True)
