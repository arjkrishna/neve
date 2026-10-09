#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\vmtk_fix2.py (= /sp/v6/wall/vmtk_fix2.py in the build containers)
"""v6_wall: further VMTK repairs for a side vmtk_fix.py could not repair (ia172_069 L plateau). Tries, in order:
  C  normals made consistent and outward (vtkPolyDataNormals auto-orient) + clean, seeds 2 mm in;
  D  C + seeds 5 mm in;
  E  the same label's sub-voxel iso surface (<v>_iso.vtp), seeds 2 mm in;
  F  C + 5 windowed-sinc iterations (pass band 0.1) -- a mild extra smoothing, seeds 2 mm in.
The first that gives >= 90 % of the station length is applied as vmtk_fix.py does (surface + line replaced, originals
kept as *_multi.vtp / *_vmtk_failed.json).
    python3 vmtk_fix2.py <seg_dir> <rva|lva> [--apply]   (neve-build-vmtk, -u 0)"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import vtk

d, v = sys.argv[1], sys.argv[2]
apply = "--apply" in sys.argv
st = json.load(open(os.path.join(d, "stations_rays.json")))[v]
Ps, s = np.asarray(st["P"], float), np.asarray(st["s"], float)
L_st = float(s[-1])
src = os.path.join(d, "%s_stageA_multi.vtp" % v) if os.path.exists(os.path.join(d, "%s_stageA_multi.vtp" % v)) else \
    os.path.join(d, "%s_stageA.vtp" % v)


def read(p):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(p)
    r.Update()
    return r.GetOutput()


def write(pd, p):
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(p)
    w.SetInputData(pd)
    w.Write()


def fixnormals(pd):
    c = vtk.vtkCleanPolyData()
    c.SetInputData(pd)
    c.Update()
    n = vtk.vtkPolyDataNormals()
    n.SetInputConnection(c.GetOutputPort())
    n.ConsistencyOn()
    n.AutoOrientNormalsOn()
    n.SplittingOff()
    n.Update()
    return n.GetOutput()


def seeds_in(mm):
    a = Ps[int(np.searchsorted(s, mm))]
    b = Ps[int(np.searchsorted(s, s[-1] - mm)) - 1]
    p = os.path.join(d, "%s_seeds_in%g.json" % (v, mm))
    json.dump({"source": a.tolist(), "target": b.tolist(), "note": "vmtk_fix2: station path points %g mm from each end" % mm},
              open(p, "w"))
    return p


def run(surface, seeds_p, out):
    subprocess.call(["timeout", "900", "python3", "/sp/v5_va/vmr_vmtk.py", surface, seeds_p, out],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        r = json.load(open(out))
        return float(r["length_mm"]), int(r["n"])
    except Exception:  # noqa: BLE001
        return 0.0, 0


base = read(src)
C = fixnormals(base)
pc = os.path.join(d, "%s_stageA_fixC.vtp" % v)
write(C, pc)
sm = vtk.vtkWindowedSincPolyDataFilter()
sm.SetInputData(C)
sm.SetNumberOfIterations(5)
sm.SetPassBand(0.1)
sm.NonManifoldSmoothingOn()
sm.NormalizeCoordinatesOn()
sm.Update()
pf = os.path.join(d, "%s_stageA_fixF.vtp" % v)
write(fixnormals(sm.GetOutput()), pf)
trials = [("C", pc, seeds_in(2.0)), ("D", pc, seeds_in(5.0)), ("E", os.path.join(d, "%s_iso.vtp" % v), seeds_in(2.0)),
          ("F", pf, seeds_in(2.0))]
res = {"station_length_mm": L_st, "trials": {}}
best = None
for name, surf, sp in trials:
    out = os.path.join(d, "%s_stageA_vmtk_fix%s.json" % (v, name))
    Lr, n = run(surf, sp, out)
    res["trials"][name] = {"surface": surf, "seeds": sp, "length_mm": Lr, "n": n}
    print(name, Lr, n, flush=True)
    if Lr >= 0.9 * (L_st - 10.0):
        best = (name, surf, out)
        break
res["chosen"] = best[0] if best else None
if best and apply:
    vj = os.path.join(d, "%s_stageA_vmtk.json" % v)
    sp_ = os.path.join(d, "%s_stageA.vtp" % v)
    if os.path.exists(vj) and not os.path.exists(os.path.join(d, "%s_stageA_vmtk_failed.json" % v)):
        shutil.copy2(vj, os.path.join(d, "%s_stageA_vmtk_failed.json" % v))
    if not os.path.exists(os.path.join(d, "%s_stageA_multi.vtp" % v)):
        shutil.copy2(sp_, os.path.join(d, "%s_stageA_multi.vtp" % v))
    shutil.copy2(best[1], sp_)
    r = json.load(open(best[2]))
    r["note"] = (r.get("note") or "") + "; vmtk_fix2 repair %s" % best[0]
    r["surface"] = sp_
    json.dump(r, open(vj, "w"))
    res["applied"] = True
json.dump(res, open(os.path.join(d, "%s_vmtk_fix2.json" % v), "w"), indent=1)
print("result", json.dumps(res), flush=True)
