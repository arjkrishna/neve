"""vmtksurfaceremeshing with a per-point target edge length: fine where the
device goes, coarse elsewhere. Runs in vmtk_env on the host."""
import json, os, sys, glob
import numpy as np, vtk
from vtk.util import numpy_support as ns
from vmtk import vmtkscripts

ROOT = "d:/neve/.claude/worktrees/rl_improv_16_resume"   # host path; runs in vmtk_env
OUT = ROOT + "/saved/mesher_probe"

def route_of(anat_dir):
    f = [x for x in glob.glob(anat_dir + "/Centrelines_comb/*.json") if "RCCA" in x.upper()][0]
    d = json.load(open(f, encoding="utf-8"))
    pos = np.array([cp["position"] for cp in d["markups"][0]["controlPoints"]], float)
    return np.c_[pos[:, 1], -pos[:, 2], -pos[:, 0]]          # loader: (y, -z, -x)

def remesh(vtp, route, near_edge, far_edge, ramp_mm, tag):
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(vtp); r.Update(); s = r.GetOutput()
    pts = ns.vtk_to_numpy(s.GetPoints().GetData())
    d = np.empty(len(pts))                       # vmtk_env has no scipy: chunked brute force
    for i in range(0, len(pts), 20000):
        blk = pts[i:i + 20000]
        d[i:i + 20000] = np.sqrt(((blk[:, None, :] - route[None, :, :]) ** 2).sum(-1)).min(1)
    edge = near_edge + (far_edge - near_edge) * np.clip((d - 4.0) / ramp_mm, 0, 1)
    arr = ns.numpy_to_vtk(edge.astype(np.float64), deep=1); arr.SetName("TargetEdgeLength")
    s.GetPointData().AddArray(arr)
    rm = vmtkscripts.vmtkSurfaceRemeshing()
    rm.Surface = s
    rm.ElementSizeMode = "edgelengtharray"
    rm.TargetEdgeLengthArrayName = "TargetEdgeLength"
    rm.NumberOfIterations = 10
    rm.Execute()
    out = rm.Surface
    w = vtk.vtkXMLPolyDataWriter(); p = vtp.replace("_sdf_full.vtp", "_remesh_%s.vtp" % tag)
    w.SetFileName(p); w.SetInputData(out); w.Write()
    print("%-46s near %.1f far %.1f -> %6d tris  %s" % (os.path.basename(vtp), near_edge, far_edge, out.GetNumberOfCells(), os.path.basename(p)), flush=True)

for name, anat in (("topcow_mr_001", ROOT + "/topbrain_data/anatomies/topcow_mr_001"),
                   ("case_k_004_left__topcow_mr_010", ROOT + "/carotid_data/anatomies/case_k_004_left__topcow_mr_010")):
    vtp = OUT + "/" + name + "_sdf_full.vtp"
    route = route_of(anat)
    np.savetxt(OUT + "/" + name + "_route.txt", route)
    remesh(vtp, route, 1.4, 6.0, 30.0, "a")     # ~8 segments round a 1.8 mm vessel on the route
    remesh(vtp, route, 1.8, 7.0, 30.0, "b")     # leaner
    remesh(vtp, route, 2.5, 2.5, 30.0, "u")     # uniform, for reference
