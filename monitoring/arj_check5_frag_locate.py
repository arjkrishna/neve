"""CHECK 5, pass 3 -- are any detached fragments INSIDE the navigable lumen, and what is
the recurring ~70-cell stub?

Pass 2 showed every anatomy's 17 branches attribute to one component, and fragment
vertices sit a median 7 mm OUTSIDE the main wall. But sd on a thin sliver can flip sign
per-vertex (max frac_inside 0.42), so this pass records the MOST-INSIDE fragment vertex
per fragment and compares it to the SOFA contactDistance of 0.30 mm. It also locates each
fragment against the centerline tree (nearest branch + distance) to identify the stub.

Sign control (HANDOFF 11.2) on topbrain topcow_mr_001 first: positive == inside.
"""
import os, sys, json, glob
import numpy as np
import pyvista as pv
import vtk

CAR = "/opt/eve_training/carotid/anatomies"
TB = "/opt/eve_training/results_topbrain/anatomies"


def load_cl(folder):
    out = []
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("Centerline curve ") and fn.endswith(".json")):
            continue
        j = json.load(open(os.path.join(folder, fn)))
        pts = []
        for m in j["markups"]:
            if m["type"] != "Curve":
                continue
            for cp in m["controlPoints"]:
                x, y, z = [float(v) for v in cp["position"]]
                pts.append((y, -z, -x))
        nm = os.path.splitext(os.path.splitext(fn)[0])[0].replace("Centerline curve ", "").strip()
        out.append((nm, np.array(pts, float)))
    return out


def sdist(mesh, pts):
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(mesh)
    return np.array([f.EvaluateFunction(p) for p in pts])


def analyse(d):
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate()
    m = m.clean(tolerance=1e-6, absolute=False)
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(m); conn.SetExtractionModeToAllRegions(); conn.Update()
    n = conn.GetNumberOfExtractedRegions()
    comps = []
    for i in range(n):
        c2 = vtk.vtkPolyDataConnectivityFilter()
        c2.SetInputData(m); c2.SetExtractionModeToSpecifiedRegions()
        c2.AddSpecifiedRegion(i); c2.Update()
        g = vtk.vtkCleanPolyData(); g.SetInputData(c2.GetOutput()); g.Update()
        cp = pv.wrap(g.GetOutput())
        if cp.n_cells:
            comps.append(cp)
    comps.sort(key=lambda c: -c.n_cells)
    main = comps[0]
    cls = load_cl(os.path.join(d, "Centrelines_comb"))
    frags = []
    for cp in comps[1:]:
        v = np.asarray(cp.points)
        s = sdist(main, v)
        ctr = np.asarray(cp.center)
        best = (1e9, "?")
        for nm, p in cls:
            dd = float(np.sqrt(((p - ctr) ** 2).sum(1)).min())
            if dd < best[0]:
                best = (dd, nm)
        frags.append(dict(cells=int(cp.n_cells), area=float(cp.area),
                          sd_max=float(s.max()), sd_med=float(np.median(s)),
                          n_in_03=int((s > 0.30).sum()), nverts=int(len(v)),
                          near_br=best[1], near_d=round(best[0], 2),
                          ctr=[round(float(x), 1) for x in ctr]))
    return dict(name=name, ncomp=len(comps), frags=frags)


c = analyse(os.path.join(TB, "topcow_mr_001"))
print("CONTROL topcow_mr_001 ncomp=%d frags=%s" % (c["ncomp"], json.dumps(c["frags"])))
for tag, root in (("CAR", CAR), ("TB", TB)):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        try:
            r = analyse(d); r["set"] = tag
            print("ROW " + json.dumps(r))
        except Exception as ex:
            print("ERR %s %s" % (os.path.basename(d), repr(ex)[:160]))
        sys.stdout.flush()
print("DONE")
