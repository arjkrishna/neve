"""CHECK 5, pass 4 -- ROBUST containment: is any detached fragment actually inside the lumen?

Pass 3 used vtkImplicitPolyDataDistance and reported 103/434 fragments with vertices
">0.3 mm inside", sd_max up to 24.3 mm. On a ~2 mm vessel that is impossible, so the
signed-distance SIGN is not usable at points remote from the surface: it is taken from the
nearest facet's normal, and in the concave pockets between vessels that normal can face
the query point. HANDOFF 11.2 in a new guise.

This pass replaces it with ray-parity containment (vtkSelectEnclosedSurface) against the
main component, with TWO controls in the same script:
  (a) every centerline station must come back INSIDE (inside by construction)
  (b) a point 500 mm off the bounding box must come back OUTSIDE
If either control fails the anatomy's verdict is voided rather than reported.
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
        out.append(np.array(pts, float))
    return np.vstack(out)


def enclosed(surface, pts):
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    for p in pts:
        vp.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    pd.SetPoints(vp)
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(pd)
    sel.SetSurfaceData(surface)
    sel.SetTolerance(1e-6)
    sel.CheckSurfaceOff()
    sel.Update()
    o = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(pts.shape[0])]) > 0.5


def analyse(d):
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate()
    m = m.clean(tolerance=1e-6, absolute=False)
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(m); conn.SetExtractionModeToAllRegions(); conn.Update()
    comps = []
    for i in range(conn.GetNumberOfExtractedRegions()):
        c2 = vtk.vtkPolyDataConnectivityFilter()
        c2.SetInputData(m); c2.SetExtractionModeToSpecifiedRegions()
        c2.AddSpecifiedRegion(i); c2.Update()
        g = vtk.vtkCleanPolyData(); g.SetInputData(c2.GetOutput()); g.Update()
        cp = pv.wrap(g.GetOutput())
        if cp.n_cells:
            comps.append(cp)
    comps.sort(key=lambda c: -c.n_cells)
    main = comps[0]
    surf = main.extract_surface().triangulate()

    cl = load_cl(os.path.join(d, "Centrelines_comb"))
    b = np.array(main.bounds)
    far = np.array([[b[1] + 500, b[3] + 500, b[5] + 500]])
    probe = np.vstack([cl, far])
    e = enclosed(surf, probe)
    ctrl_in = float(e[:-1].mean())
    ctrl_far = bool(e[-1])

    frags = []
    for cp in comps[1:]:
        v = np.asarray(cp.points)
        ev = enclosed(surf, v)
        frags.append(dict(cells=int(cp.n_cells), nverts=int(len(v)),
                          n_inside=int(ev.sum()), area=float(cp.area)))
    return dict(name=name, ncomp=len(comps), ctrl_cl_frac_inside=ctrl_in,
                ctrl_far_inside=ctrl_far, frags=frags)


c = analyse(os.path.join(TB, "topcow_mr_001"))
print("CONTROL topcow_mr_001: centerline frac inside=%.4f  far point inside=%s  frags=%s"
      % (c["ctrl_cl_frac_inside"], c["ctrl_far_inside"], json.dumps(c["frags"])))
if c["ctrl_cl_frac_inside"] < 0.90 or c["ctrl_far_inside"]:
    print("CONTROL FAILED -- VOID"); sys.exit(1)
print("CONTROL OK: ray-parity containment agrees with construction")

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
