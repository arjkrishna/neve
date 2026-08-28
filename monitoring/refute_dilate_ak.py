"""REFUTATION PASS: judge the cohort against FAITHFUL references, not the host collision mesh.
(1) Each surface vs ITS OWN declared radii (the VMTK MISR ground truth) -- no host involved.
    Ideal-tube probe: points at exactly stated_r(s) around C(s) in the normal plane,
    exact signed distance to the surface. 0 = perfect, +inside = dilated, -outside = deflated.
(2) Facet-resolution asymmetry: cut-loop segment count + mesh edge length near the RCCA.
(3) Chord-error decomposition of r_eff.
(4) Absolute declared calibre (DIAMETER) of the cohort siphons vs the host's own.
Read-only. JSON -> /tmp/out/refute_dilate.json
"""
import glob, json, os, sys
import numpy as np, pyvista as pv, vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches

ROOT = "/opt/eve_training/results_topbrain/anatomies"
EXCL = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
RET = [os.path.basename(d) for d in sorted(glob.glob(ROOT + "/*"))
       if os.path.isdir(d) and os.path.basename(d) not in EXCL]

def arclen(c):
    return np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))

def frames(C):
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.tile([0., 0., 1.], (len(C), 1))
    bad = np.abs((T * a).sum(1)) > 0.9
    a[bad] = [1., 0., 0.]
    U = np.cross(T, a); U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    return T, U, V

def load(tag):
    if tag.startswith("HOST"):
        vt = DualDeviceNav().vessel_tree
        p = vt.mesh_path if tag == "HOST_COLLISION" else vt.visu_mesh_path
        return pv.read(p).triangulate().clean(), list(vt.branches)
    d = os.path.join(ROOT, tag)
    return (pv.read(d + "/vessel_architecture_collision.obj").triangulate().clean(),
            load_branches(d + "/Centrelines_comb"))

def cutloop(mesh, o, n):
    pl = vtk.vtkPlane(); pl.SetOrigin(*[float(x) for x in o]); pl.SetNormal(*[float(x) for x in n])
    cu = vtk.vtkCutter(); cu.SetInputData(mesh); cu.SetCutFunction(pl); cu.Update()
    if cu.GetOutput().GetNumberOfPoints() == 0: return None
    cn = vtk.vtkPolyDataConnectivityFilter(); cn.SetInputData(cu.GetOutput())
    cn.SetExtractionModeToClosestPointRegion(); cn.SetClosestPoint(*[float(x) for x in o]); cn.Update()
    st = vtk.vtkStripper(); st.SetInputData(cn.GetOutput()); st.JoinContiguousSegmentsOn(); st.Update()
    sp = st.GetOutput()
    if sp.GetNumberOfPoints() == 0: return None
    P = vtk_to_numpy(sp.GetPoints().GetData()); L = sp.GetLines(); L.InitTraversal()
    ida = vtk.vtkIdList(); best = None; bA = 0.
    nn = np.asarray(n, float); nn = nn / np.linalg.norm(nn)
    a = np.array([1., 0., 0.]);  a = np.array([0., 1., 0.]) if abs(nn @ a) > 0.9 else a
    u = np.cross(nn, a); u /= np.linalg.norm(u); v = np.cross(nn, u)
    while L.GetNextCell(ida):
        idx = [ida.GetId(k) for k in range(ida.GetNumberOfIds())]
        if len(idx) < 4: continue
        if idx[0] == idx[-1]: idx = idx[:-1]
        Q = P[idx]; x = Q @ u; y = Q @ v
        A = .5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if A > bA: bA, best = A, Q
    if best is None: return None
    return bA, best

def analyze(tag, nang=48):
    mesh, brs = load(tag)
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float); R = np.asarray(rc.radii, float); S = arclen(C)
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(mesh)
    # sign convention: probe centerline
    sd_c = np.array([imp.EvaluateFunction(p) for p in C])
    sgn = -1.0 if np.median(sd_c) < 0 else 1.0   # make "inside" positive
    T, U, V = frames(C)
    th = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    tube_dev = np.full(len(C), np.nan)   # signed dist of ideal-tube points (+ => mesh dilated)
    tube_dev_p50 = np.full(len(C), np.nan)
    for i in range(len(C)):
        pts = C[i] + R[i] * (np.outer(cs, U[i]) + np.outer(sn, V[i]))
        d = sgn * np.array([imp.EvaluateFunction(p) for p in pts])
        tube_dev[i] = d.mean(); tube_dev_p50[i] = np.median(d)
    # cross sections
    reff = np.full(len(C), np.nan); nseg = np.zeros(len(C), int)
    rvert = np.full(len(C), np.nan); seglen = np.full(len(C), np.nan)
    for i in range(len(C)):
        r = cutloop(mesh, C[i], T[i])
        if r is None: continue
        A, Q = r
        reff[i] = np.sqrt(A / np.pi); nseg[i] = len(Q)
        ctr = Q.mean(0)
        rvert[i] = np.median(np.linalg.norm(Q - ctr, axis=1))
        seglen[i] = np.median(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1))
    # mesh edge length near RCCA
    F = mesh.faces.reshape(-1, 4)[:, 1:]; P = mesh.points
    ctr_f = P[F].mean(1)
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(C).query(ctr_f)
    near = dd < 6.0
    E = []
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        E.append(np.linalg.norm(P[F[near, a_]] - P[F[near, b_]], axis=1))
    E = np.concatenate(E)
    return dict(tag=tag, npts=int(mesh.n_points), ncell=int(mesh.n_cells),
                S=S.tolist(), R=R.tolist(), tube_dev=tube_dev.tolist(),
                tube_dev_p50=tube_dev_p50.tolist(), reff=reff.tolist(),
                nseg=nseg.tolist(), rvert=rvert.tolist(), seglen=seglen.tolist(),
                edge_near_med=float(np.median(E)), edge_near_p90=float(np.percentile(E, 90)),
                nface_near=int(near.sum()))

out = {}
for t in ["HOST_COLLISION", "HOST_VISUAL"] + RET:
    try:
        out[t] = analyze(t)
        o = out[t]; S = np.array(o["S"]); R = np.array(o["R"])
        td = np.array(o["tube_dev"]); rf = np.array(o["reff"]); rv = np.array(o["rvert"])
        ns = np.array(o["nseg"], float)
        m = (S > 20) & (S < S[-1] - 8)
        print("%-15s pts=%5d | edge_med=%.3f nseg_med=%4.0f | tube_dev med=%+.3f mm  "
              "reff/r=%.3f  rvert/r=%.3f  stated_r med=%.3f" %
              (t, o["npts"], o["edge_near_med"], np.nanmedian(ns[m]), np.nanmedian(td[m]),
               np.nanmedian(rf[m] / R[m]), np.nanmedian(rv[m] / R[m]), np.median(R[m])), flush=True)
    except Exception as e:
        print("FAIL", t, repr(e), flush=True)
os.makedirs("/tmp/out", exist_ok=True)
json.dump(out, open("/tmp/out/refute_dilate.json", "w"))
print("WROTE")
