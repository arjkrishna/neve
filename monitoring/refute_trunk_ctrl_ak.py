"""REFUTATION PASS: is the shared-trunk control an identity check or a mesher comparison?
T1  re-bake the HOST's own branches through the cohort mesher (generate_mesh: voxel 0.6/0.6/0.9,
    2x gaussian, marching cubes, decimate 0.99) and measure it on the SAME common trunk polyline.
T2  vertex correspondence: cohort trunk vertices vs host-collision / host-visual vertices.
T3  faceting decomposition: loop vertex count, centroid-referenced circumradius / inradius, r_eff.
Writes /tmp/out/refute_trunk.json. Read-only on the mounts; rebakes go to /tmp.
"""
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve.intervention.vesseltree.util.meshing import generate_mesh  # noqa: E402
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
COH = ["topcow_mr_001", "topcow_mr_003", "topcow_mr_016", "topcow_mr_024"]
os.makedirs("/tmp/out", exist_ok=True)


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def resample(C, step=0.5, smax=None):
    S = arclen(C)
    hi = S[-1] if smax is None else min(smax, S[-1])
    q = np.arange(0.0, hi, step)
    return np.stack([np.interp(q, S, C[:, k]) for k in range(3)], 1), q


def tangents(C):
    T = np.gradient(C, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n == 0] = 1
    return T / n


def exact_signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def loop_metrics(pts3, origin, normal):
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    rel = pts3 - origin
    x = rel @ u
    y = rel @ v
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))))
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A6 = 3 * np.sum(cr)
    if abs(A6) < 1e-12:
        cx, cy = float(x.mean()), float(y.mean())
    else:
        cx = float(np.sum((x + np.roll(x, -1)) * cr) / A6)
        cy = float(np.sum((y + np.roll(y, -1)) * cr) / A6)
    rad = np.hypot(x - cx, y - cy)
    px, py = x - cx, y - cy
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    dx, dy = qx - px, qy - py
    L2 = dx * dx + dy * dy
    L2[L2 == 0] = 1e-12
    t = np.clip(-(px * dx + py * dy) / L2, 0, 1)
    dseg = np.hypot(px + t * dx, py + t * dy)
    return dict(area=float(area), perim=perim, nv=int(len(x)), coff=float(np.hypot(cx, cy)),
                Rc=float(rad.max()), rin_c=float(dseg.min()), rmean_c=float(rad.mean()))


def xsec(mesh, origin, normal):
    pl = vtk.vtkPlane()
    pl.SetOrigin(*[float(t) for t in origin])
    pl.SetNormal(*[float(t) for t in normal])
    cu = vtk.vtkCutter()
    cu.SetInputData(mesh)
    cu.SetCutFunction(pl)
    cu.Update()
    o = cu.GetOutput()
    if o.GetNumberOfPoints() == 0:
        return None
    cnf = vtk.vtkPolyDataConnectivityFilter()
    cnf.SetInputData(o)
    cnf.SetExtractionModeToClosestPointRegion()
    cnf.SetClosestPoint(*[float(t) for t in origin])
    cnf.Update()
    st = vtk.vtkStripper()
    st.SetInputData(cnf.GetOutput())
    st.JoinContiguousSegmentsOn()
    st.Update()
    sp = st.GetOutput()
    if sp.GetNumberOfPoints() == 0:
        return None
    P = vtk_to_numpy(sp.GetPoints().GetData())
    ln = sp.GetLines()
    ln.InitTraversal()
    ida = vtk.vtkIdList()
    best = None
    while ln.GetNextCell(ida):
        idx = [ida.GetId(k) for k in range(ida.GetNumberOfIds())]
        if len(idx) < 4:
            continue
        if idx[0] == idx[-1]:
            idx = idx[:-1]
        m = loop_metrics(P[idx], np.asarray(origin, float), np.asarray(normal, float))
        if best is None or m["area"] > best["area"]:
            best = m
    if best is None or best["area"] <= 0:
        return None
    best["reff"] = float(np.sqrt(best["area"] / np.pi))
    best["iso"] = float(4 * np.pi * best["area"] / best["perim"] ** 2)
    return best


def rcca(brs):
    b = next(x for x in brs if "RCCA" in str(x.name).upper())
    return np.asarray(b.coordinates, float), np.asarray(b.radii, float)


vt = DualDeviceNav().vessel_tree
hostbr = list(vt.branches)
mesh_hc = pv.read(vt.mesh_path).triangulate().clean()
mesh_hv = pv.read(vt.visu_mesh_path).triangulate().clean()
Ch, Rh = rcca(hostbr)
COMMON, QS = resample(Ch, 0.5, 130.0)
TH = tangents(COMMON)
print("common trunk %d stations 0->%.1f" % (len(COMMON), QS[-1]), flush=True)
print("host branches: %d %s" % (len(hostbr), sorted(str(b.name) for b in hostbr)), flush=True)

cohbr = {c: load_branches(os.path.join(ROOT, c, "Centrelines_comb")) for c in COH}
cnames = sorted(str(b.name) for b in cohbr[COH[0]])
print("cohort branches: %d %s" % (len(cnames), cnames), flush=True)
keep = set(cnames)
host_sub = [b for b in hostbr if str(b.name) in keep]
print("host branches matching cohort names: %d" % len(host_sub), flush=True)

rebakes = {}
generate_mesh(hostbr, "/tmp/out/host_rebake_all.obj", 0.99)
rebakes["HOST_REBAKE_ALLBR"] = pv.read("/tmp/out/host_rebake_all.obj").triangulate().clean()
if len(host_sub) and len(host_sub) != len(hostbr):
    generate_mesh(host_sub, "/tmp/out/host_rebake_sub.obj", 0.99)
    rebakes["HOST_REBAKE_COHORTBR"] = pv.read("/tmp/out/host_rebake_sub.obj").triangulate().clean()

SURF = {"HOST_COLLISION": mesh_hc, "HOST_VISUAL": mesh_hv}
SURF.update(rebakes)
for c in COH:
    SURF[c] = pv.read(os.path.join(ROOT, c, "vessel_architecture_collision.obj")).triangulate().clean()

Rq = np.interp(QS, arclen(Ch), Rh)
res = {"QS": QS.tolist(), "stated_r": Rq.tolist(), "per": {}}
for tag, m in SURF.items():
    sd = exact_signed(m, COMMON)
    ins = sd < 0
    if ins.mean() < 0.5:
        ins = ~ins
        sd = -sd
    cl = np.where(ins, np.abs(sd), 0.0)
    rows = [xsec(m, COMMON[i], TH[i]) for i in range(len(COMMON))]

    def g(k, rows=rows):
        return np.array([np.nan if r is None else r[k] for r in rows], float)

    res["per"][tag] = dict(npts=int(m.n_points), ncells=int(m.n_cells), cl=cl.tolist(),
                           reff=g("reff").tolist(), nv=g("nv").tolist(), Rc=g("Rc").tolist(),
                           rin_c=g("rin_c").tolist(), coff=g("coff").tolist(),
                           iso=g("iso").tolist(), perim=g("perim").tolist())
    w = QS < 100
    print("%-22s pts=%6d cells=%6d | s<100 cl=%.3f reff=%.3f Rc=%.3f rin=%.3f nv=%.1f iso=%.3f "
          "coff=%.3f reff/stated=%.3f"
          % (tag, m.n_points, m.n_cells, np.nanmedian(cl[w]), np.nanmedian(g("reff")[w]),
             np.nanmedian(g("Rc")[w]), np.nanmedian(g("rin_c")[w]), np.nanmedian(g("nv")[w]),
             np.nanmedian(g("iso")[w]), np.nanmedian(g("coff")[w]),
             np.nanmedian(g("reff")[w]) / np.median(Rq[w])), flush=True)

tree_common = cKDTree(COMMON)


def trunkverts(m, rad=8.0):
    V = np.asarray(m.points, float)
    d, _ = tree_common.query(V)
    return V[d < rad]


ref = {"HOST_COLLISION": trunkverts(mesh_hc), "HOST_VISUAL": trunkverts(mesh_hv)}
res["vertex"] = {}
for tag, m in SURF.items():
    V = trunkverts(m)
    e = {}
    for rn, RV in ref.items():
        if len(RV) == 0 or len(V) == 0:
            continue
        d, _ = cKDTree(RV).query(V)
        e[rn] = dict(n=int(len(V)), nref=int(len(RV)), nn_med=float(np.median(d)),
                     nn_min=float(d.min()), frac_exact=float(np.mean(d < 1e-6)))
    F = m.faces.reshape(-1, 4)[:, 1:]
    P = np.asarray(m.points, float)
    dC, _ = tree_common.query(P[F].mean(1))
    sel = F[dC < 8.0]
    if len(sel):
        el = np.concatenate([np.linalg.norm(P[sel[:, i]] - P[sel[:, (i + 1) % 3]], axis=1)
                             for i in range(3)])
        e["edge_med"] = float(np.median(el))
        e["edge_p90"] = float(np.percentile(el, 90))
        e["ntri_trunk"] = int(len(sel))
    res["vertex"][tag] = e
    print("VERT %-22s ntrunk=%5d ntri=%5d edge_med=%.3f | vsHC nn_med=%.3f exact=%.3f | "
          "vsHV nn_med=%.3f exact=%.3f"
          % (tag, len(V), e.get("ntri_trunk", 0), e.get("edge_med", np.nan),
             e.get("HOST_COLLISION", {}).get("nn_med", np.nan),
             e.get("HOST_COLLISION", {}).get("frac_exact", np.nan),
             e.get("HOST_VISUAL", {}).get("nn_med", np.nan),
             e.get("HOST_VISUAL", {}).get("frac_exact", np.nan)), flush=True)

with open("/tmp/out/refute_trunk.json", "w") as f:
    json.dump(res, f)
print("WROTE /tmp/out/refute_trunk.json")
