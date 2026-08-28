"""PASS 2: shared-trunk surface control.
(a) Is the cohort RCCA trunk curve (s<130) geometrically the same curve as the host's?
(b) Evaluate EVERY surface along ONE common trunk polyline (host trunk resampled 0.5mm)
    -> clearance difference is then attributable to the SURFACE only.
(c) Decompose clearance into lumen SIZE (r_eff from exact cross-section) and CENTEREDNESS
    (clearance / r_eff, and |centerline - section centroid|).
Read-only. Writes /tmp/out/nav_quality2.json
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
TRUNK_MAX = 130.0
EXCLUDE = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
RETAINED = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
            if os.path.isdir(d) and os.path.basename(d) not in EXCLUDE]


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def resample(C, step=0.5, smax=None):
    S = arclen(C)
    hi = S[-1] if smax is None else min(smax, S[-1])
    q = np.arange(0.0, hi, step)
    P = np.stack([np.interp(q, S, C[:, k]) for k in range(3)], 1)
    return P, q


def exact_signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def tangents(C):
    T = np.gradient(C, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n == 0] = 1
    return T / n


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
    cen = np.array([x.mean(), y.mean()])
    rad = np.hypot(x, y)
    return area, float(np.linalg.norm(cen)), float(rad.min()), float(rad.max())


def xsec(mesh, origin, normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(*[float(t) for t in origin])
    plane.SetNormal(*[float(t) for t in normal])
    cut = vtk.vtkCutter()
    cut.SetInputData(mesh)
    cut.SetCutFunction(plane)
    cut.Update()
    o = cut.GetOutput()
    if o.GetNumberOfPoints() == 0:
        return (np.nan,) * 4
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(o)
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(*[float(t) for t in origin])
    conn.Update()
    st = vtk.vtkStripper()
    st.SetInputData(conn.GetOutput())
    st.JoinContiguousSegmentsOn()
    st.Update()
    sp = st.GetOutput()
    if sp.GetNumberOfPoints() == 0:
        return (np.nan,) * 4
    P = vtk_to_numpy(sp.GetPoints().GetData())
    lines = sp.GetLines()
    lines.InitTraversal()
    ida = vtk.vtkIdList()
    best = None
    for _ in range(10000):
        if not lines.GetNextCell(ida):
            break
        idx = [ida.GetId(k) for k in range(ida.GetNumberOfIds())]
        if len(idx) < 4:
            continue
        if idx[0] == idx[-1]:
            idx = idx[:-1]
        m = loop_metrics(P[idx], np.asarray(origin, float), np.asarray(normal, float))
        if best is None or m[0] > best[0]:
            best = m
    if best is None or best[0] <= 0:
        return (np.nan,) * 4
    area, off, rmin, rmax = best
    return float(np.sqrt(area / np.pi)), off, rmin, rmax


def load(tag):
    if tag == "HOST_COLLISION":
        vt = DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    if tag == "HOST_VISUAL":
        vt = DualDeviceNav().vessel_tree
        return pv.read(vt.visu_mesh_path).triangulate().clean(), list(vt.branches)
    d0 = os.path.join(ROOT, tag)
    return (pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0, "Centrelines_comb")))


def rcca(brs):
    b = next(x for x in brs if "RCCA" in str(x.name).upper())
    return np.asarray(b.coordinates, float), np.asarray(b.radii, float)


TAGS = ["HOST_COLLISION", "HOST_VISUAL"] + RETAINED

hostmesh_c, hostbr = load("HOST_COLLISION")
Ch, Rh = rcca(hostbr)
Sh = arclen(Ch)
COMMON, QS = resample(Ch, 0.5, TRUNK_MAX)   # host trunk curve, uniform 0.5 mm
Th = tangents(COMMON)
print("common trunk polyline: %d stations, 0 -> %.2f mm" % (len(COMMON), QS[-1]), flush=True)

out = {"common_s": QS.tolist(), "per": {}}
for tag in TAGS:
    mesh, brs = load(tag)
    C, R = rcca(brs)
    S = arclen(C)
    # (a) curve deviation vs host over trunk
    Pown, _ = resample(C, 0.5, TRUNK_MAX)
    m = min(len(Pown), len(COMMON))
    dev = np.linalg.norm(Pown[:m] - COMMON[:m], axis=1)
    # (b) clearance on the COMMON curve
    sd = exact_signed(mesh, COMMON)
    ins = sd < 0
    if ins.mean() < 0.5:
        ins = ~ins
        sd = -sd
    cl = np.where(ins, np.abs(sd), 0.0)
    xs = np.array([xsec(mesh, COMMON[i], Th[i]) for i in range(len(COMMON))])
    reff, coff, rmin, rmax = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3]
    # stated radius of THIS anatomy interpolated onto the common arclength
    Rq = np.interp(QS, S, R)
    out["per"][tag] = {
        "dev_med": float(np.median(dev)), "dev_p95": float(np.percentile(dev, 95)),
        "dev_max": float(dev.max()), "own_trunk_stations": int((S < TRUNK_MAX).sum()),
        "own_spacing_med": float(np.median(np.diff(S))),
        "cl": cl.tolist(), "reff": reff.tolist(), "coff": coff.tolist(),
        "rmin_loop": rmin.tolist(), "rmax_loop": rmax.tolist(), "stated_r": Rq.tolist(),
    }
    fin = np.isfinite(reff)
    print("%-16s dev(med/p95/max)=%.4f/%.4f/%.4f | COMMON-curve cl med=%.3f p05=%.3f min=%.3f "
          "| reff med=%.3f | cl/reff med=%.3f | coff med=%.3f | statedr med=%.3f"
          % (tag, np.median(dev), np.percentile(dev, 95), dev.max(),
             np.median(cl), np.percentile(cl, 5), cl.min(),
             np.nanmedian(reff), np.nanmedian(cl[fin] / reff[fin]), np.nanmedian(coff),
             np.median(Rq)), flush=True)

os.makedirs("/tmp/out", exist_ok=True)
with open("/tmp/out/nav_quality2.json", "w") as f:
    json.dump(out, f)
print("WROTE /tmp/out/nav_quality2.json")
