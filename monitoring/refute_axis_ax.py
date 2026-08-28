"""Is the cohort cross-section's SHORT axis oriented consistently in WORLD space?

If the tubes are flattened along a fixed world direction, the minor axis of the
exact cross-section will cluster; if the deficit were tessellation noise or
tangent error the minor axis would be uniform on the circle of directions
perpendicular to the vessel.

Reports, per anatomy: the mean resultant length of the DOUBLED angle in the
station frame (axis data, so double before averaging) is meaningless across
stations with different frames, so instead the minor-axis unit vector is taken
in WORLD coordinates and its orientation tensor sum(d d^T) is eigen-decomposed.
Largest eigenvalue near 1/3 = isotropic, near 1 = one fixed world direction.
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

vtk.vtkObject.GlobalWarningDisplayOff()
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except Exception:
    pass

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
ONLY = sys.argv[1:] if len(sys.argv) > 1 else None


def rcca(bs):
    return next(b for b in bs if "RCCA" in str(b.name).upper())


def basis(t):
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u = u / np.linalg.norm(u)
    return u, np.cross(t, u)


def ordered_loops(lines):
    segs = []
    k = 0
    while k < len(lines):
        m = lines[k]
        ids = lines[k + 1:k + 1 + m]
        for j in range(m - 1):
            segs.append((int(ids[j]), int(ids[j + 1])))
        k += 1 + m
    adj = {}
    for a, b in segs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, loops = set(), []
    for st in list(adj):
        if st in seen:
            continue
        lp = [st]
        seen.add(st)
        prev, cur = None, st
        while True:
            nx = [x for x in adj[cur] if x != prev and x not in seen]
            if not nx:
                break
            prev, cur = cur, nx[0]
            seen.add(cur)
            lp.append(cur)
        if len(lp) >= 3:
            loops.append(lp)
    return loops


def run(name, mesh, coords, radii):
    n = len(coords)
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    ctr = mesh.points[f].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    sub = mesh.extract_cells(np.where(d < 25.0)[0]).extract_surface().triangulate().clean()
    cut = vtk.vtkCutter()
    cut.SetInputData(sub)
    pl = vtk.vtkPlane()
    cut.SetCutFunction(pl)

    D = []          # world-space minor-axis unit vectors
    ratio = []      # b/a from the second-moment ellipse fit
    aa, bb = [], []
    for i in range(0, n, 2):
        c = coords[i]
        t = coords[min(i + 1, n - 1)] - coords[max(i - 1, 0)]
        t = t / np.linalg.norm(t)
        u, w = basis(t)
        pl.SetOrigin(*[float(x) for x in c])
        pl.SetNormal(*[float(x) for x in t])
        cut.Update()
        o = pv.wrap(cut.GetOutput())
        if o.n_points < 3 or not np.asarray(o.lines).size:
            continue
        P = np.asarray(o.points)
        best = None
        for lp in ordered_loops(np.asarray(o.lines)):
            Q = P[lp] - c
            P2 = np.stack([Q @ u, Q @ w], axis=1)
            th = np.arctan2(P2[:, 1], P2[:, 0])
            dd = np.diff(np.concatenate([th, th[:1]]))
            dd = (dd + np.pi) % (2 * np.pi) - np.pi
            if abs(dd.sum()) / (2 * np.pi) < 0.9:
                continue
            x, y = P2[:, 0], P2[:, 1]
            ar = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            if best is None or ar < best[1]:
                best = (P2, ar)
        if best is None:
            continue
        P2, ar = best
        # densely resample the closed polygon, arclength weighted second moments
        A, B = P2, np.roll(P2, -1, axis=0)
        L = np.linalg.norm(B - A, axis=1)
        tt = (np.arange(30) + 0.5) / 30
        S = (A[:, None, :] + tt[None, :, None] * (B - A)[:, None, :]).reshape(-1, 2)
        Wt = np.repeat(L / 30, 30)
        S = S - (S * Wt[:, None]).sum(0) / Wt.sum()
        M = (S[:, :, None] * S[:, None, :] * Wt[:, None, None]).sum(0) / Wt.sum()
        ev, evec = np.linalg.eigh(M)
        if ev[0] <= 0:
            continue
        b_ = np.sqrt(ev[0])
        a_ = np.sqrt(ev[1])
        ratio.append(b_ / a_)
        aa.append(a_ * 2.0)   # 2*sqrt(second moment) ~ semi-axis for a ring
        bb.append(b_ * 2.0)
        dm = evec[:, 0]
        D.append(dm[0] * u + dm[1] * w)

    if len(D) < 10:
        return None
    D = np.array(D)
    T = (D[:, :, None] * D[:, None, :]).mean(0)
    ev, evec = np.linalg.eigh(T)
    return dict(n=len(D), ratio=float(np.median(ratio)),
                a_over_r=float(np.median(np.array(aa))),
                b_over_r=float(np.median(np.array(bb))),
                lam=[float(x) for x in ev[::-1]],
                axis=[float(x) for x in evec[:, -1]])


out = {}
host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca(hvt.branches)
out["HOST"] = run("HOST", hm, np.asarray(hb.coordinates, float),
                  np.asarray(hb.radii, float))
ds = [d for d in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(d)]
if ONLY:
    ds = [d for d in ds if os.path.basename(d) in ONLY]
for d in ds:
    nm = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    out[nm] = run(nm, m, np.asarray(br.coordinates, float), np.asarray(br.radii, float))
    print("done " + nm, file=sys.stderr)

print("")
print("CROSS-SECTION SHAPE (second-moment ellipse of the exact perpendicular section)")
print("%-16s%5s%9s%10s%10s | %-34s %s" % (
    "anatomy", "n", "b/a", "lam1", "lam2", "principal world axis of the SHORT "
    "axis", "(lam1=1/3 isotropic, 1 = one fixed direction)"))
for k, v in out.items():
    if v is None:
        print("%-16s  --" % k)
        continue
    print("%-16s%5d%9.3f%10.3f%10.3f | [%6.3f %6.3f %6.3f]" % (
        k, v["n"], v["ratio"], v["lam"][0], v["lam"][1],
        v["axis"][0], v["axis"][1], v["axis"][2]))

co = [v for k, v in out.items() if k != "HOST" and v]
if co:
    print("")
    print("cohort median b/a = %.3f  (range %.3f - %.3f)" % (
        np.median([v["ratio"] for v in co]),
        min(v["ratio"] for v in co), max(v["ratio"] for v in co)))
    print("cohort median lam1 = %.3f  (isotropic would be 0.333)" % np.median(
        [v["lam"][0] for v in co]))
print("")
print("MACHINE_JSON")
print(json.dumps(out))
