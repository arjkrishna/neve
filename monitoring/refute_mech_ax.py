"""Where does the 12% gap between the vertex ring (0.98 r) and the surface
between the vertices (0.875 r) actually come from?  A 12-sided cross-section can
only lose cos(pi/12) = 3.4% to CIRCUMFERENTIAL faceting.

Hypothesis: the loss is AXIAL.  The triangles that the perpendicular plane cuts
span +/-8-10 mm along the vessel.  A flat facet chording across a TORTUOUS axis
cuts inside the tube by roughly L^2/(8 Rc), which on a 2.4 mm lumen is large.

Test: per station, regress the section deficit on (a) the local radius of
curvature of the centerline and (b) the axial span of the cut triangles.
If it is axial chording, the deficit tracks 1/Rc and the facet span; if the
declared radii were simply inflated, it would track neither.
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


def curvature(coords, s):
    """local radius of curvature by circumcircle over a +/-4 mm window"""
    n = len(coords)
    Rc = np.full(n, np.inf)
    for i in range(n):
        a = np.argmin(np.abs(s - (s[i] - 4.0)))
        b = np.argmin(np.abs(s - (s[i] + 4.0)))
        if b - a < 2:
            continue
        A, B, C = coords[a], coords[i], coords[b]
        ab, cb = A - B, C - B
        cr = np.linalg.norm(np.cross(ab, cb))
        if cr < 1e-9:
            continue
        Rc[i] = (np.linalg.norm(ab) * np.linalg.norm(cb)
                 * np.linalg.norm(A - C)) / (2 * cr)
    return Rc


def run(name, mesh, coords, radii):
    n = len(coords)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    Rc = curvature(coords, s)
    F = mesh.faces.reshape(-1, 4)[:, 1:]
    V = np.asarray(mesh.points, float)
    ctr = V[F].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    keep = d < 25.0
    Fk, ctrk = F[keep], ctr[keep]
    sub = mesh.extract_cells(np.where(keep)[0]).extract_surface().triangulate().clean()
    cut = vtk.vtkCutter()
    cut.SetInputData(sub)
    pl = vtk.vtkPlane()
    cut.SetCutFunction(pl)

    rows = []
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
        P2 = best[0]
        A, B = P2, np.roll(P2, -1, axis=0)
        L = np.linalg.norm(B - A, axis=1)
        tt = (np.arange(40) + 0.5) / 40
        S = A[:, None, :] + tt[None, :, None] * (B - A)[:, None, :]
        Rs = np.linalg.norm(S, axis=2).ravel()
        Ws = np.repeat(L / 40, 40)
        so = np.argsort(Rs)
        cw = np.cumsum(Ws[so])
        secmed = Rs[so][min(np.searchsorted(cw, cw[-1] * 0.5), len(so) - 1)]

        near = np.linalg.norm(ctrk - c, axis=1) < 12.0
        f = Fk[near]
        if len(f) == 0:
            continue
        sd = (V[f] - c) @ t
        f = f[(sd.min(axis=1) < 0) & (sd.max(axis=1) > 0)]
        if len(f) == 0:
            continue
        vid = np.unique(f)
        ax = (V[vid] - c) @ t
        span = float(ax.max() - ax.min())
        # longest edge, and longest AXIAL extent, of the cut triangles
        tri = V[f]
        el = np.max([np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
                     np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                     np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=0)
        axext = ((tri - c) @ t)
        axspan_tri = np.median(axext.max(axis=1) - axext.min(axis=1))
        ringmed = float(np.median(cKDTree(
            np.vstack([coords[:-1] + q * np.diff(coords, axis=0)
                       for q in np.linspace(0, 1, 20, endpoint=False)])
        ).query(V[vid])[0]))
        rows.append((s[i], radii[i], secmed, ringmed, Rc[i], span,
                     float(np.median(el)), axspan_tri))
    return np.array(rows)


def report(name, A):
    m = np.isfinite(A[:, 2]) & (A[:, 1] > 1e-6) & np.isfinite(A[:, 4])
    A = A[m]
    r = A[:, 1]
    def_ = A[:, 2] / r                      # section / stated
    dsr = A[:, 2] / A[:, 3]                 # section / ring  (the pure surface term)
    Rc = A[:, 4]
    span = A[:, 5]
    el = A[:, 6]
    # predicted axial sagitta loss, normalised by the lumen radius
    sag = (A[:, 7] ** 2) / (8.0 * Rc) / A[:, 3]
    ok = np.isfinite(sag) & (sag < 5)

    def cor(x, y):
        x, y = np.asarray(x)[ok], np.asarray(y)[ok]
        g = np.isfinite(x) & np.isfinite(y)
        if g.sum() < 10:
            return float("nan")
        return float(np.corrcoef(x[g], y[g])[0, 1])
    print("%-15s n=%3d | sec/r %.3f  sec/ring %.3f | Rc med %6.1f mm  p10 %5.1f | "
          "tri axial span med %5.2f mm  edge med %5.2f mm | predicted sagitta/R "
          "%.3f | corr(sec/ring, 1/Rc) %+.2f  corr(sec/ring, sagitta) %+.2f"
          % (name, len(A), np.median(def_), np.median(dsr), np.median(Rc),
             np.percentile(Rc, 10), np.median(A[:, 7]), np.median(el),
             np.median(sag[ok]), cor(1.0 / Rc, dsr), cor(sag, dsr)))
    return dict(n=len(A), sec_over_r=float(np.median(def_)),
                sec_over_ring=float(np.median(dsr)), Rc_med=float(np.median(Rc)),
                Rc_p10=float(np.percentile(Rc, 10)),
                tri_axspan=float(np.median(A[:, 7])), edge=float(np.median(el)),
                sagitta_pred=float(np.median(sag[ok])),
                corr_invRc=cor(1.0 / Rc, dsr), corr_sag=cor(sag, dsr))


out = {}
host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca(hvt.branches)
print("")
print("MECHANISM TEST.  sec/ring is the pure SURFACE term: how far inside its own "
      "vertices the facetted surface lies.")
print("A 12-sided cross-section can lose at most cos(pi/12) = 0.966 to "
      "circumferential faceting.")
print("")
out["HOST"] = report("HOST", run("HOST", hm, np.asarray(hb.coordinates, float),
                                 np.asarray(hb.radii, float)))
ds = [d for d in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(d)]
if ONLY:
    ds = [d for d in ds if os.path.basename(d) in ONLY]
for d in ds:
    nm = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    out[nm] = report(nm, run(nm, m, np.asarray(br.coordinates, float),
                             np.asarray(br.radii, float)))

co = [v for k, v in out.items() if k != "HOST"]
if co:
    print("")
    print("cohort medians: sec/r %.3f   sec/ring %.3f   Rc %.1f mm   tri axial span "
          "%.2f mm   predicted sagitta/R %.3f"
          % tuple(np.median([v[k] for v in co]) for k in
                  ("sec_over_r", "sec_over_ring", "Rc_med", "tri_axspan",
                   "sagitta_pred")))
    print("cohort corr(sec/ring, 1/Rc) median %+.2f ; corr(sec/ring, sagitta) "
          "median %+.2f" % (np.median([v["corr_invRc"] for v in co]),
                            np.median([v["corr_sag"] for v in co])))
print("")
print("MACHINE_JSON")
print(json.dumps(out))
