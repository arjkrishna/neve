"""SYNTHETIC GROUND TRUTH.  Builds tubes whose true radius is known exactly and
runs BOTH estimators on them: the original slab+azimuth-bin one and my exact
plane-section one.  Answers, without any appeal to theory:

  (a) how much does the slab+bin estimator bias a KNOWN circular tube?
  (b) does a coarsely DECIMATED tube (median edge ~6.5 mm on a ~2.4 mm-radius
      lumen, i.e. exactly the real cohort mesh) reproduce the observed
      Rmed/r ~ 0.87, Rmin/Rmax ~ 0.65, ring/r ~ 1.0 signature?
  (c) what does a genuine 4-gon tube read (the diagnosis predicted 0.765)?
  (d) does a NEIGHBOURING vessel 6 mm away inside the slab corrupt anything?
  (e) does a jagged / noisy centerline (bad tangent) corrupt anything?
"""
import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

vtk.vtkObject.GlobalWarningDisplayOff()
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except Exception:
    pass

SLAB, NBIN, SUBN, BALL = 1.0, 36, 8, 12.0


def bary_lattice(n):
    us, vs = [], []
    for i in range(n):
        for j in range(n - i):
            us.append((i + 1.0 / 3) / n)
            vs.append((j + 1.0 / 3) / n)
        for j in range(n - i - 1):
            us.append((i + 2.0 / 3) / n)
            vs.append((j + 2.0 / 3) / n)
    return np.array(us)[:, None], np.array(vs)[:, None]


U, V = bary_lattice(SUBN)


def dense_surface(mesh, C=None):
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    if C is not None:
        ctr = v[f].mean(axis=1)
        d, _ = cKDTree(C).query(ctr)
        f = f[d < 15.0]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    pts = a[None] + U[:, :, None] * (b - a)[None] + V[:, :, None] * (c - a)[None]
    return np.vstack([v, pts.reshape(-1, 3)])


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
    for start in list(adj):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def make_tube(axis_fn, rad_fn, n_circ, n_ax, tmax, ell=1.0):
    ts = np.linspace(0, tmax, n_ax)
    C = np.array([axis_fn(t) for t in ts])
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    pts = []
    up = np.array([0.0, 0.0, 1.0])
    for i, (c, t) in enumerate(zip(C, T)):
        u = np.cross(t, up)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(t, np.array([1.0, 0.0, 0.0]))
        u /= np.linalg.norm(u)
        w = np.cross(t, u)
        a = 2 * np.pi * np.arange(n_circ) / n_circ
        r = rad_fn(ts[i])
        pts.append(c + (r * np.cos(a))[:, None] * u
                   + (r * ell * np.sin(a))[:, None] * w)
    P = np.vstack(pts)
    faces = []
    for i in range(n_ax - 1):
        for j in range(n_circ):
            a0 = i * n_circ + j
            a1 = i * n_circ + (j + 1) % n_circ
            b0 = a0 + n_circ
            b1 = a1 + n_circ
            faces += [3, a0, a1, b1, 3, a0, b1, b0]
    m = pv.PolyData(P, np.array(faces))
    # cap the ends so it is closed
    return m.clean().triangulate(), C, T


def measure(mesh, C, r_true, label, jitter=0.0, seed=0):
    """Run BOTH estimators on the same stations."""
    rng = np.random.default_rng(seed)
    Cq = C + rng.normal(0, jitter, C.shape) if jitter > 0 else C
    n = len(Cq)
    surf = dense_surface(mesh, Cq)
    tree = cKDTree(surf)
    vt = cKDTree(mesh.points)
    cut = vtk.vtkCutter()
    cut.SetInputData(mesh)
    plane = vtk.vtkPlane()
    cut.SetCutFunction(plane)

    lo, hi = int(0.2 * n), int(0.8 * n)
    acc = {k: [] for k in ["sl_med", "sl_min", "sl_max", "sl_area", "nvert",
                           "b_med", "b_min", "b_max", "b_ecc", "clr",
                           "ring_min", "ring_med", "minmax"]}
    for i in range(lo, hi, 4):
        c = Cq[i]
        t = Cq[min(i + 1, n - 1)] - Cq[max(i - 1, 0)]
        t = t / np.linalg.norm(t)
        u, w = basis(t)

        plane.SetOrigin(*[float(x) for x in c])
        plane.SetNormal(*[float(x) for x in t])
        cut.Update()
        o = pv.wrap(cut.GetOutput())
        if o.n_points >= 3 and np.asarray(o.lines).size:
            P = np.asarray(o.points)
            best = None
            for lp in ordered_loops(np.asarray(o.lines)):
                Q = P[lp] - c
                P2 = np.stack([Q @ u, Q @ w], axis=1)
                th = np.arctan2(P2[:, 1], P2[:, 0])
                d = np.diff(np.concatenate([th, th[:1]]))
                d = (d + np.pi) % (2 * np.pi) - np.pi
                if abs(d.sum()) / (2 * np.pi) < 0.9:
                    continue
                x, y = P2[:, 0], P2[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                A, B = P2, np.roll(P2, -1, axis=0)
                L = np.linalg.norm(B - A, axis=1)
                tt = (np.arange(40) + 0.5) / 40
                S = A[:, None, :] + tt[None, :, None] * (B - A)[:, None, :]
                Rs = np.linalg.norm(S, axis=2).ravel()
                Ws = np.repeat(L / 40, 40)
                so = np.argsort(Rs)
                cw = np.cumsum(Ws[so])
                med = Rs[so][min(np.searchsorted(cw, cw[-1] * 0.5), len(so) - 1)]
                st = (med, Rs.min(), Rs.max(), np.sqrt(area / np.pi), len(lp), area)
                if best is None or st[5] < best[5]:
                    best = st
            if best:
                acc["sl_med"].append(best[0])
                acc["sl_min"].append(best[1])
                acc["sl_max"].append(best[2])
                acc["sl_area"].append(best[3])
                acc["nvert"].append(best[4])
                acc["minmax"].append(best[1] / best[2])

        idx = tree.query_ball_point(c, BALL)
        if idx:
            p = surf[idx] - c
            ax = p @ t
            m = np.abs(ax) < SLAB
            if m.sum() >= NBIN:
                q = p[m] - ax[m][:, None] * t[None]
                R = np.linalg.norm(q, axis=1)
                th = np.arctan2(q @ w, q @ u)
                b = ((th + np.pi) / (2 * np.pi) * NBIN).astype(int) % NBIN
                wallR = np.full(NBIN, np.inf)
                np.minimum.at(wallR, b, R)
                ok = np.isfinite(wallR)
                if ok.sum() >= NBIN // 2:
                    rw = wallR[ok]
                    bc = (np.arange(NBIN)[ok] + 0.5) / NBIN * 2 * np.pi - np.pi
                    qv = rw[:, None] * np.stack([np.cos(bc), np.sin(bc)], axis=1)
                    acc["b_med"].append(np.median(rw))
                    acc["b_min"].append(rw.min())
                    acc["b_max"].append(rw.max())
                    acc["b_ecc"].append(np.linalg.norm(qv.mean(axis=0)))
        acc["clr"].append(tree.query(c)[0])

        ids = vt.query_ball_point(c, BALL)
        if ids:
            q = mesh.points[ids] - c
            ax = q @ t
            m = np.abs(ax) < SLAB
            if m.sum() >= 3:
                rr = np.linalg.norm(q[m] - ax[m][:, None] * t[None], axis=1)
                rr = rr[rr < 2.5 * np.median(rr)]
                if len(rr) >= 3:
                    acc["ring_min"].append(rr.min())
                    acc["ring_med"].append(np.median(rr))

    ed = mesh.extract_all_edges()
    ep = ed.points[ed.lines.reshape(-1, 3)[:, 1:]]
    el = np.linalg.norm(ep[:, 0] - ep[:, 1], axis=1)

    def g(k):
        return np.median(acc[k]) / r_true if acc[k] else float("nan")
    print("%-34s r_true %.2f  cells %5d  edge_med %5.2f | SLmed %.3f SLmin %.3f "
          "SLmax %.3f SLarea %.3f mn/mx %.3f nv %4.1f | BINmed %.3f BINmin %.3f "
          "BINmax %.3f ecc %.3f | clr %.3f | ring_mn %.3f ring_md %.3f | BIN/SL %.3f"
          % (label, r_true, mesh.n_cells, np.median(el),
             g("sl_med"), g("sl_min"), g("sl_max"), g("sl_area"),
             np.median(acc["minmax"]) if acc["minmax"] else float("nan"),
             np.median(acc["nvert"]) if acc["nvert"] else float("nan"),
             g("b_med"), g("b_min"), g("b_max"), g("b_ecc"), g("clr"),
             g("ring_min"), g("ring_med"),
             (np.median(acc["b_med"]) / np.median(acc["sl_med"]))
             if acc["b_med"] and acc["sl_med"] else float("nan")))


R = 2.4
# gently curved axis, radius of curvature ~ 60 mm, length ~ 200 mm
def ax_fn(t):
    return np.array([60 * np.sin(t / 60.0), 60 * (1 - np.cos(t / 60.0)), 0.15 * t])


print("=" * 190)
print("SYNTHETIC TUBES OF KNOWN RADIUS.  All numbers are ratios to the TRUE radius.")
print("=" * 190)

for nc in (4, 6, 8, 12, 24, 60):
    m, C, T = make_tube(ax_fn, lambda t: R, nc, 200, 200.0)
    measure(m, C, R, "circular tube, n_circ=%2d, fine axial" % nc)

print("")
print("-- effect of AXIAL coarseness at n_circ=12 --")
for na in (200, 120, 60, 31):
    m, C, T = make_tube(ax_fn, lambda t: R, 12, na, 200.0)
    Cf = np.array([ax_fn(t) for t in np.linspace(0, 200.0, 200)])
    measure(m, Cf, R, "n_circ=12, axial spacing %.1f mm" % (200.0 / (na - 1)))

print("")
print("-- DECIMATED to the real cohort's tessellation coarseness --")
for red in (0.90, 0.95, 0.965, 0.975, 0.98, 0.985):
    m0, C, T = make_tube(ax_fn, lambda t: R, 40, 300, 200.0)
    m = m0.decimate(red).clean().triangulate()
    Cf = np.array([ax_fn(t) for t in np.linspace(0, 200.0, 200)])
    measure(m, Cf, R, "decimated %.3f" % red)

print("")
print("-- ELLIPTICAL tube (true ovality), n_circ=24 --")
for e in (1.0, 0.85, 0.7, 0.55):
    m, C, T = make_tube(ax_fn, lambda t: R, 24, 200, 200.0, ell=e)
    measure(m, C, R, "elliptical b/a=%.2f (a=r_true)" % e)

print("")
print("-- NEIGHBOURING vessel 6 mm away (tests slab contamination) --")
m1, C, T = make_tube(ax_fn, lambda t: R, 24, 200, 200.0)
m2, _, _ = make_tube(lambda t: ax_fn(t) + np.array([0.0, 0.0, 6.0]),
                     lambda t: R, 24, 200, 200.0)
measure(m1, C, R, "isolated tube")
measure((m1 + m2).clean(), C, R, "same tube + neighbour at 6.0 mm")
m3, _, _ = make_tube(lambda t: ax_fn(t) + np.array([0.0, 0.0, 5.2]),
                     lambda t: R, 24, 200, 200.0)
measure((m1 + m3).clean(), C, R, "same tube + neighbour at 5.2 mm")

print("")
print("-- JAGGED centerline (bad tangent estimate) --")
m, C, T = make_tube(ax_fn, lambda t: R, 24, 200, 200.0)
for j in (0.0, 0.05, 0.1, 0.2, 0.4):
    measure(m, C, R, "centerline jitter sigma=%.2f mm" % j, jitter=j, seed=1)

print("")
print("-- CENTERLINE displaced off-axis (tests the ecc/off-axis rejection) --")
m, C, T = make_tube(ax_fn, lambda t: R, 24, 200, 200.0)
for off in (0.0, 0.2, 0.5, 0.8):
    Co = C + np.array([0.0, 0.0, off])
    measure(m, Co, R, "centerline offset %.1f mm" % off)
