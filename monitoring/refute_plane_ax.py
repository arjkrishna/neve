"""REFUTATION PROBE 2: is the measured cross-section shape real, or an artifact
of a badly-oriented cutting plane?

For a CIRCULAR tube cut by a plane tilted by phi from perpendicular, the section
is an ellipse whose SEMI-MINOR axis is exactly the true radius and whose
semi-major is r/cos(phi).  So ovality (Rmin/Rmax << 1) is the signature of tilt.
Test: search the plane normal that MINIMISES the cross-sectional area (the
minimum-area cut of a tube IS the perpendicular one) and re-measure there.

Also:
  - naive tangent vs PCA-smoothed tangent vs optimal normal, tilt angles reported
  - TRUE polygon side count from the ordered loop
  - vertex ring: perpendicular distance from the axis to the vertices of the
    cells the optimal plane cuts
  - inside/outside of the centerline w.r.t. the FULL (uncropped) mesh
  - watertightness of the meshes
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

vtk.vtkObject.GlobalWarningDisplayOff()
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except Exception:
    pass

ROOT = "/opt/eve_training/results_topbrain/anatomies"
SPLIT = 137.0
STRIDE = int(os.environ.get("STRIDE", "3"))
ONLY = sys.argv[1:] if len(sys.argv) > 1 else None


def rcca(branches):
    return next(b for b in branches if "RCCA" in str(b.name).upper())


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
    seen = set()
    loops = []
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


def section(cut, plane, c, nrm):
    """Return stats of the loop encircling c on the plane (c, nrm), or None."""
    u, w = basis(nrm)
    plane.SetOrigin(float(c[0]), float(c[1]), float(c[2]))
    plane.SetNormal(float(nrm[0]), float(nrm[1]), float(nrm[2]))
    cut.Update()
    out = cut.GetOutput()
    if out.GetNumberOfPoints() < 3:
        return None
    P = np.asarray(pv.wrap(out).points)
    lines = np.asarray(pv.wrap(out).lines)
    if lines.size == 0:
        return None
    best = None
    for lp in ordered_loops(lines):
        Q = P[lp] - c
        P2 = np.stack([Q @ u, Q @ w], axis=1)
        th = np.arctan2(P2[:, 1], P2[:, 0])
        d = np.diff(np.concatenate([th, th[:1]]))
        d = (d + np.pi) % (2 * np.pi) - np.pi
        if abs(d.sum()) / (2 * np.pi) < 0.9:
            continue
        x, y = P2[:, 0], P2[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        A = P2
        B = np.roll(P2, -1, axis=0)
        L = np.linalg.norm(B - A, axis=1)
        NS = 40
        tt = (np.arange(NS) + 0.5) / NS
        S = A[:, None, :] + tt[None, :, None] * (B - A)[:, None, :]
        Rs = np.linalg.norm(S, axis=2).ravel()
        Ws = np.repeat(L / NS, NS)
        o = np.argsort(Rs)
        cw = np.cumsum(Ws[o])
        med = Rs[o][min(np.searchsorted(cw, cw[-1] * 0.5), len(o) - 1)]
        st = dict(area=area, Rmed=med, Rmin=Rs.min(), Rmax=Rs.max(),
                  Rarea=np.sqrt(area / np.pi), nvert=len(lp),
                  cent=np.linalg.norm(P2.mean(axis=0)), perim=L.sum())
        if best is None or st["area"] < best["area"]:
            best = st
    return best


def perturb(t, a, b):
    u, w = basis(t)
    v = t + a * u + b * w
    return v / np.linalg.norm(v)


def run(name, mesh, coords, radii):
    n = len(coords)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    ctr = v[f].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    sub = mesh.extract_cells(np.where(d < 25.0)[0]).extract_surface().triangulate().clean()

    full = mesh
    edges = full.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                       manifold_edges=False, non_manifold_edges=False)
    nopen = edges.n_cells
    try:
        inside = np.asarray(pv.PolyData(coords).select_enclosed_points(
            full, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
    except Exception:
        inside = np.zeros(n, bool)

    cut = vtk.vtkCutter()
    cut.SetInputData(sub)
    plane = vtk.vtkPlane()
    cut.SetCutFunction(plane)

    vt = cKDTree(sub.points)

    idxs = list(range(0, n, STRIDE))
    K = ["area", "Rmed", "Rmin", "Rmax", "Rarea", "nvert", "cent"]
    NAI = {k: np.full(len(idxs), np.nan) for k in K}
    OPT = {k: np.full(len(idxs), np.nan) for k in K}
    SMO = {k: np.full(len(idxs), np.nan) for k in K}
    tilt_opt = np.full(len(idxs), np.nan)
    tilt_smo = np.full(len(idxs), np.nan)
    ringmin = np.full(len(idxs), np.nan)
    ringmed = np.full(len(idxs), np.nan)
    ringn = np.full(len(idxs), np.nan)

    for j, i in enumerate(idxs):
        c = coords[i]
        if i == 0:
            t0 = coords[1] - coords[0]
        elif i == n - 1:
            t0 = coords[-1] - coords[-2]
        else:
            t0 = coords[i + 1] - coords[i - 1]
        t0 = t0 / np.linalg.norm(t0)
        # PCA-smoothed tangent over +/-3 mm of arclength
        m = np.abs(s - s[i]) < 3.0
        if m.sum() >= 3:
            X = coords[m] - coords[m].mean(axis=0)
            ts = np.linalg.svd(X, full_matrices=False)[2][0]
            if ts @ t0 < 0:
                ts = -ts
        else:
            ts = t0

        st = section(cut, plane, c, t0)
        if st:
            for k in K:
                NAI[k][j] = st[k]
        st = section(cut, plane, c, ts)
        if st:
            for k in K:
                SMO[k][j] = st[k]
        tilt_smo[j] = np.degrees(np.arccos(np.clip(ts @ t0, -1, 1)))

        # minimum-area normal search around the smoothed tangent
        best, bn = None, ts
        for stage, (amp, nst) in enumerate(((0.45, 5), (0.15, 5), (0.05, 5))):
            grid = np.linspace(-amp, amp, nst)
            for a in grid:
                for b in grid:
                    nn = perturb(bn, a, b)
                    st = section(cut, plane, c, nn)
                    if st is None:
                        continue
                    if best is None or st["area"] < best["area"]:
                        best, cand = st, nn
            if best is not None:
                bn = cand
        if best is not None:
            for k in K:
                OPT[k][j] = best[k]
            tilt_opt[j] = np.degrees(np.arccos(np.clip(abs(bn @ t0), -1, 1)))
            # vertex ring about the optimal axis
            ids = vt.query_ball_point(c, 12.0)
            if ids:
                q = sub.points[ids] - c
                ax = q @ bn
                sel = np.abs(ax) < 0.75
                if sel.sum() >= 3:
                    rr = np.linalg.norm(q[sel] - ax[sel][:, None] * bn[None], axis=1)
                    rr = rr[rr < 3.0 * best["Rmed"]]
                    if len(rr) >= 3:
                        ringmin[j] = rr.min()
                        ringmed[j] = np.median(rr)
                        ringn[j] = len(rr)

    return dict(name=name, s=s[idxs], r=radii[idxs], inside=inside[idxs],
                nopen=nopen, ncell=full.n_cells,
                nai=NAI, opt=OPT, smo=SMO, tilt_opt=tilt_opt, tilt_smo=tilt_smo,
                ringmin=ringmin, ringmed=ringmed, ringn=ringn,
                inside_all=float(inside.mean()))


def agg(p, lo, hi):
    s, r = p["s"], p["r"]
    base = (s >= lo) & (s < hi) & (r > 1e-6)
    o = {"n": int(base.sum())}
    if base.sum() < 3:
        return o
    o["stated_r"] = float(np.median(r[base]))
    o["inside"] = float(p["inside"][base].mean())
    for tag, D in (("nai", p["nai"]), ("smo", p["smo"]), ("opt", p["opt"])):
        m = base & np.isfinite(D["Rmed"])
        o["n_" + tag] = int(m.sum())
        if m.sum() < 3:
            continue
        o[tag + "_Rmed"] = float(np.median(D["Rmed"][m] / r[m]))
        o[tag + "_Rmin"] = float(np.median(D["Rmin"][m] / r[m]))
        o[tag + "_Rmax"] = float(np.median(D["Rmax"][m] / r[m]))
        o[tag + "_Rarea"] = float(np.median(D["Rarea"][m] / r[m]))
        o[tag + "_Rarea_mm"] = float(np.median(D["Rarea"][m]))
        o[tag + "_minmax"] = float(np.median(D["Rmin"][m] / D["Rmax"][m]))
        o[tag + "_nvert"] = float(np.median(D["nvert"][m]))
        o[tag + "_cent"] = float(np.median(D["cent"][m] / r[m]))
    m = base & np.isfinite(p["tilt_opt"])
    if m.sum() >= 3:
        o["tilt_opt_deg"] = float(np.median(p["tilt_opt"][m]))
        o["tilt_opt_p90"] = float(np.percentile(p["tilt_opt"][m], 90))
    m = base & np.isfinite(p["tilt_smo"])
    if m.sum() >= 3:
        o["tilt_smo_deg"] = float(np.median(p["tilt_smo"][m]))
    m = base & np.isfinite(p["ringmin"])
    if m.sum() >= 3:
        o["ringmin"] = float(np.median(p["ringmin"][m] / r[m]))
        o["ringmed"] = float(np.median(p["ringmed"][m] / r[m]))
        o["ringn"] = float(np.median(p["ringn"][m]))
    return o


res = {}
prof = {}
host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca(hvt.branches)
p = run("HOST", hm, np.asarray(hb.coordinates, float), np.asarray(hb.radii, float))
prof["HOST"] = p
res["HOST"] = {t_: agg(p, lo, hi) for t_, lo, hi in
               (("ALL", -1, 1e9), ("PROX", -1, SPLIT), ("DIST", SPLIT, 1e9))}
print("done HOST", file=sys.stderr)

dirs_ = [d for d in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(d)]
if ONLY:
    dirs_ = [d for d in dirs_ if os.path.basename(d) in ONLY]
for d in dirs_:
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    p = run(name, m, np.asarray(br.coordinates, float), np.asarray(br.radii, float))
    prof[name] = p
    res[name] = {t_: agg(p, lo, hi) for t_, lo, hi in
                 (("ALL", -1, 1e9), ("PROX", -1, SPLIT), ("DIST", SPLIT, 1e9))}
    print("done " + name, file=sys.stderr)

print("")
print("MESH TOPOLOGY  (open boundary edges of the FULL mesh; fraction of RCCA "
      "centerline stations INSIDE the full mesh)")
for k in ["HOST"] + [x for x in sorted(prof) if x != "HOST"]:
    p = prof[k]
    print("  %-16s cells %6d  open boundary edges %5d   centerline inside %.3f"
          % (k, p["ncell"], p["nopen"], p["inside_all"]))

print("")
W = ("%-15s%-5s%4s%6s|%7s%7s%7s%7s%6s|%7s%7s%7s%7s%6s|%7s%7s%7s%7s%6s%6s|%6s%6s|"
     "%7s%7s%5s")
print("=" * 165)
print("CROSS-SECTION UNDER THREE PLANE ORIENTATIONS.  ratios to stated_r.  "
      "NAIVE = the tangent used by the original pipeline")
print("=" * 165)
print(W % ("anatomy", "seg", "n", "r_mm",
           "Nmed", "Nmin", "Nmax", "Nar", "Nmnmx",
           "Smed", "Smin", "Smax", "Sar", "Smnmx",
           "Omed", "Omin", "Omax", "Oar", "Omnmx", "Onv",
           "tiltO", "tiltS", "ringmn", "ringmd", "rngN"))
for k in ["HOST"] + [x for x in sorted(res) if x != "HOST"]:
    for seg in ("ALL", "PROX", "DIST"):
        a = res[k][seg]
        if "stated_r" not in a:
            print("%-15s%-5s  -- insufficient --" % (k, seg))
            continue

        def g(z, w=7, a=a):
            return (("%" + str(w) + ".3f") % a[z]) if z in a else " " * (w - 2) + "--"
        print(W % (k, seg, a["n"], "%6.2f" % a["stated_r"],
                   g("nai_Rmed"), g("nai_Rmin"), g("nai_Rmax"), g("nai_Rarea"),
                   g("nai_minmax", 6),
                   g("smo_Rmed"), g("smo_Rmin"), g("smo_Rmax"), g("smo_Rarea"),
                   g("smo_minmax", 6),
                   g("opt_Rmed"), g("opt_Rmin"), g("opt_Rmax"), g("opt_Rarea"),
                   g("opt_minmax", 6), g("opt_nvert", 6),
                   g("tilt_opt_deg", 6), g("tilt_smo_deg", 6),
                   g("ringmin"), g("ringmed"), g("ringn", 5)))

print("")
print("ABSOLUTE LUMEN (optimal plane, area-equivalent radius, mm) vs stated r (mm)")
for k in ["HOST"] + [x for x in sorted(res) if x != "HOST"]:
    for seg in ("ALL", "PROX", "DIST"):
        a = res[k][seg]
        if "opt_Rarea_mm" in a:
            print("  %-15s %-5s  R_area %.3f mm   stated %.3f mm   ratio %.3f   "
                  "tilt_opt %.1f deg (p90 %.1f)"
                  % (k, seg, a["opt_Rarea_mm"], a["stated_r"], a["opt_Rarea"],
                     a.get("tilt_opt_deg", float("nan")),
                     a.get("tilt_opt_p90", float("nan"))))

print("")
print("MACHINE_JSON")
print(json.dumps(res))
