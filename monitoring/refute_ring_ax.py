"""The diagnosis called the VERTEX-RING test decisive: it reported the nearest
mesh vertex in a +/-1 mm slab at 0.931 of the stated radius and concluded
"the generator places tube vertices ON the declared circle, stated_r IS the
polygon circumradius", which is what rules out RADII_INFLATED and forces the
polygon-chord (SURFACE_ERODED) explanation.

That test used (a) a MINIMUM over very few vertices and (b) a 2.5x-median
outlier filter.  Re-do it properly: take exactly the vertices of the triangles
that the perpendicular plane actually CUTS -- that is the true local ring, no
slab, no outlier filter -- and report the whole distribution of their
perpendicular distance to the axis, with counts.

If the ring really sits at ~1.0 r while the exact section curve sits at 0.87 r,
a 12-sided polygon would have to have a chord deficit of 13%, which needs
n = pi/acos(0.87) = 6.2, not 12.  Reporting both settles it.
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
SPLIT = 137.0
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
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    F = mesh.faces.reshape(-1, 4)[:, 1:]
    V = np.asarray(mesh.points, float)
    ctr = V[F].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    keep = d < 25.0
    Fk = F[keep]
    ctrk = ctr[keep]
    sub = mesh.extract_cells(np.where(keep)[0]).extract_surface().triangulate().clean()
    cut = vtk.vtkCutter()
    cut.SetInputData(sub)
    pl = vtk.vtkPlane()
    cut.SetCutFunction(pl)

    # dense centerline polyline for curvature-free vertex-to-axis distance
    seg = np.linspace(0, 1, 20, endpoint=False)
    DENSE = (coords[:-1][:, None, :]
             + seg[None, :, None] * np.diff(coords, axis=0)[:, None, :]
             ).reshape(-1, 3)
    DENSE = np.vstack([DENSE, coords[-1:]])
    band15, band30, band60, curvering = [], [], [], []

    rows = []
    for i in range(0, n, 2):
        c = coords[i]
        t = coords[min(i + 1, n - 1)] - coords[max(i - 1, 0)]
        t = t / np.linalg.norm(t)
        u, w = basis(t)

        # ---- exact section (reference) ----
        pl.SetOrigin(*[float(x) for x in c])
        pl.SetNormal(*[float(x) for x in t])
        cut.Update()
        o = pv.wrap(cut.GetOutput())
        secmed = np.nan
        secmax = np.nan
        secn = np.nan
        if o.n_points >= 3 and np.asarray(o.lines).size:
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
            if best is not None:
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
                secmax = Rs.max()
                secn = len(P2)

        # ---- TRUE local ring: vertices of the triangles this plane cuts ----
        near = np.linalg.norm(ctrk - c, axis=1) < 12.0
        f = Fk[near]
        if len(f) == 0:
            continue
        sd = (V[f] - c) @ t                      # signed plane distance, 3 per tri
        cutmask = (sd.min(axis=1) < 0) & (sd.max(axis=1) > 0)
        f = f[cutmask]
        if len(f) == 0:
            continue
        vid = np.unique(f)
        q = V[vid] - c
        ax = q @ t
        rp = np.linalg.norm(q - ax[:, None] * t[None], axis=1)
        # curvature-free alternative: distance from each vertex to the centerline
        # CURVE (nearest point on the dense polyline), not to the tangent LINE
        dcurve = cKDTree(DENSE).query(V[vid])[0]
        # axially band-limited tangent-line ring
        for BW, box in ((1.5, band15), (3.0, band30), (6.0, band60)):
            sel = np.abs(ax) < BW
            rr = rp[sel]
            if np.isfinite(secmed):
                rr = rr[rr < 2.0 * secmed]
            box.append(np.median(rr) / radii[i] if len(rr) >= 3 and radii[i] > 1e-6
                       else np.nan)
        rrc = dcurve
        if np.isfinite(secmed):
            rrc = rrc[rrc < 2.0 * secmed]
        curvering.append(np.median(rrc) / radii[i]
                         if len(rrc) >= 3 and radii[i] > 1e-6 else np.nan)
        # keep the ring of THIS lumen: within 2x the section's own median radius
        if np.isfinite(secmed):
            rp2 = rp[rp < 2.0 * secmed]
        else:
            rp2 = rp
        if len(rp2) < 3:
            continue
        rows.append((s[i], radii[i], secmed, secmax, secn,
                     rp2.min(), np.percentile(rp2, 25), np.median(rp2),
                     np.percentile(rp2, 75), rp2.max(), len(rp2),
                     np.abs(ax[rp < 2.0 * secmed]).max() if np.isfinite(secmed)
                     else np.nan))
    def med(x):
        x = np.array(x, float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else float('nan')
    extra = dict(band15=med(band15), band30=med(band30), band60=med(band60),
                 curvering=med(curvering))
    return (np.array(rows), extra) if rows else None


def agg(A, lo, hi):
    if A is None:
        return None
    m = (A[:, 0] >= lo) & (A[:, 0] < hi) & (A[:, 1] > 1e-6) & np.isfinite(A[:, 2])
    if m.sum() < 3:
        return None
    r = A[m, 1]
    return dict(
        n=int(m.sum()),
        sec_med=float(np.median(A[m, 2] / r)),
        sec_max=float(np.median(A[m, 3] / r)),
        sec_nv=float(np.median(A[m, 4])),
        ring_min=float(np.median(A[m, 5] / r)),
        ring_p25=float(np.median(A[m, 6] / r)),
        ring_med=float(np.median(A[m, 7] / r)),
        ring_p75=float(np.median(A[m, 8] / r)),
        ring_max=float(np.median(A[m, 9] / r)),
        ring_n=float(np.median(A[m, 10])),
        ring_over_sec=float(np.median(A[m, 7] / A[m, 2])),
        axspan=float(np.median(A[m, 11])),
        stated_r=float(np.median(r)))


out = {}
host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca(hvt.branches)
RH = run("HOST", hm, np.asarray(hb.coordinates, float), np.asarray(hb.radii, float))
A, EX = RH
EXTRA = {"HOST": EX}
out["HOST"] = {k: agg(A, l, h) for k, l, h in
               (("ALL", -1, 1e9), ("PROX", -1, SPLIT), ("DIST", SPLIT, 1e9))}
ds = [d for d in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(d)]
if ONLY:
    ds = [d for d in ds if os.path.basename(d) in ONLY]
for d in ds:
    nm = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    A, EX = run(nm, m, np.asarray(br.coordinates, float), np.asarray(br.radii, float))
    EXTRA[nm] = EX
    out[nm] = {k: agg(A, l, h) for k, l, h in
               (("ALL", -1, 1e9), ("PROX", -1, SPLIT), ("DIST", SPLIT, 1e9))}
    print("done " + nm, file=sys.stderr)

print("")
print("TRUE LOCAL VERTEX RING = vertices of the triangles the perpendicular plane CUTS")
print("all columns are ratios to the stated centerline radius")
print("%-15s%-5s%5s%7s|%8s%8s%7s|%8s%8s%8s%8s%8s%7s|%9s%7s" % (
    "anatomy", "seg", "n", "r_mm", "sec_med", "sec_max", "sec_nv",
    "ring_mn", "ring25", "ring_md", "ring75", "ring_mx", "ring_n",
    "ring/sec", "axspan"))
for k in ["HOST"] + [x for x in sorted(out) if x != "HOST"]:
    for seg in ("ALL", "PROX", "DIST"):
        a = out[k][seg]
        if a is None:
            print("%-15s%-5s   --" % (k, seg))
            continue
        print("%-15s%-5s%5d%7.2f|%8.3f%8.3f%7.1f|%8.3f%8.3f%8.3f%8.3f%8.3f%7.1f|"
              "%9.3f%7.2f" % (
                  k, seg, a["n"], a["stated_r"], a["sec_med"], a["sec_max"],
                  a["sec_nv"], a["ring_min"], a["ring_p25"], a["ring_med"],
                  a["ring_p75"], a["ring_max"], a["ring_n"], a["ring_over_sec"],
                  a["axspan"]))

co = [out[k]["ALL"] for k in out if k != "HOST" and out[k]["ALL"]]
if co:
    print("")
    for f in ("sec_med", "ring_min", "ring_med", "ring_max", "ring_over_sec",
              "sec_nv", "ring_n"):
        v = [x[f] for x in co]
        print("  cohort %-14s median %.3f   range %.3f - %.3f"
              % (f, np.median(v), min(v), max(v)))
    print("")
    print("  implied polygon sides if the section deficit were pure chord error:")
    sm = np.median([x["sec_med"] for x in co])
    rm = np.median([x["ring_med"] for x in co])
    import math
    ratio = sm / rm
    print("    section_median / ring_median = %.3f -> n = pi/acos(%.3f) = %.1f"
          % (ratio, min(ratio, 0.999999), math.pi / math.acos(min(ratio, 0.999999))))
    print("    but the DIRECTLY COUNTED section polygon has %.0f sides, for which"
          % np.median([x["sec_nv"] for x in co]))
    nv = np.median([x["sec_nv"] for x in co])
    print("    the maximum possible chord deficit is cos(pi/n) = %.3f"
          % math.cos(math.pi / nv))

print("")
print("CURVATURE-CONTROLLED RING.  ring_md above uses the TANGENT LINE and is fed by")
print("vertices up to +/-8-10 mm away axially on a CURVED tube, which inflates it.")
print("%-15s%12s%12s%12s%14s%12s" % ("anatomy", "band|ax|<1.5", "<3.0", "<6.0",
                                     "dist-to-CURVE", "section"))
for k in ["HOST"] + [x for x in sorted(EXTRA) if x != "HOST"]:
    e = EXTRA[k]
    sm = out[k]["ALL"]["sec_med"] if out[k]["ALL"] else float("nan")
    print("%-15s%12.3f%12.3f%12.3f%14.3f%12.3f" % (
        k, e["band15"], e["band30"], e["band60"], e["curvering"], sm))

print("")
print("MACHINE_JSON")
print(json.dumps(out))
