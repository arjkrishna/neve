"""REFUTATION PROBE: re-measure the RCCA lumen with methods that share NO
assumption with the slab+azimuth-bin pipeline.

Method A: EXACT PLANE CROSS-SECTION.  vtkCutter with the plane (c[i], t_i);
          zero thickness (no slab), connectivity to isolate THE lumen loop that
          encircles c[i] (no neighbouring-vessel contamination possible), true
          ordered polygon from the line cells (no azimuth bins, no per-bin MIN).
Method B: RAY CAST.  vtkOBBTree, FIRST intersection only, 72 in-plane
          directions from c[i].  No slab, no bins, no min-operator.
Method C: inside/outside test of c[i] w.r.t. the mesh.

Also re-runs the ORIGINAL slab+bin estimator on the SAME stations so the two
are differenced station-by-station.
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
vtk_quiet=True
from scipy.spatial import cKDTree
import vtk as _v
_v.vtkObject.GlobalWarningDisplayOff()
try:
    _v.vtkLogger.SetStderrVerbosity(_v.vtkLogger.VERBOSITY_OFF)
except Exception:
    pass

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
SPLIT = 137.0
NRAY = 72
SLAB = 1.0
NBIN = 36
SUBN = 24
CROP = 25.0
BALL = 12.0

ONLY = sys.argv[1:] if len(sys.argv) > 1 else None


def rcca(branches):
    return next(b for b in branches if "RCCA" in str(b.name).upper())


def frame(coords, i, n):
    if i == 0:
        t = coords[1] - coords[0]
    elif i == n - 1:
        t = coords[-1] - coords[-2]
    else:
        t = coords[i + 1] - coords[i - 1]
    t = t / np.linalg.norm(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u = u / np.linalg.norm(u)
    w = np.cross(t, u)
    return t, u, w


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


def dense_surface(mesh, coords):
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    ctr = v[f].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    keep = d < CROP
    f = f[keep]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    pts = a[None] + U[:, :, None] * (b - a)[None] + V[:, :, None] * (c - a)[None]
    return np.vstack([v[np.unique(f)], pts.reshape(-1, 3)])


def crop_mesh(mesh, coords, rad):
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    ctr = v[f].mean(axis=1)
    d, _ = cKDTree(coords).query(ctr)
    ids = np.where(d < rad)[0]
    return mesh.extract_cells(ids).extract_surface().triangulate().clean()


def ordered_loops(P, lines):
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


def loop_stats(P2):
    q = P2
    R = np.linalg.norm(q, axis=1)
    th = np.arctan2(q[:, 1], q[:, 0])
    dth = np.diff(np.concatenate([th, th[:1]]))
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    wind = abs(dth.sum()) / (2 * np.pi)
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
    return dict(R_med_curve=float(med), R_min=float(Rs.min()), R_max=float(Rs.max()),
                R_area=float(np.sqrt(area / np.pi)), area=float(area),
                nvert=int(len(P2)), wind=float(wind),
                cent_off=float(np.linalg.norm(P2.mean(axis=0))),
                perim=float(L.sum()))


KEYS = ["R_med_curve", "R_min", "R_max", "R_area", "nvert", "wind", "cent_off",
        "perim", "area"]


def run(name, mesh, coords, radii):
    n = len(coords)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    sub = crop_mesh(mesh, coords, CROP)
    surf = dense_surface(mesh, coords)
    tree = cKDTree(surf)
    clr, _ = tree.query(coords)

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(sub)
    obb.BuildLocator()

    try:
        sel = pv.PolyData(coords).select_enclosed_points(
            mesh, tolerance=0.0, check_surface=False)
        inside = np.asarray(sel["SelectedPoints"], bool)
    except Exception:
        inside = np.zeros(n, bool)

    cut = vtk.vtkCutter()
    cut.SetInputData(sub)
    plane = vtk.vtkPlane()
    cut.SetCutFunction(plane)

    SL = {k: np.full(n, np.nan) for k in KEYS}
    RAY = {k: np.full(n, np.nan) for k in ["ray_med", "ray_min", "ray_max", "ray_miss"]}
    BIN = {k: np.full(n, np.nan) for k in ["b_med", "b_min", "b_max", "b_ecc", "b_empty"]}
    nloop = np.full(n, np.nan)

    ang = np.arange(NRAY) / NRAY * 2 * np.pi
    ca, sa = np.cos(ang), np.sin(ang)
    pts = vtk.vtkPoints()

    for i in range(n):
        t, u, w = frame(coords, i, n)
        c = coords[i]

        plane.SetOrigin(float(c[0]), float(c[1]), float(c[2]))
        plane.SetNormal(float(t[0]), float(t[1]), float(t[2]))
        cut.Update()
        sl = pv.wrap(cut.GetOutput())
        if sl.n_points >= 3:
            try:
                conn = sl.connectivity()
                rid = np.asarray(conn.point_data["RegionId"])
                P = np.asarray(conn.points)
                lines = np.asarray(conn.lines)
                loops = ordered_loops(P, lines)
                best = None
                nloop[i] = len(loops)
                for lp in loops:
                    Q = P[lp] - c
                    P2 = np.stack([Q @ u, Q @ w], axis=1)
                    st = loop_stats(P2)
                    if st["wind"] < 0.9:
                        continue
                    if best is None or st["R_med_curve"] < best["R_med_curve"]:
                        best = st
                if best is not None:
                    for k in KEYS:
                        SL[k][i] = best[k]
            except Exception:
                pass

        dirs = ca[:, None] * u[None] + sa[:, None] * w[None]
        Rr = []
        miss = 0
        for d_ in dirs:
            p1 = c + d_ * 30.0
            pts.Reset()
            hit = obb.IntersectWithLine(
                [float(c[0]), float(c[1]), float(c[2])],
                [float(p1[0]), float(p1[1]), float(p1[2])], pts, None)
            if hit == 0 or pts.GetNumberOfPoints() == 0:
                miss += 1
                continue
            Rr.append(np.linalg.norm(np.array(pts.GetPoint(0)) - c))
        if Rr:
            Rr = np.array(Rr)
            RAY["ray_med"][i] = np.median(Rr)
            RAY["ray_min"][i] = Rr.min()
            RAY["ray_max"][i] = Rr.max()
        RAY["ray_miss"][i] = miss

        idx = tree.query_ball_point(c, BALL)
        if idx:
            p = surf[idx] - c
            ax = p @ t
            m = np.abs(ax) < SLAB
            if m.sum() >= NBIN:
                qq = p[m] - ax[m][:, None] * t[None]
                R = np.linalg.norm(qq, axis=1)
                th = np.arctan2(qq @ w, qq @ u)
                b = ((th + np.pi) / (2 * np.pi) * NBIN).astype(int) % NBIN
                wallR = np.full(NBIN, np.inf)
                np.minimum.at(wallR, b, R)
                ok = np.isfinite(wallR)
                BIN["b_empty"][i] = NBIN - ok.sum()
                if ok.sum() >= NBIN // 2:
                    rw = wallR[ok]
                    bc = (np.arange(NBIN)[ok] + 0.5) / NBIN * 2 * np.pi - np.pi
                    qv = rw[:, None] * np.stack([np.cos(bc), np.sin(bc)], axis=1)
                    BIN["b_med"][i] = np.median(rw)
                    BIN["b_min"][i] = rw.min()
                    BIN["b_max"][i] = rw.max()
                    BIN["b_ecc"][i] = np.linalg.norm(qv.mean(axis=0))

    out = dict(name=name, s=s, r=radii, clr=clr, inside=inside, nloop=nloop)
    out.update(SL)
    out.update(RAY)
    out.update(BIN)
    return out


def agg(p, lo, hi):
    s, r = p["s"], p["r"]
    base = (s >= lo) & (s < hi) & (r > 1e-6)
    o = {"n_st": int(base.sum())}
    if base.sum() == 0:
        return o
    m = base & np.isfinite(p["R_med_curve"])
    o["n_slice"] = int(m.sum())
    if m.sum() >= 3:
        for k, tag in (("R_med_curve", "sl_med"), ("R_min", "sl_min"),
                       ("R_max", "sl_max"), ("R_area", "sl_area")):
            o[tag] = float(np.median(p[k][m] / r[m]))
        o["sl_med_mm"] = float(np.median(p["R_med_curve"][m]))
        o["sl_area_mm"] = float(np.median(p["R_area"][m]))
        o["nvert"] = float(np.median(p["nvert"][m]))
        o["cent_off_r"] = float(np.median(p["cent_off"][m] / r[m]))
        o["minmax"] = float(np.median(p["R_min"][m] / p["R_max"][m]))
    mr = base & np.isfinite(p["ray_med"])
    if mr.sum() >= 3:
        o["ray_med"] = float(np.median(p["ray_med"][mr] / r[mr]))
        o["ray_min"] = float(np.median(p["ray_min"][mr] / r[mr]))
        o["ray_max"] = float(np.median(p["ray_max"][mr] / r[mr]))
        o["ray_miss"] = float(np.mean(p["ray_miss"][base]))
    mb = base & np.isfinite(p["b_med"])
    if mb.sum() >= 3:
        o["b_med"] = float(np.median(p["b_med"][mb] / r[mb]))
        o["b_min"] = float(np.median(p["b_min"][mb] / r[mb]))
        o["b_max"] = float(np.median(p["b_max"][mb] / r[mb]))
        o["b_ecc"] = float(np.median(p["b_ecc"][mb] / r[mb]))
    o["clr"] = float(np.median(p["clr"][base] / r[base]))
    o["stated_r"] = float(np.median(r[base]))
    o["frac_inside"] = float(p["inside"][base].mean())
    both = base & np.isfinite(p["R_med_curve"]) & np.isfinite(p["b_med"])
    if both.sum() >= 3:
        o["paired_n"] = int(both.sum())
        o["bin_minus_slice_med"] = float(np.median(p["b_med"][both] - p["R_med_curve"][both]))
        o["bin_over_slice_med"] = float(np.median(p["b_med"][both] / p["R_med_curve"][both]))
        o["bin_over_slice_max"] = float(np.median(p["b_max"][both] / p["R_max"][both]))
        o["bin_over_slice_min"] = float(np.median(p["b_min"][both] / p["R_min"][both]))
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

hdr = ("%-16s%-5s%5s%6s%7s|%8s%8s%8s%8s%7s%7s|%8s%8s%8s%6s|%8s%8s%8s|%7s%7s" % (
    "anatomy", "seg", "nst", "nsl", "r_mm", "SLmed", "SLmin", "SLmax", "SLarea",
    "nvert", "mn/mx", "RAYmed", "RAYmin", "RAYmax", "miss", "BINmed", "BINmin",
    "BINmax", "clr/r", "insid"))
print("=" * len(hdr))
print("EXACT SLICE (SL) vs RAY vs the ORIGINAL SLAB+BIN (BIN).  all ratios to stated_r")
print("=" * len(hdr))
print(hdr)
for k in ["HOST"] + [x for x in sorted(res) if x != "HOST"]:
    for seg in ("ALL", "PROX", "DIST"):
        a = res[k][seg]
        if "stated_r" not in a:
            print("%-16s%-5s  -- insufficient --" % (k, seg))
            continue

        def g(z, a=a):
            return ("%8.3f" % a[z]) if z in a else "    --  "
        print("%-16s%-5s%5d%6d%7.2f|%s%s%s%s%7.1f%7.3f|%s%s%s%6.1f|%s%s%s|%7.3f%7.2f" % (
            k, seg, a["n_st"], a.get("n_slice", 0), a["stated_r"],
            g("sl_med"), g("sl_min"), g("sl_max"), g("sl_area"),
            a.get("nvert", float("nan")), a.get("minmax", float("nan")),
            g("ray_med"), g("ray_min"), g("ray_max"), a.get("ray_miss", float("nan")),
            g("b_med"), g("b_min"), g("b_max"),
            a["clr"], a["frac_inside"]))

print("")
print("PAIRED BIAS OF THE SLAB+BIN ESTIMATOR (ratio to the exact slice, same stations)")
for k in ["HOST"] + [x for x in sorted(res) if x != "HOST"]:
    a = res[k]["ALL"]
    if "paired_n" in a:
        print("  %-16s n=%3d  BIN/SL med %.3f (%+0.3f mm)  max %.3f  min %.3f"
              % (k, a["paired_n"], a["bin_over_slice_med"], a["bin_minus_slice_med"],
                 a["bin_over_slice_max"], a["bin_over_slice_min"]))

print("")
print("ARCLENGTH SCAN 10 mm: SLICEmed/r  SLICEarea/r  nvert  BINmed/r  r_mm  SLmed_mm")
coh = [k for k in prof if k != "HOST"]
for lo in np.arange(0, 240, 10.0):
    row = []
    for k in ["HOST"] + coh:
        p = prof[k]
        m = ((p["s"] >= lo) & (p["s"] < lo + 10) & np.isfinite(p["R_med_curve"])
             & (p["r"] > 1e-6))
        if m.sum() < 1:
            row.append(None)
            continue
        bm = p["b_med"][m]
        row.append((float(np.median(p["R_med_curve"][m] / p["r"][m])),
                    float(np.median(p["R_area"][m] / p["r"][m])),
                    float(np.median(p["nvert"][m])),
                    float(np.median(bm[np.isfinite(bm)] / p["r"][m][np.isfinite(bm)]))
                    if np.isfinite(bm).any() else float("nan"),
                    float(np.median(p["r"][m])),
                    float(np.median(p["R_med_curve"][m]))))
    h = row[0]
    cc = [x for x in row[1:] if x is not None]
    if h is None and not cc:
        continue
    hs_ = ("H %.3f/%.3f nv%5.1f bin%.3f r%.2f Rm%.2f" % h) if h else "H" + " " * 36
    if cc:
        med = [float(np.median([x[j] for x in cc])) for j in range(6)]
        cs = "| C %.3f/%.3f nv%5.1f bin%.3f r%.2f Rm%.2f n%2d" % tuple(med + [len(cc)])
    else:
        cs = ""
    print("%6.0f %s %s" % (lo, hs_, cs))

print("")
print("MACHINE_JSON")
print(json.dumps(res))
