"""Radial cross-section profile of the RCCA lumen vs the DECLARED centerline radius.

Discriminates SURFACE_ERODED vs CENTERLINE_OFF_AXIS vs RADII_INFLATED.
Per station: thin perpendicular slab through the dense surface point cloud,
36 azimuth bins, MIN radius per bin = the wall in that direction.
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
SLAB = 1.0          # mm half-thickness of perpendicular slab
NBIN = 36           # 10 deg azimuth bins
SUBN = 24           # 24*24 = 576 barycentric samples per triangle
CROP = 25.0         # mm : keep triangles whose centroid is near the RCCA
BALL = 12.0         # mm : neighbourhood radius per station
SPLIT = 137.0       # mm arclength : carrier | subject graft


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
    pts = pts.reshape(-1, 3)
    return np.vstack([v[np.unique(f)], pts]), int(keep.sum())


def profile(mesh, coords, radii, label):
    surf, ntri = dense_surface(mesh, coords)
    tree = cKDTree(surf)
    n = len(coords)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    clr, _ = tree.query(coords)          # direct clearance, calibration vs prior audit

    Rmed = np.full(n, np.nan)
    Rmin = np.full(n, np.nan)
    Rmax = np.full(n, np.nan)
    Rp10 = np.full(n, np.nan)
    Ecc = np.full(n, np.nan)
    empty = np.zeros(n, int)

    for i in range(n):
        if i == 0:
            t = coords[1] - coords[0]
        elif i == n - 1:
            t = coords[-1] - coords[-2]
        else:
            t = coords[i + 1] - coords[i - 1]
        nt = np.linalg.norm(t)
        if nt < 1e-9:
            continue
        t = t / nt
        ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(t, ref)
        u = u / np.linalg.norm(u)
        w = np.cross(t, u)
        idx = tree.query_ball_point(coords[i], BALL)
        if not idx:
            continue
        p = surf[idx] - coords[i]
        ax = p @ t
        m = np.abs(ax) < SLAB
        if m.sum() < NBIN:
            continue
        q = p[m] - ax[m][:, None] * t[None]
        R = np.linalg.norm(q, axis=1)
        th = np.arctan2(q @ w, q @ u)
        b = ((th + np.pi) / (2 * np.pi) * NBIN).astype(int) % NBIN
        wallR = np.full(NBIN, np.inf)
        np.minimum.at(wallR, b, R)
        ok = np.isfinite(wallR)
        empty[i] = NBIN - int(ok.sum())
        if ok.sum() < NBIN // 2:
            continue
        rw = wallR[ok]
        bc = (np.arange(NBIN)[ok] + 0.5) / NBIN * 2 * np.pi - np.pi
        qv = rw[:, None] * np.stack([np.cos(bc), np.sin(bc)], axis=1)
        Rmed[i] = np.median(rw)
        Rmin[i] = rw.min()
        Rmax[i] = rw.max()
        Rp10[i] = np.percentile(rw, 10)
        Ecc[i] = np.linalg.norm(qv.mean(axis=0))

    return dict(label=label, s=s, r=radii, Rmed=Rmed, Rmin=Rmin, Rmax=Rmax,
                Rp10=Rp10, Ecc=Ecc, empty=empty, clr=clr, nsurf=len(surf), ntri=ntri)


def agg(p, lo, hi):
    s, r = p["s"], p["r"]
    m = (s >= lo) & (s < hi) & np.isfinite(p["Rmed"]) & (r > 1e-6)
    if m.sum() < 3:
        return None
    out = {"n": int(m.sum())}
    for k in ("Rmed", "Rmin", "Rmax", "Rp10", "Ecc"):
        out[k] = float(np.median(p[k][m] / r[m]))
    out["clr"] = float(np.median(p["clr"][m] / r[m]))
    out["stated_r"] = float(np.median(r[m]))
    out["Rmed_abs"] = float(np.median(p["Rmed"][m]))
    out["ecc_abs"] = float(np.median(p["Ecc"][m]))
    out["gap_abs"] = float(np.median(r[m] - p["Rmin"][m]))
    out["spread"] = float(np.median((p["Rmax"][m] - p["Rmin"][m]) / r[m]))
    return out


def rcca_branch(branches):
    return next(b for b in branches if "RCCA" in str(b.name).upper())


results = {}
profiles = {}

# ---------- HOST (control) : rule 1, mesh via vessel_tree.mesh_path ----------
host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca_branch(hvt.branches)
p = profile(hm, np.asarray(hb.coordinates, float), np.asarray(hb.radii, float), "HOST")
profiles["HOST"] = p
results["HOST"] = {"ALL": agg(p, -1, 1e9), "PROX": agg(p, -1, SPLIT),
                   "DIST": agg(p, SPLIT, 1e9), "ncells": int(hm.n_cells),
                   "nsurf": p["nsurf"], "L": float(p["s"][-1])}
print("done HOST", file=sys.stderr)

for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca_branch(load_branches(os.path.join(d, "Centrelines_comb")))
    p = profile(m, np.asarray(br.coordinates, float), np.asarray(br.radii, float), name)
    profiles[name] = p
    results[name] = {"ALL": agg(p, -1, 1e9), "PROX": agg(p, -1, SPLIT),
                     "DIST": agg(p, SPLIT, 1e9), "ncells": int(m.n_cells),
                     "nsurf": p["nsurf"], "L": float(p["s"][-1])}
    print("done " + name, file=sys.stderr)

EDGES = np.arange(0, 240, 10.0)
arc = {}
for k, p in profiles.items():
    s, r = p["s"], p["r"]
    rows = []
    for lo in EDGES:
        m = (s >= lo) & (s < lo + 10) & np.isfinite(p["Rmed"]) & (r > 1e-6)
        if m.sum() < 2:
            rows.append(None)
            continue
        rows.append(dict(lo=float(lo), n=int(m.sum()),
                         Rmed=float(np.median(p["Rmed"][m] / r[m])),
                         Rmin=float(np.median(p["Rmin"][m] / r[m])),
                         Rmax=float(np.median(p["Rmax"][m] / r[m])),
                         Ecc=float(np.median(p["Ecc"][m] / r[m])),
                         r=float(np.median(r[m])),
                         Rmed_abs=float(np.median(p["Rmed"][m]))))
    arc[k] = rows

print("=" * 120)
print("PER-ANATOMY RADIAL PROFILE (median over stations of the per-station ratio to stated r)")
print("=" * 120)
print("%-16s%-6s%5s%7s%8s%8s%8s%8s%8s%7s%9s%8s" % (
    "anatomy", "seg", "n", "r_med", "Rmed/r", "Rmin/r", "Rmax/r", "ecc/r",
    "clr/r", "sprd", "Rmed_mm", "ecc_mm"))
order = ["HOST"] + [x for x in sorted(results) if x != "HOST"]
for k in order:
    for seg in ("ALL", "PROX", "DIST"):
        a = results[k][seg]
        if a is None:
            print("%-16s%-6s   -- insufficient stations --" % (k, seg))
            continue
        print("%-16s%-6s%5d%7.2f%8.3f%8.3f%8.3f%8.3f%8.3f%7.3f%9.3f%8.3f" % (
            k, seg, a["n"], a["stated_r"], a["Rmed"], a["Rmin"], a["Rmax"],
            a["Ecc"], a["clr"], a["spread"], a["Rmed_abs"], a["ecc_abs"]))

print("")
print("=" * 120)
print("ARCLENGTH PROFILE (cohort column = median over the 25); SPLIT at %.0f mm" % SPLIT)
print("=" * 120)
print("%8s | %12s%9s%8s%7s | %12s%9s%9s%8s%7s%9s%4s" % (
    "arc(mm)", "HOST Rmed/r", "Rmin/r", "ecc/r", "r_mm",
    "COH Rmed/r", "Rmin/r", "Rmax/r", "ecc/r", "r_mm", "Rmed_mm", "nA"))
coh = [k for k in results if k != "HOST"]
for i, lo in enumerate(EDGES):
    h = arc["HOST"][i]
    cc = [arc[k][i] for k in coh if arc[k][i] is not None]
    if not cc and h is None:
        continue
    if h:
        hs = "%12.3f%9.3f%8.3f%7.2f" % (h["Rmed"], h["Rmin"], h["Ecc"], h["r"])
    else:
        hs = "%12s%9s%8s%7s" % ("-", "-", "-", "-")
    if cc:
        def f_(key):
            return float(np.median([c[key] for c in cc]))
        cs = "%12.3f%9.3f%9.3f%8.3f%7.2f%9.3f%4d" % (
            f_("Rmed"), f_("Rmin"), f_("Rmax"), f_("Ecc"), f_("r"), f_("Rmed_abs"), len(cc))
    else:
        cs = "%12s%9s%9s%8s%7s%9s%4d" % ("-", "-", "-", "-", "-", "-", 0)
    star = "  <<< SPLIT" if lo <= SPLIT < lo + 10 else ""
    print("%8.0f | %s | %s%s" % (lo, hs, cs, star))

print("")
print("EMPTY-BIN / SAMPLING DIAGNOSTICS")
for k in ["HOST"] + sorted(coh)[:3]:
    p = profiles[k]
    print("  %-16s surf pts %9d  tris kept %5d  mean empty bins/station %.2f  "
          "stations >6 empty %d/%d  finite %d" % (
              k, p["nsurf"], p["ntri"], p["empty"].mean(),
              int((p["empty"] > 6).sum()), len(p["empty"]),
              int(np.isfinite(p["Rmed"]).sum())))
print("  cohort mean empty bins/station: %.2f" % np.mean(
    [profiles[k]["empty"].mean() for k in coh]))

print("")
print("CONSTANCY TEST (is Rmed/r a clean scalar?)")
for k in ["HOST"] + sorted(coh)[:5]:
    p = profiles[k]
    m = np.isfinite(p["Rmed"]) & (p["r"] > 1e-6)
    v = p["Rmed"][m] / p["r"][m]
    print("  %-16s Rmed/r med %.3f  p10 %.3f  p90 %.3f  CV %.3f" % (
        k, np.median(v), np.percentile(v, 10), np.percentile(v, 90), v.std() / v.mean()))
cv = []
for k in coh:
    p = profiles[k]
    m = np.isfinite(p["Rmed"]) & (p["r"] > 1e-6)
    v = p["Rmed"][m] / p["r"][m]
    cv.append(v.std() / v.mean())
print("  cohort mean CV of Rmed/r: %.3f" % np.mean(cv))

print("")
print("MACHINE_JSON")
print(json.dumps({k: {s: results[k][s] for s in ("ALL", "PROX", "DIST")} for k in results}))
