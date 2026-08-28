"""Follow-up: is the declared radius the CIRCUMRADIUS of the tessellated tube?

Test 1: distance from the centerline to the mesh VERTICES in a perpendicular slab
        (the vertex ring). If ring_R == stated_r the generator puts vertices ON the
        declared circle and the facets cut chords inside it (polygonal deficit,
        declared radius = circumradius).
Test 2: how many vertices form that ring -> polygon side count n; predicted
        inradius/circumradius = cos(pi/n).
Test 3: fine (5 mm) arclength scan of stated radius, host-vs-cohort, to locate where
        the declared radii diverge, vs where the coordinates diverge (137 mm).
"""
import glob
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
SLAB = 1.0
SPLIT = 137.0


def rcca(branches):
    return next(b for b in branches if "RCCA" in str(b.name).upper())


def vertex_ring(mesh, coords, radii):
    v = np.asarray(mesh.points, float)
    tree = cKDTree(v)
    n = len(coords)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    ringR = np.full(n, np.nan)
    ringRmin = np.full(n, np.nan)
    ringRmax = np.full(n, np.nan)
    nvert = np.zeros(n, int)
    ecc = np.full(n, np.nan)
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
        idx = tree.query_ball_point(coords[i], 15.0)
        if len(idx) < 3:
            continue
        p = v[idx] - coords[i]
        ax = p @ t
        m = np.abs(ax) < SLAB
        if m.sum() < 3:
            continue
        q = p[m] - ax[m][:, None] * t[None]
        R = np.linalg.norm(q, axis=1)
        # keep only the ring belonging to THIS vessel: within 2.5x the median
        keep = R < 2.5 * np.median(R)
        R = R[keep]
        q = q[keep]
        if len(R) < 3:
            continue
        ringR[i] = np.median(R)
        ringRmin[i] = R.min()
        ringRmax[i] = R.max()
        nvert[i] = len(R)
        ecc[i] = np.linalg.norm(q.mean(axis=0))
    return s, ringR, ringRmin, ringRmax, nvert, ecc


def line(name, s, ringR, ringRmin, ringRmax, nvert, ecc, r):
    out = []
    for tag, lo, hi in (("ALL", -1, 1e9), ("PROX", -1, SPLIT), ("DIST", SPLIT, 1e9)):
        m = (s >= lo) & (s < hi) & np.isfinite(ringR) & (r > 1e-6)
        if m.sum() < 3:
            out.append(None)
            continue
        out.append(dict(n=int(m.sum()),
                        ring=float(np.median(ringR[m] / r[m])),
                        ringmin=float(np.median(ringRmin[m] / r[m])),
                        ringmax=float(np.median(ringRmax[m] / r[m])),
                        nv=float(np.median(nvert[m])),
                        ecc=float(np.median(ecc[m] / r[m])),
                        ring_abs=float(np.median(ringR[m])),
                        r_abs=float(np.median(r[m]))))
    return out


host = DualDeviceNav()
hvt = host.vessel_tree
hm = pv.read(hvt.mesh_path).triangulate().clean()
hb = rcca(hvt.branches)
hc = np.asarray(hb.coordinates, float)
hr = np.asarray(hb.radii, float)
hs, hR, hRmn, hRmx, hnv, hec = vertex_ring(hm, hc, hr)

rows = {"HOST": line("HOST", hs, hR, hRmn, hRmx, hnv, hec, hr)}
store = {"HOST": (hs, hR, hr)}

for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    br = rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    c = np.asarray(br.coordinates, float)
    r = np.asarray(br.radii, float)
    s, R, Rmn, Rmx, nv, ec = vertex_ring(m, c, r)
    rows[name] = line(name, s, R, Rmn, Rmx, nv, ec, r)
    store[name] = (s, R, r)
    print("done " + name, file=sys.stderr)

print("=" * 108)
print("VERTEX-RING TEST: is the DECLARED radius the circumradius of the polygonal tube?")
print("=" * 108)
print("%-16s%-6s%5s%9s%10s%10s%10s%8s%9s%8s" % (
    "anatomy", "seg", "n", "ring/r", "ringmin/r", "ringmax/r", "n_vert", "ecc/r",
    "ring_mm", "r_mm"))
for k in ["HOST"] + [x for x in sorted(rows) if x != "HOST"]:
    for j, tag in enumerate(("ALL", "PROX", "DIST")):
        a = rows[k][j]
        if a is None:
            print("%-16s%-6s  --" % (k, tag))
            continue
        print("%-16s%-6s%5d%9.3f%10.3f%10.3f%10.1f%8.3f%9.3f%8.3f" % (
            k, tag, a["n"], a["ring"], a["ringmin"], a["ringmax"], a["nv"],
            a["ecc"], a["ring_abs"], a["r_abs"]))

print("")
print("=" * 108)
print("FINE 5 mm SCAN: stated radius and vertex-ring radius, host vs cohort median")
print("=" * 108)
print("%8s | %9s%10s%9s | %9s%10s%9s%5s" % (
    "arc(mm)", "HOST r", "HOST ring", "ring/r", "COH r", "COH ring", "ring/r", "nA"))
coh = [k for k in store if k != "HOST"]
for lo in np.arange(0, 235, 5.0):
    def stat(k):
        s, R, r = store[k]
        m = (s >= lo) & (s < lo + 5) & np.isfinite(R) & (r > 1e-6)
        if m.sum() < 1:
            return None
        return float(np.median(r[m])), float(np.median(R[m])), float(np.median(R[m] / r[m]))
    h = stat("HOST")
    cc = [stat(k) for k in coh]
    cc = [x for x in cc if x is not None]
    if h is None and not cc:
        continue
    hs_ = "%9.3f%10.3f%9.3f" % h if h else "%9s%10s%9s" % ("-", "-", "-")
    if cc:
        cs = "%9.3f%10.3f%9.3f%5d" % (np.median([x[0] for x in cc]),
                                      np.median([x[1] for x in cc]),
                                      np.median([x[2] for x in cc]), len(cc))
    else:
        cs = "%9s%10s%9s%5d" % ("-", "-", "-", 0)
    star = "  <<< coord SPLIT" if lo <= SPLIT < lo + 5 else ""
    print("%8.0f | %s | %s%s" % (lo, hs_, cs, star))

print("")
print("DECLARED-RADIUS AGREEMENT host vs cohort (resampled to common arclength)")
L = min(min(store[k][0][-1] for k in coh), hs[-1])
g = np.linspace(0, L, 400)
hri = np.interp(g, hs, hr)
C = np.stack([np.interp(g, store[k][0], store[k][2]) for k in coh])
dif = np.abs(C - hri[None])
md = np.median(dif, axis=0)
first = np.argmax(md > 0.05)
print("  |cohort r - host r| median over 25: max in 0-90mm = %.4f mm" %
      md[g < 90].max())
print("  first arclength where median |dr| > 0.05 mm: %.1f mm" % g[first])
for f in (0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0):
    i = min(int(f * 399), 399)
    print("    arc %6.1f mm  host r %.3f  cohort r med %.3f  |dr| med %.4f  spread(25) %.4f"
          % (g[i], hri[i], np.median(C[:, i]), md[i], C[:, i].std()))
