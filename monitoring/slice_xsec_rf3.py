"""Decisive test of the CENTERLINE_OFF_AXIS rejection.

For every station: exact perpendicular cross-section polygon; measure
  off   = |area centroid of the lumen contour - centerline point|
  Req   = sqrt(A/pi)
  Rmin/Rmax about the CENTERLINE point   (what the guidewire feels)
  rmin/rmax about the CONTOUR CENTROID   (pure shape, off-axis removed)
If off/Req is large and rmin/rmax >> Rmin/Rmax, the deficit is off-axis, not erosion.
"""
import sys, os, math, glob
import numpy as np
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import pyvista as pv
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

COHORT = "/opt/eve_training/results_topbrain/anatomies"


def parse_segments(slc):
    lines = np.asarray(slc.lines)
    segs = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        idx = lines[i + 1:i + 1 + n]
        for k in range(n - 1):
            segs.append((int(idx[k]), int(idx[k + 1])))
        i += 1 + n
    return segs


def comps(segs):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in segs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    o = {}
    for a, b in segs:
        o.setdefault(find(a), []).append((a, b))
    return o


def order_loop(segs):
    adj = {}
    for a, b in segs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    clean = all(len(v) == 2 for v in adj.values())
    start = None
    for p, ns in adj.items():
        if len(ns) == 1:
            start = p
            break
    closed = start is None
    if start is None:
        start = next(iter(adj))
    order = [start]
    prev, cur = None, start
    while len(order) < len(adj) + 5:
        nxt = None
        for n in adj[cur]:
            if n != prev:
                nxt = n
                break
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order, closed, clean


def poly_area_centroid(p2):
    x, y = p2[:, 0], p2[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        return 0.0, p2.mean(axis=0)
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


def inside(p2):
    cnt = 0
    n = len(p2)
    for i in range(n):
        x1, y1 = p2[i]
        x2, y2 = p2[(i + 1) % n]
        if (y1 > 0) != (y2 > 0):
            if x1 + (0 - y1) * (x2 - x1) / (y2 - y1) > 0:
                cnt += 1
    return cnt % 2 == 1


def basis(t):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, t)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def tangent(cl, i, half=3):
    a, b = max(i - half, 0), min(i + half, len(cl) - 1)
    seg = cl[a:b + 1] - cl[a:b + 1].mean(axis=0)
    _, _, vt = np.linalg.svd(seg, full_matrices=False)
    t = vt[0]
    if np.dot(t, cl[b] - cl[a]) < 0:
        t = -t
    return t / np.linalg.norm(t)


def station(mesh, c, t):
    u, v = basis(t)
    slc = mesh.slice(normal=t, origin=c)
    if slc is None or slc.n_points == 0:
        return None
    pts = np.asarray(slc.points)
    segs = parse_segments(slc)
    if not segs:
        return None
    best = None
    for _, sl in comps(segs).items():
        order, closed, clean = order_loop(sl)
        if len(order) < 3 or not (closed and clean):
            continue
        d = pts[order] - c
        p2 = np.stack([d @ u, d @ v], axis=1)
        if not inside(p2):
            continue
        A, cen = poly_area_centroid(p2)
        if A <= 0:
            continue
        per = float(np.sum(np.linalg.norm(np.roll(p2, -1, axis=0) - p2, axis=1)))
        if per >= 40:
            continue
        R = np.linalg.norm(p2, axis=1)
        Rc = np.linalg.norm(p2 - cen, axis=1)
        cand = dict(Req=math.sqrt(A / math.pi), Rmin=float(R.min()), Rmax=float(R.max()),
                    rmin=float(Rc.min()), rmax=float(Rc.max()), off=float(np.linalg.norm(cen)),
                    perim=per, n_seg=len(sl))
        if best is None or cand["Req"] < best["Req"]:
            best = cand
    return best


def get_rcca(brs):
    for b in brs:
        if "rcca" in b.name.lower():
            return b
    raise KeyError([b.name for b in brs])


def run(name, mesh, cl, rad):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    rows = []
    for i in range(3, len(cl) - 3):
        r = station(mesh, cl[i], tangent(cl, i))
        if r is None:
            continue
        r.update(arc=float(s[i]), r=float(rad[i]))
        rows.append(r)
    return rows


def summ(name, rows):
    m = lambda g: float(np.median([g(x) for x in rows]))
    print("%-16s n=%3d  Req/r=%.3f  off/Req=%.3f off_mm=%.3f | about CENTERLINE Rmin/Rmax=%.3f "
          "| about CENTROID rmin/rmax=%.3f  rmin/Req=%.3f  Rmin/Req=%.3f"
          % (name, len(rows), m(lambda x: x["Req"] / x["r"]), m(lambda x: x["off"] / x["Req"]),
             m(lambda x: x["off"]), m(lambda x: x["Rmin"] / x["Rmax"]), m(lambda x: x["rmin"] / x["rmax"]),
             m(lambda x: x["rmin"] / x["Req"]), m(lambda x: x["Rmin"] / x["Req"])), flush=True)


dn = DualDeviceNav()
hm = pv.read(dn.vessel_tree.mesh_path).triangulate().clean()
hb = get_rcca(load_branches("/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"))
hrows = run("HOST", hm, np.asarray(hb.coordinates, float), np.asarray(hb.radii, float))
summ("HOST", hrows)

ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(COHORT, "topcow_mr_*")))
pooled = []
for aid in ids:
    d = os.path.join(COHORT, aid)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    b = get_rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    rows = run(aid, m, np.asarray(b.coordinates, float), np.asarray(b.radii, float))
    pooled += rows
    summ(aid, rows)

print("\nPOOLED COHORT BY ARCLENGTH  (off = centerline-to-lumen-centroid offset)", flush=True)
print(" arc   n   Req/r  off/Req off_mm  Rmin/r  Rmax/r  Rmin/Rmax  rmin/rmax  rmin/Req", flush=True)
for lo in range(0, 240, 10):
    b = [x for x in pooled if lo <= x["arc"] < lo + 10]
    if not b:
        continue
    m = lambda g: float(np.median([g(x) for x in b]))
    print(" %4d %4d  %.3f   %.3f  %.3f   %.3f   %.3f    %.3f      %.3f      %.3f"
          % (lo, len(b), m(lambda x: x["Req"] / x["r"]), m(lambda x: x["off"] / x["Req"]),
             m(lambda x: x["off"]), m(lambda x: x["Rmin"] / x["r"]), m(lambda x: x["Rmax"] / x["r"]),
             m(lambda x: x["Rmin"] / x["Rmax"]), m(lambda x: x["rmin"] / x["rmax"]),
             m(lambda x: x["rmin"] / x["Req"])), flush=True)

print("\nHOST BY ARCLENGTH", flush=True)
for lo in range(0, 240, 20):
    b = [x for x in hrows if lo <= x["arc"] < lo + 20]
    if not b:
        continue
    m = lambda g: float(np.median([g(x) for x in b]))
    print(" %4d %4d  Req/r=%.3f Req_mm=%.3f stated=%.3f off/Req=%.3f off_mm=%.3f Rmin/Rmax=%.3f rmin/rmax=%.3f"
          % (lo, len(b), m(lambda x: x["Req"] / x["r"]), m(lambda x: x["Req"]), m(lambda x: x["r"]),
             m(lambda x: x["off"] / x["Req"]), m(lambda x: x["off"]),
             m(lambda x: x["Rmin"] / x["Rmax"]), m(lambda x: x["rmin"] / x["rmax"])), flush=True)
print("DONE", flush=True)
