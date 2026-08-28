"""Robustness probe for the plane-slice cross-sections.

Tests the two ways slice_xsec_rf.py could be wrong:
  (A) tangent tilt  -> search over normal perturbations for the MINIMAL-AREA plane
  (B) loop contamination -> circularity 4*pi*A/P^2, n_loops, PCA semi-axes
and reports the true circumferential side count and the chord-error budget.
"""
import sys, os, json, math, glob
import numpy as np
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import pyvista as pv
from scipy.spatial import cKDTree
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

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in segs:
        union(a, b)
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


def shoelace(p2):
    x, y = p2[:, 0], p2[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


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
    v = np.cross(t, u)
    return u, v


def cut(mesh, c, t):
    u, v = basis(t)
    try:
        slc = mesh.slice(normal=t, origin=c)
    except Exception:
        return None
    if slc is None or slc.n_points == 0:
        return None
    pts = np.asarray(slc.points)
    segs = parse_segments(slc)
    if not segs:
        return None
    cs = comps(segs)
    best = None
    nloop = len(cs)
    for _, seglist in cs.items():
        order, closed, clean = order_loop(seglist)
        if len(order) < 3:
            continue
        d = pts[order] - c
        p2 = np.stack([d @ u, d @ v], axis=1)
        if not inside(p2):
            continue
        R = np.linalg.norm(p2, axis=1)
        per = float(np.sum(np.linalg.norm(np.roll(p2, -1, axis=0) - p2, axis=1)))
        A = shoelace(p2)
        if A <= 0:
            continue
        m = p2 - p2.mean(axis=0)
        w, _ = np.linalg.eigh(np.cov(m.T))
        w = np.sqrt(np.clip(w, 1e-12, None)) * 2.0
        cand = dict(n_seg=len(seglist), closed=bool(closed), clean=bool(clean),
                    Rmin=float(R.min()), Rmax=float(R.max()), area=float(A),
                    Req=float(math.sqrt(A / math.pi)), perim=per,
                    circ=float(4 * math.pi * A / (per * per)),
                    pca_aspect=float(w[0] / w[1]), n_loops=nloop)
        if best is None or cand["area"] < best["area"]:
            best = cand
    return best


def tangent(cl, i, half):
    n = len(cl)
    a, b = max(i - half, 0), min(i + half, n - 1)
    seg = cl[a:b + 1]
    m = seg - seg.mean(axis=0)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    t = vt[0]
    if np.dot(t, cl[b] - cl[a]) < 0:
        t = -t
    return t / np.linalg.norm(t)


def minarea_plane(mesh, c, t0, maxdeg=20, step=5):
    u, v = basis(t0)
    best = None
    bt = None
    for a in range(-maxdeg, maxdeg + 1, step):
        for b in range(-maxdeg, maxdeg + 1, step):
            t = t0 + math.tan(math.radians(a)) * u + math.tan(math.radians(b)) * v
            t /= np.linalg.norm(t)
            r = cut(mesh, c, t)
            if r is None:
                continue
            if best is None or r["area"] < best["area"]:
                best = r
                bt = (a, b)
    if best is not None:
        best["tilt"] = bt
    return best


def get_rcca(brs):
    for b in brs:
        if "rcca" in b.name.lower():
            return b
    raise KeyError([b.name for b in brs])


def run(name, mesh, cl, rad, step=5):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    rows = []
    for i in range(3, len(cl) - 3, step):
        c = cl[i]
        r = float(rad[i])
        e = {}
        for k, h in (("h1", 1), ("h3", 3), ("h6", 6)):
            e[k] = cut(mesh, c, tangent(cl, i, h))
        e["opt"] = minarea_plane(mesh, c, tangent(cl, i, 3))
        if e["h3"] is None or e["opt"] is None:
            continue
        rows.append(dict(i=i, arc=float(s[i]), r=r, **e))

    def med(key, field):
        vals = []
        for x in rows:
            o = x.get(key)
            if o is None:
                continue
            if not (o["closed"] and o["clean"] and o["perim"] < 40):
                continue
            vals.append(o[field] / x["r"] if field in ("Req", "Rmin", "Rmax") else o[field])
        return (float(np.median(vals)) if vals else float("nan")), len(vals)

    print("\n=== %s  stations=%d  stated_r_med=%.3f" % (name, len(rows), np.median([x["r"] for x in rows])), flush=True)
    for k in ("h1", "h3", "h6", "opt"):
        L = {f: med(k, f)[0] for f in ("Req", "Rmin", "Rmax", "circ", "pca_aspect", "n_seg", "n_loops", "perim")}
        n = med(k, "Req")[1]
        print("  %-4s n=%4d Req/r=%.3f Rmin/r=%.3f Rmax/r=%.3f circ=%.3f pca_asp=%.2f nseg=%.1f nloops=%.0f perim=%.2f"
              % (k, n, L["Req"], L["Rmin"], L["Rmax"], L["circ"], L["pca_aspect"], L["n_seg"], L["n_loops"], L["perim"]),
              flush=True)
    ok = [x for x in rows if x["h3"] and x["h3"]["closed"] and x["h3"]["clean"] and x["h3"]["perim"] < 40]
    if ok:
        fr = np.mean([x["h3"]["Rmax"] / x["r"] > 1.0 for x in ok])
        fro = np.mean([x["opt"]["Rmax"] / x["r"] > 1.0 for x in ok if x["opt"]])
        print("  frac stations Rmax/stated_r > 1 : h3 %.3f  opt %.3f" % (fr, fro), flush=True)
        ns = float(np.median([x["h3"]["n_seg"] for x in ok]))
        print("  chord-error ceiling for n=%.0f: inrad/circum=cos(pi/n)=%.4f  Req/Rcircum=%.4f"
              % (ns, math.cos(math.pi / ns), math.sqrt(ns * math.sin(2 * math.pi / ns) / (2 * math.pi))), flush=True)
        print("  arc  Req/r Rmin/r Rmax/r  in/circ  circ   n", flush=True)
        for lo in range(0, 240, 20):
            b = [x for x in ok if lo <= x["arc"] < lo + 20]
            if not b:
                continue
            f = lambda g: float(np.median([g(x) for x in b]))
            print("  %4d %.3f %.3f %.3f  %.3f  %.3f  %d"
                  % (lo, f(lambda x: x["h3"]["Req"] / x["r"]), f(lambda x: x["h3"]["Rmin"] / x["r"]),
                     f(lambda x: x["h3"]["Rmax"] / x["r"]),
                     f(lambda x: x["h3"]["Rmin"] / x["h3"]["Rmax"]), f(lambda x: x["h3"]["circ"]), len(b)), flush=True)
    return rows


dn = DualDeviceNav()
hm = pv.read(dn.vessel_tree.mesh_path).triangulate().clean()
hb = get_rcca(load_branches("/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"))
run("HOST", hm, np.asarray(hb.coordinates, float), np.asarray(hb.radii, float))

for aid in ["topcow_mr_001", "topcow_mr_013", "topcow_mr_017", "topcow_mr_025", "topcow_mr_005"]:
    d = os.path.join(COHORT, aid)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    b = get_rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    run(aid, m, np.asarray(b.coordinates, float), np.asarray(b.radii, float))

print("\n=== TRIANGLE EDGE LENGTHS ON THE RCCA TUBE ONLY ===", flush=True)
for aid in ["topcow_mr_001", "topcow_mr_017"]:
    d = os.path.join(COHORT, aid)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    b = get_rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    cl = np.asarray(b.coordinates, float)
    tr = cKDTree(cl)
    f = m.faces.reshape(-1, 4)[:, 1:]
    P = np.asarray(m.points)
    d0, _ = tr.query(P[f].mean(axis=1))
    sel = f[d0 < 6.0]
    e = np.concatenate([np.linalg.norm(P[sel[:, a]] - P[sel[:, bb]], axis=1) for a, bb in ((0, 1), (1, 2), (2, 0))])
    print("%s: tube tris=%d edge med=%.3f p10=%.3f p90=%.3f max=%.3f"
          % (aid, len(sel), np.median(e), np.percentile(e, 10), np.percentile(e, 90), e.max()), flush=True)
hf = hm.faces.reshape(-1, 4)[:, 1:]
HP = np.asarray(hm.points)
htr = cKDTree(np.asarray(hb.coordinates, float))
hd, _ = htr.query(HP[hf].mean(axis=1))
hsel = hf[hd < 6.0]
he = np.concatenate([np.linalg.norm(HP[hsel[:, a]] - HP[hsel[:, bb]], axis=1) for a, bb in ((0, 1), (1, 2), (2, 0))])
print("HOST: tube tris=%d edge med=%.3f p10=%.3f p90=%.3f max=%.3f"
      % (len(hsel), np.median(he), np.percentile(he, 10), np.percentile(he, 90), he.max()), flush=True)
print("DONE", flush=True)
