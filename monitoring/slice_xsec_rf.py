"""Refutation probe: exact planar cross-sections of the RCCA lumen.

Independent of the azimuth-bin MIN statistic used by the diagnosis.
For each centerline station we cut the mesh with the exact perpendicular plane,
take the closed contour loop that ENCIRCLES the centerline point, and report:
  n_seg      = number of triangles the plane cuts  -> TRUE side count of the cross-section
  R_eq       = sqrt(area/pi)                       -> unbiased calibre
  R_min/R_max= inradius / circumradius of that actual polygon
  perim
No vertex statistics, no slab, no neighbour contamination (loop must contain origin).
"""
import sys, os, json, math, glob
import numpy as np
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import pyvista as pv
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

COHORT = "/opt/eve_training/results_topbrain/anatomies"


# ---------------- loop extraction ----------------
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
    out = {}
    for a, b in segs:
        out.setdefault(find(a), []).append((a, b))
    return out


def order_loop(segs):
    adj = {}
    for a, b in segs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    degs = [len(v) for v in adj.values()]
    clean = all(d == 2 for d in degs)
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
    guard = len(adj) + 5
    while len(order) < guard:
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
    # crossing number for origin
    cnt = 0
    n = len(p2)
    for i in range(n):
        x1, y1 = p2[i]
        x2, y2 = p2[(i + 1) % n]
        if (y1 > 0) != (y2 > 0):
            xint = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            if xint > 0:
                cnt += 1
    return cnt % 2 == 1


def frame(c, i):
    n = len(c)
    a = c[max(i - 1, 0)]
    b = c[min(i + 1, n - 1)]
    t = b - a
    t /= (np.linalg.norm(t) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, t)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return t, u, v


def profile(mesh, cl, radii, step=1, tag=""):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    rows = []
    for i in range(1, len(cl) - 1, step):
        c = cl[i]
        t, u, v = frame(cl, i)
        try:
            slc = mesh.slice(normal=t, origin=c)
        except Exception:
            continue
        if slc is None or slc.n_points == 0:
            continue
        pts = np.asarray(slc.points)
        segs = parse_segments(slc)
        if not segs:
            continue
        best = None
        for _, cs in comps(segs).items():
            order, closed, clean = order_loop(cs)
            if len(order) < 3:
                continue
            P = pts[order]
            d = P - c
            p2 = np.stack([d @ u, d @ v], axis=1)
            if not inside(p2):
                continue
            R = np.linalg.norm(p2, axis=1)
            per = float(np.sum(np.linalg.norm(np.roll(p2, -1, axis=0) - p2, axis=1)))
            A = shoelace(p2)
            cand = dict(n_seg=len(cs), n_pt=len(order), closed=bool(closed), clean=bool(clean),
                        Rmin=float(R.min()), Rmax=float(R.max()), Rmed=float(np.median(R)),
                        Req=float(math.sqrt(A / math.pi)), perim=per, area=float(A))
            if best is None or cand["perim"] < best["perim"]:
                best = cand
        if best is None:
            continue
        # exact nearest surface point (not vertex) for the clearance calibration
        try:
            cid, cp = mesh.find_closest_cell(c, return_closest_point=True)
            clear = float(np.linalg.norm(cp - c))
        except Exception:
            clear = float("nan")
        best.update(i=i, arc=float(s[i]), r=float(radii[i]), clearance=clear)
        rows.append(best)
    return rows


def agg(rows, lo=None, hi=None):
    R = [x for x in rows if x["closed"] and x["clean"] and x["perim"] < 40.0]
    if lo is not None:
        R = [x for x in R if x["arc"] >= lo]
    if hi is not None:
        R = [x for x in R if x["arc"] < hi]
    if not R:
        return None
    g = lambda k: float(np.median([x[k] for x in R]))
    gr = lambda k: float(np.median([x[k] / x["r"] for x in R]))
    return dict(n=len(R), r=g("r"), Req=g("Req"), Rmin=g("Rmin"), Rmax=g("Rmax"),
                Req_r=gr("Req"), Rmin_r=gr("Rmin"), Rmax_r=gr("Rmax"),
                clear_r=gr("clearance"),
                inrad_circ=float(np.median([x["Rmin"] / x["Rmax"] for x in R])),
                n_seg=g("n_seg"), perim=g("perim"),
                chord=float(np.median([x["perim"] / x["n_seg"] for x in R])))


def get_rcca(brs):
    for b in brs:
        if b.name.lower() in ("rcca", "r_cca"):
            return b
    for b in brs:
        if "rcca" in b.name.lower():
            return b
    raise KeyError([b.name for b in brs])


def load_mesh(p):
    m = pv.read(p).triangulate().clean()
    return m


out = {}

# ---- HOST ----
dn = DualDeviceNav()
hmesh = load_mesh(dn.vessel_tree.mesh_path)
hbr = get_rcca(load_branches("/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"))
hcl, hrad = np.asarray(hbr.coordinates, float), np.asarray(hbr.radii, float)
print("HOST mesh cells", hmesh.n_cells, "cl", hcl.shape, flush=True)
hrows = profile(hmesh, hcl, hrad, step=1)
out["HOST"] = dict(ALL=agg(hrows), PROX=agg(hrows, hi=137), DIST=agg(hrows, lo=137))
print("HOST", json.dumps(out["HOST"], indent=1), flush=True)

ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(COHORT, "topcow_mr_*")))
print("n anatomies", len(ids), flush=True)

allrows = {}
for aid in ids:
    d = os.path.join(COHORT, aid)
    m = load_mesh(os.path.join(d, "vessel_architecture_collision.obj"))
    b = get_rcca(load_branches(os.path.join(d, "Centrelines_comb")))
    cl, rad = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
    rows = profile(m, cl, rad, step=1)
    allrows[aid] = rows
    out[aid] = dict(ALL=agg(rows), PROX=agg(rows, hi=137), DIST=agg(rows, lo=137),
                    cells=m.n_cells, nstat=len(rows))
    a = out[aid]["ALL"]
    print(f"{aid}: n={a['n']} r={a['r']:.3f} Req/r={a['Req_r']:.3f} Rmin/r={a['Rmin_r']:.3f} "
          f"Rmax/r={a['Rmax_r']:.3f} clear/r={a['clear_r']:.3f} nseg={a['n_seg']:.1f} "
          f"chord={a['chord']:.2f} in/circ={a['inrad_circ']:.3f}", flush=True)

# ---- arclength profile, cohort pooled + host ----
print("\nARC BINS (10mm): host_r host_Req host_Req/r | coh_r coh_Req coh_Req/r coh_Rmax/r coh_nseg", flush=True)
pooled = [x for rows in allrows.values() for x in rows if x["closed"] and x["clean"] and x["perim"] < 40]
hclean = [x for x in hrows if x["closed"] and x["clean"] and x["perim"] < 40]
bins = []
for lo in range(0, 240, 10):
    hb = [x for x in hclean if lo <= x["arc"] < lo + 10]
    cb = [x for x in pooled if lo <= x["arc"] < lo + 10]
    if not cb:
        continue
    row = dict(arc=lo,
               h_r=float(np.median([x["r"] for x in hb])) if hb else None,
               h_Req=float(np.median([x["Req"] for x in hb])) if hb else None,
               h_Req_r=float(np.median([x["Req"] / x["r"] for x in hb])) if hb else None,
               h_Rmax_r=float(np.median([x["Rmax"] / x["r"] for x in hb])) if hb else None,
               h_nseg=float(np.median([x["n_seg"] for x in hb])) if hb else None,
               c_r=float(np.median([x["r"] for x in cb])),
               c_Req=float(np.median([x["Req"] for x in cb])),
               c_Req_r=float(np.median([x["Req"] / x["r"] for x in cb])),
               c_Rmax_r=float(np.median([x["Rmax"] / x["r"] for x in cb])),
               c_Rmin_r=float(np.median([x["Rmin"] / x["r"] for x in cb])),
               c_nseg=float(np.median([x["n_seg"] for x in cb])),
               c_chord=float(np.median([x["perim"] / x["n_seg"] for x in cb])),
               n=len(cb))
    bins.append(row)
    f = lambda z: ("  na " if z is None else f"{z:5.3f}")
    print(f"{lo:4d} | h {f(row['h_r'])} {f(row['h_Req'])} {f(row['h_Req_r'])} {f(row['h_Rmax_r'])} "
          f"nseg {row['h_nseg']} | c {f(row['c_r'])} {f(row['c_Req'])} {f(row['c_Req_r'])} "
          f"Rmax/r {f(row['c_Rmax_r'])} Rmin/r {f(row['c_Rmin_r'])} nseg {row['c_nseg']:.1f} "
          f"chord {row['c_chord']:.2f} n={row['n']}", flush=True)
out["bins"] = bins

# ---- distal stated radius: constant fill? ----
print("\nSTATED RADIUS STRUCTURE (distinct values, distal plateau test)", flush=True)
for aid in ids[:6]:
    b = get_rcca(load_branches(os.path.join(COHORT, aid, "Centrelines_comb")))
    r = np.asarray(b.radii, float)
    cl = np.asarray(b.coordinates, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    dm = s >= 137
    print(f"{aid}: n={len(r)} distal n={dm.sum()} distal uniq={len(np.unique(np.round(r[dm],6)))} "
          f"distal min/med/max={r[dm].min():.4f}/{np.median(r[dm]):.4f}/{r[dm].max():.4f} "
          f"spacing med={np.median(np.diff(s)):.4f} prox_sp={np.median(np.diff(s)[s[1:]<137]):.4f} "
          f"dist_sp={np.median(np.diff(s)[s[1:]>=137]):.4f}", flush=True)
hs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(hcl, axis=0), axis=1))])
print(f"HOST: n={len(hrad)} spacing med={np.median(np.diff(hs)):.4f} "
      f"r min/med/max={hrad.min():.4f}/{np.median(hrad):.4f}/{hrad.max():.4f}", flush=True)

with open("/tmp/slice_xsec_rf.json", "w") as f:
    json.dump(out, f)
print("\nDONE", flush=True)
