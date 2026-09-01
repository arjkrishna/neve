"""CHECKS 7 + 12 on the 216-anatomy three-source carotid set.  READ-ONLY.

7  reachability ceiling: fraction of the CenterlineRandom admissible pool that is
   NOT distal to a mid-vessel blockage, per anatomy and pool-weighted cohort.
12 seam positions (measured, not nominal) + shared-course fraction of the pool.

Traps guarded (HANDOFF 11): we only compare a centerline to its OWN mesh, and
centerline-to-centerline, so no mesh/branch frame mixing; sign is fixed by
majority AND cross-checked against vtkSelectEnclosedPoints on control
anatomies; the centerline is densified to 0.25 mm before any minimum.
"""
import sys, os, json, math
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
HOST = "/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA_NAME = "Centerline curve - RCCA.mrk"
MIN_ARC = 40.0
TERM_MM = 8.0
STEP = 0.25
THR = {"wire_0.18": 0.18, "sofa_0.30": 0.30, "cath_0.35": 0.35,
       "sofa_wire_0.48": 0.48, "sofa_cath_0.65": 0.65}
CONTROL_TAG = "topcow_mr_001"


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def densify(C, step=STEP):
    S = arclen(C)
    g = np.arange(0.0, S[-1], step)
    g = np.append(g, S[-1])
    P = np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)
    return P, g


def signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def enclosed(mesh, pts):
    ps = pv.PolyData(np.asarray(pts, float))
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(ps)
    sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6)
    sel.CheckSurfaceOff()
    sel.Update()
    o = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))]) > 0.5


def runs(mask, g, d):
    r = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            r.append(dict(s0=float(g[i]), s1=float(g[j]),
                          length=float(g[j] - g[i]),
                          min_d=float(d[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    return r


def dist_to_polyline(P, Q):
    """Exact point-to-POLYLINE (segment) distance, never nearest-vertex."""
    A = Q[:-1]
    B = Q[1:]
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    out = np.empty(len(P))
    for k in range(len(P)):
        p = P[k]
        t = np.einsum("ij,ij->i", p - A, AB) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = A + t[:, None] * AB
        out[k] = float(np.min(np.linalg.norm(proj - p, axis=1)))
    return out


def first_sustained(g, d, thr):
    """First arclength beyond which d never returns to <= thr.  None if never."""
    over = d > thr
    if not over.any():
        return None
    idx = np.where(~over)[0]
    last_ok = int(idx[-1]) if len(idx) else -1
    if last_ok == len(d) - 1:
        return None
    return float(g[last_ok + 1])


names = sorted(d for d in os.listdir(ROOT)
               if os.path.isdir(os.path.join(ROOT, d, "Centrelines_comb")))
print("anatomies: %d" % len(names), flush=True)

hb = load_branches(HOST)
host_rcca = next(b for b in hb if "RCCA" in str(b.name).upper())
HC = np.asarray(host_rcca.coordinates, float)
HS = arclen(HC)
print("host RCCA: %d stations, L=%.2f mm" % (len(HC), HS[-1]), flush=True)

REC = {}
CURVES = {}
n_ctrl = 0
for n_i, name in enumerate(names):
    d0 = os.path.join(ROOT, name)
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    bynm = dict((str(b.name), b) for b in brs)
    rc = bynm.get(RCCA_NAME)
    if rc is None:
        rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    S = arclen(C)
    L = float(S[-1])
    P, g = densify(C)
    CURVES[name] = (C, S, L)

    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    sd = signed(mesh, P)
    flip = bool((sd > 0).mean() < 0.5)
    if flip:
        sd = -sd
    clear = sd

    ctrl = None
    if CONTROL_TAG in name and n_ctrl < 4:
        n_ctrl += 1
        enc = enclosed(mesh, P)
        ctrl = dict(agree_pct=float(100.0 * ((clear > 0) == enc).mean()),
                    frac_enclosed=float(enc.mean()),
                    frac_sd_pos=float((sd > 0).mean()))
        print("  [CONTROL] %s  sd>0 %.3f  enclosed %.3f  agree %.2f%%  flipped=%s"
              % (name, ctrl["frac_sd_pos"], ctrl["frac_enclosed"],
                 ctrl["agree_pct"], flip), flush=True)

    rec = dict(name=name,
               lower=name.split("__")[0], siphon=name.split("__")[1],
               L=L, nstat=int(len(C)), npts=int(len(P)), flipped=flip,
               frac_pos=float((clear > 0).mean()),
               min_clear=float(clear.min()), med_clear=float(np.median(clear)),
               n_outside=int((clear <= 0).sum()), ctrl=ctrl)

    # ---- admissible pool: exact CenterlineRandom reproduction --------------
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    keep = cum >= MIN_ARC
    pts = C[keep]
    sp = cum[keep]
    inex = np.zeros(len(pts), bool)
    for b in brs:
        if b is rc:
            continue
        rr_ = getattr(b, "radii", None)
        if rr_ is not None and len(rr_):
            inex |= np.asarray(b.in_branch(pts), bool)
    pool_s = sp[~inex]
    rec["pool_n"] = int(len(pool_s))
    rec["n_ge40"] = int(keep.sum())
    rec["n_dropped_excl"] = int(inex.sum())
    rec["pool_s"] = [float(x) for x in pool_s]

    # ---- blockages + ceiling ----------------------------------------------
    for k in THR:
        t = THR[k]
        rr = runs(clear < t, g, clear)
        for r in rr:
            r["terminal"] = bool(r["s1"] >= L - TERM_MM)
        mid = [r for r in rr if not r["terminal"]]
        sb = min(r["s0"] for r in mid) if mid else None
        nblk = int((pool_s >= sb).sum()) if sb is not None else 0
        rec[k] = dict(n_runs=len(rr), n_mid=len(mid), s_block=sb,
                      n_pool_blocked=nblk,
                      ceiling=(float(1.0 - nblk / float(len(pool_s)))
                               if len(pool_s) else None),
                      mid_runs=[dict((kk, r[kk]) for kk in
                                     ("s0", "s1", "length", "min_d"))
                                for r in mid][:5])

    # ---- seam 1: departure from the HOST RCCA -----------------------------
    dh = dist_to_polyline(P, HC)
    rec["seam1_1mm"] = first_sustained(g, dh, 1.0)
    rec["seam1_05mm"] = first_sustained(g, dh, 0.5)
    rec["seam1_01mm"] = first_sustained(g, dh, 0.1)
    REC[name] = rec
    if n_i % 20 == 0:
        print("[%3d] %-46s L=%6.1f pool=%3d minclr=%.3f nout=%2d seam1=%s"
              % (n_i, name, L, rec["pool_n"], rec["min_clear"],
                 rec["n_outside"], rec["seam1_1mm"]), flush=True)

# ---- seam 2: departure from SIBLINGS sharing the same lower donor ---------
groups = {}
for name in REC:
    groups.setdefault(REC[name]["lower"], []).append(name)
print("\nlower-donor groups: %d  sizes %s"
      % (len(groups), sorted(set(len(v) for v in groups.values()))), flush=True)

for lo in groups:
    mem = groups[lo]
    if len(mem) < 2:
        for m in mem:
            REC[m]["seam2_1mm"] = None
            REC[m]["seam2_med"] = None
            REC[m]["seam2_n_sib"] = 0
        continue
    for m in mem:
        C, S, L = CURVES[m]
        P, g = densify(C)
        vals = []
        for o in mem:
            if o == m:
                continue
            s2 = first_sustained(g, dist_to_polyline(P, CURVES[o][0]), 1.0)
            vals.append(s2 if s2 is not None else L)
        REC[m]["seam2_1mm"] = float(np.min(vals))
        REC[m]["seam2_med"] = float(np.median(vals))
        REC[m]["seam2_n_sib"] = len(vals)

print("\n@@@JSON@@@")
print(json.dumps(REC))
