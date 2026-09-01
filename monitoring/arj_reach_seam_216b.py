"""PASS 2.  Fixes two things pass 1 could not settle:

A) SEAMS by ARCLENGTH-MATCHED distance, not nearest-point-on-polyline.
   Pass 1 used nearest-point, which reports "still on the host" whenever the
   composite happens to run back near some other part of the host curve --
   that produced a nonsense seam1 of 226 mm. Before a seam the two curves are
   the SAME curve from the SAME origin, so matched-arclength distance is ~0
   and the first departure is unambiguous.

B) Whether the proximal blockage cluster at s ~ 75-85 mm is a real stenosis or
   the CAROTID BIFURCATION CARINA. vtkImplicitPolyDataDistance returns distance
   to the NEAREST surface; at a flow divider that surface is the carina beside
   the device, not a wall around it. Test: (i) locate the RCCA/RECA bifurcation
   on RCCA arclength, (ii) RE-CENTRE the station in the plane normal to the
   centerline (maximal inscribed sphere), capped at 2.0 mm of offset and
   required to stay nearer the RCCA than the RECA. A dip that vanishes on
   re-centring is a misplaced centerline at the apex, not a blocked vessel.

READ-ONLY.  Control anatomies re-run for the sign convention.
"""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
HOST = "/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA_NAME = "Centerline curve - RCCA.mrk"
RECA_NAME = "Centerline curve - RECA.mrk"
MIN_ARC = 40.0
TERM_MM = 8.0
STEP = 0.25
THR = {"wire_0.18": 0.18, "sofa_0.30": 0.30, "cath_0.35": 0.35,
       "sofa_wire_0.48": 0.48, "sofa_cath_0.65": 0.65}
MAX_OFFSET = 2.0


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def densify(C, step=STEP):
    S = arclen(C)
    g = np.arange(0.0, S[-1], step)
    g = np.append(g, S[-1])
    P = np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)
    return P, g


def resample_at(C, S, g):
    return np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)


def runs(mask, g, d):
    out = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(dict(i0=i, i1=j, s0=float(g[i]), s1=float(g[j]),
                            length=float(g[j] - g[i]),
                            min_d=float(d[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    return out


def first_cross(g, d, thr):
    w = np.where(d > thr)[0]
    return float(g[w[0]]) if len(w) else None


def dist_to_polyline(P, Q):
    A = Q[:-1]
    B = Q[1:]
    AB = B - A
    den = np.einsum("ij,ij->i", AB, AB)
    den = np.where(den < 1e-12, 1e-12, den)
    out = np.empty(len(P))
    for k in range(len(P)):
        p = P[k]
        t = np.clip(np.einsum("ij,ij->i", p - A, AB) / den, 0.0, 1.0)
        out[k] = float(np.min(np.linalg.norm(A + t[:, None] * AB - p, axis=1)))
    return out


def frame(P, i):
    a = max(0, i - 2)
    b = min(len(P) - 1, i + 2)
    t = P[b] - P[a]
    n = np.linalg.norm(t)
    t = t / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, t)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return u, v


def recenter(imp, P, i, rcca_line, reca_line, cap=MAX_OFFSET):
    """Max inscribed-sphere radius in the plane normal to the centerline.
    Capped offset; recentred point must stay nearer RCCA than RECA."""
    p0 = P[i]
    u, v = frame(P, i)
    best = imp.EvaluateFunction(p0)
    boff = 0.0
    bp = p0
    rad = cap
    step = cap / 5.0
    ctr = p0
    for _ in range(3):
        cand = [ctr]
        for r in np.arange(step, rad + 1e-9, step):
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                cand.append(ctr + r * (np.cos(a) * u + np.sin(a) * v))
        for c in cand:
            off = float(np.linalg.norm(c - p0))
            if off > cap:
                continue
            d = imp.EvaluateFunction(c)
            if d > best:
                best, boff, bp = d, off, c
        ctr = bp
        rad = step
        step = step / 4.0
    if reca_line is not None:
        d_r = dist_to_polyline(bp[None, :], rcca_line)[0]
        d_e = dist_to_polyline(bp[None, :], reca_line)[0]
        if d_e < d_r:
            return float(imp.EvaluateFunction(p0)), 0.0, True
    return float(best), float(boff), False


names = sorted(d for d in os.listdir(ROOT)
               if os.path.isdir(os.path.join(ROOT, d, "Centrelines_comb")))
hb = load_branches(HOST)
HC = np.asarray(next(b for b in hb if "RCCA" in str(b.name).upper()).coordinates, float)
HS = arclen(HC)
print("host RCCA L=%.2f" % HS[-1], flush=True)

OUT = {}
CUR = {}
for n_i, name in enumerate(names):
    d0 = os.path.join(ROOT, name)
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    bynm = dict((str(b.name), b) for b in brs)
    rc = bynm[RCCA_NAME]
    re_ = bynm.get(RECA_NAME)
    C = np.asarray(rc.coordinates, float)
    S = arclen(C)
    L = float(S[-1])
    P, g = densify(C)
    CUR[name] = (C, S, L)
    rec = dict(name=name, lower=name.split("__")[0], siphon=name.split("__")[1], L=L)

    # ---- A: seam1 by matched arclength -----------------------------------
    HP = resample_at(HC, HS, np.clip(g, 0, HS[-1]))
    dm = np.linalg.norm(P - HP, axis=1)
    rec["seam1_m_1mm"] = first_cross(g, dm, 1.0)
    rec["seam1_m_01mm"] = first_cross(g, dm, 0.1)
    rec["seam1_pre_max"] = float(dm[g < min(rec["seam1_m_01mm"] or 5.0, 5.0)].max())

    # ---- B: bifurcation location ------------------------------------------
    s_bif = None
    reca_line = None
    if re_ is not None:
        RE = np.asarray(re_.coordinates, float)
        reca_line = RE
        dd = np.linalg.norm(P[:, None, :] - RE[None, :, :], axis=2)
        s_bif = float(g[int(np.argmin(dd.min(axis=1)))])
        rec["reca_L"] = float(arclen(RE)[-1])
        rec["reca_min_gap"] = float(dd.min())
    rec["s_bif"] = s_bif

    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    clear = np.array([imp.EvaluateFunction(p) for p in P])
    if (clear > 0).mean() < 0.5:
        raise SystemExit("SIGN CONTROL FAILED on %s" % name)

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
    rec["pool_s"] = [float(x) for x in pool_s]

    # declared radius interpolated onto g
    rad = np.asarray(rc.radii, float)
    rec_r = np.interp(g, S, rad)

    for k in THR:
        t = THR[k]
        rr = runs(clear < t, g, clear)
        mid = [r for r in rr if r["s1"] < L - TERM_MM]
        adj = []
        for r in mid:
            i0 = max(0, r["i0"] - 4)
            i1 = min(len(P) - 1, r["i1"] + 4)
            vals = []
            offs = []
            rejects = 0
            for i in range(i0, i1 + 1):
                d, o, rej = recenter(imp, P, i, C, reca_line)
                vals.append(d)
                offs.append(o)
                rejects += int(rej)
            rc_min = float(np.min(vals))
            adj.append(dict(s0=r["s0"], s1=r["s1"], length=r["length"],
                            min_d=r["min_d"], recentred_min=rc_min,
                            max_off=float(np.max(offs)), reca_rejects=rejects,
                            decl_r=float(rec_r[r["i0"]:r["i1"] + 1].min()),
                            d_from_bif=(None if s_bif is None else float(r["s0"] - s_bif)),
                            still_blocked=bool(rc_min < t)))
        real = [a for a in adj if a["still_blocked"]]
        sb_raw = min(a["s0"] for a in adj) if adj else None
        sb_real = min(a["s0"] for a in real) if real else None
        rec[k] = dict(
            n_mid=len(adj), n_mid_real=len(real),
            s_block_raw=sb_raw, s_block_real=sb_real,
            nblk_raw=int((pool_s >= sb_raw).sum()) if sb_raw is not None else 0,
            nblk_real=int((pool_s >= sb_real).sum()) if sb_real is not None else 0,
            runs=adj[:6])
    OUT[name] = rec
    if n_i % 15 == 0:
        c = rec["cath_0.35"]
        print("[%3d] %-46s bif=%s seam1m=%s pre=%.4f  midraw=%d midreal=%d"
              % (n_i, name, ("%.1f" % s_bif) if s_bif else "NA",
                 ("%.2f" % rec["seam1_m_1mm"]) if rec["seam1_m_1mm"] else "NA",
                 rec["seam1_pre_max"], c["n_mid"], c["n_mid_real"]), flush=True)

# ---- seam2 by matched arclength within lower-donor groups ----------------
groups = {}
for n in OUT:
    groups.setdefault(OUT[n]["lower"], []).append(n)
for lo in groups:
    mem = groups[lo]
    for m in mem:
        C, S, L = CUR[m]
        P, g = densify(C)
        vals = []
        for o in mem:
            if o == m:
                continue
            Co, So, Lo = CUR[o]
            Q = resample_at(Co, So, np.clip(g, 0, So[-1]))
            dmm = np.linalg.norm(P - Q, axis=1)
            s2 = first_cross(g, dmm, 1.0)
            vals.append(s2 if s2 is not None else L)
        OUT[m]["seam2_m_1mm"] = float(np.min(vals)) if vals else None
        OUT[m]["seam2_m_med"] = float(np.median(vals)) if vals else None
        OUT[m]["seam2_n_sib"] = len(vals)

print("\n@@@JSON@@@")
print(json.dumps(OUT))
