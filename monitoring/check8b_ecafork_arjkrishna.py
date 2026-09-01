"""CHECK 8b -- is the ECA fork a real navigable decision point, all 216.

HANDOFF 11: frame (cohort .obj IS frame-consistent with its own centerlines),
sign (vtkImplicitPolyDataDistance = POSITIVE INSIDE on this build -- controlled here
against topcow_mr_001), estimator (exact signed distance, centerlines densified to
0.25 mm; never nearest-vertex, never dense surface sampling).
"""
import glob, os, sys, json
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

DS = 0.25
GW_R, CATH_R, CONTACT = 0.18, 0.35, 0.30
NAN = float("nan")


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def densify(p, r=None, step=DS):
    s = arclen(p)
    g = np.arange(0.0, s[-1] + 1e-9, step)
    q = np.column_stack([np.interp(g, s, p[:, k]) for k in range(3)])
    rr = np.interp(g, s, r) if r is not None else None
    return q, g, rr


def pt_seg_dist(q, A, B):
    """min distance from each point in q (n,3) to segments A->B, plus segment index
    and the parametric position along it."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2[L2 == 0] = 1e-12
    d = np.empty(len(q)); idx = np.empty(len(q), int); tt = np.empty(len(q))
    for i, x in enumerate(q):
        t = np.clip(np.einsum("ij,ij->i", x - A, AB) / L2, 0.0, 1.0)
        dd = np.linalg.norm(A + t[:, None] * AB - x, axis=1)
        j = int(np.argmin(dd)); d[i] = dd[j]; idx[i] = j; tt[i] = t[j]
    return d, idx, tt


def first_exceed(t, d, thr):
    k = np.nonzero(d > thr)[0]
    return float(t[k[0]]) if len(k) else None


def ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def at(p, s, target):
    return np.array([np.interp(target, s, p[:, k]) for k in range(3)])


def signed(mesh, pts):
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(mesh)
    return np.array([f.EvaluateFunction(q) for q in pts])


# ---------- SIGN / FRAME CONTROL (HANDOFF 11.2) ----------
CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
cm = pv.read(os.path.join(CTRL, "vessel_architecture_collision.obj")).triangulate().clean()
cb = [b for b in load_branches(os.path.join(CTRL, "Centrelines_comb"))
      if "RCCA" in str(b.name).upper()][0]
cs = signed(cm, np.asarray(cb.coordinates, float))
print("CONTROL topcow_mr_001 RCCA: n=%d median signed %.3f  min %.3f" % (len(cs), np.median(cs), cs.min()))
assert np.median(cs) > 0, "SIGN INVERTED -- abort"
print("  -> POSITIVE = INSIDE confirmed; magnitude at lumen scale -> frame OK\n")

ROOT = "/opt/eve_training/carotid/anatomies"
rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    try:
        prov = json.load(open(os.path.join(d, "provenance.json")))
        brs = {str(b.name).upper(): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
        kR = [k for k in brs if "RCCA" in k][0]
        kE = [k for k in brs if "RECA" in k][0]
        P = np.asarray(brs[kR].coordinates, float); PR = np.asarray(brs[kR].radii, float)
        E = np.asarray(brs[kE].coordinates, float); ER = np.asarray(brs[kE].radii, float)
        Pd, Ps, PRd = densify(P, PR)
        Ed, Es, ERd = densify(E, ER)

        # ---- fork: closest point on the RCCA polyline to the ECA origin
        d0, j0, t0 = pt_seg_dist(E[:1], P[:-1], P[1:])
        sP = arclen(P)
        j = int(j0[0])
        s_fork = float(sP[j] + t0[0] * (sP[j + 1] - sP[j]))
        origin_gap = float(d0[0])

        # ---- separation profile along the ECA
        dsep, _, _ = pt_seg_dist(Ed, Pd[:-1], Pd[1:])
        t05 = first_exceed(Es, dsep, 0.5); t10 = first_exceed(Es, dsep, 1.0)
        t20 = first_exceed(Es, dsep, 2.0); t30 = first_exceed(Es, dsep, 3.0)
        sep_max = float(dsep.max()); eca_len = float(Es[-1])

        # ---- divergence angle from the fork, tangent chords
        def dang(L):
            if s_fork + L > Ps[-1] or L > Es[-1]:
                return None
            return ang(at(Pd, Ps, s_fork + L) - at(Pd, Ps, s_fork), at(Ed, Es, L) - Ed[0])
        a2, a5, a10 = dang(2.0), dang(5.0), dang(10.0)

        # ---- exact signed distance to the mesh
        m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
        f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
        cE = np.array([f.EvaluateFunction(q) for q in Ed])
        wR = (Ps >= s_fork - 1.0) & (Ps <= s_fork + 20.0)
        cR = np.array([f.EvaluateFunction(q) for q in Pd[wR]])
        cRall = np.array([f.EvaluateFunction(q) for q in Pd])

        def cat(tv):
            return float(np.interp(tv, Es, cE)) if tv <= Es[-1] else None

        t_sep = t20 if t20 is not None else 2.0
        mE_post = float(cE[Es >= t_sep].min()) if (Es >= t_sep).any() else NAN

        r = dict(
            name=name, lower=prov["lower"], siphon=prov["siphon"],
            host_cut=prov["host_cut_mm"], cca=prov["cca_mm"], ica=prov["ica_mm"],
            eca_prov=prov["eca_mm"], eca_floor=prov["eca_floored_frac"],
            s_fork=s_fork, origin_gap=origin_gap, rcca_len=float(Ps[-1]), eca_len=eca_len,
            t05=t05, t10=t10, t20=t20, t30=t30, sep_max=sep_max,
            a2=a2, a5=a5, a10=a10,
            decl_rE2=float(np.interp(2.0, Es, ERd)),
            decl_rR=float(np.interp(s_fork + 2.0, Ps, PRd)),
            cE_min=float(cE.min()), cE_med=float(np.median(cE)),
            cE1=cat(1.0), cE2=cat(2.0), cE5=cat(5.0), cE10=cat(10.0),
            cE_post=mE_post, cE_out=int((cE < 0).sum()),
            cR_min=float(cR.min()), cR_med=float(np.median(cR)),
            cRall_med=float(np.median(cRall)),
        )
        g = np.arange(0.0, 80.0, 0.25)
        r["_pref"] = np.column_stack([np.interp(g, Ps, Pd[:, k]) for k in range(3)])
        rows.append(r)
    except Exception as e:
        print("FAIL %s: %s: %s" % (name, type(e).__name__, e))

print("measured %d anatomies\n" % len(rows))


def q(key, fmt="%.2f"):
    v = np.array([r[key] for r in rows if r[key] is not None], float)
    v = v[~np.isnan(v)]
    f = ("n=%d min " + fmt + " p10 " + fmt + " med " + fmt + " p90 " + fmt + " max " + fmt)
    return f % (len(v), v.min(), np.percentile(v, 10), np.median(v), np.percentile(v, 90), v.max())


print("=== FORK LOCATION (arclength from the RCCA ostium) ===")
print("s_fork               ", q("s_fork"))
print("origin gap ECA[0]->RCCA", q("origin_gap", "%.4f"))
print("seam1 host_cut       ", q("host_cut"))
sd = [r["s_fork"] - r["host_cut"] for r in rows]
print("s_fork - host_cut    min %.2f med %.2f max %.2f" % (min(sd), np.median(sd), max(sd)))
print("fork proximal to seam1 (inside shared host): %d / %d"
      % (sum(1 for r in rows if r["s_fork"] <= r["host_cut"]), len(rows)))
gap2 = [130.0 - r["s_fork"] for r in rows]
print("130 mm siphon seam - s_fork: min %.2f med %.2f max %.2f" % (min(gap2), np.median(gap2), max(gap2)))

print("\n=== EMPIRICAL SHARED PREFIX (all RCCAs on a common 0.25 mm arclength grid) ===")
A = np.stack([r["_pref"] for r in rows])
dev = np.linalg.norm(A - A[0], axis=2).max(axis=0)
g = np.arange(0.0, 80.0, 0.25)
for thr in (0.1, 1.0, 5.0):
    k = np.nonzero(dev > thr)[0]
    print("  first arclength with max inter-anatomy deviation > %.1f mm: %s"
          % (thr, ("%.2f mm" % g[k[0]]) if len(k) else "none within 80 mm"))
print("  distinct lower donors %d, distinct siphons %d, distinct s_fork (0.01 mm) %d"
      % (len(set(r["lower"] for r in rows)), len(set(r["siphon"] for r in rows)),
         len(set(round(r["s_fork"], 2) for r in rows))))

print("\n=== DIVERGENCE ANGLE at the fork ===")
for k, L in (("a2", 2.0), ("a5", 5.0), ("a10", 10.0)):
    print("  chord %4.1f mm  %s deg" % (L, q(k)))

print("\n=== SEPARATION: ECA arclength at which it leaves a d-mm tube around the RCCA ===")
for k, thr in (("t05", 0.5), ("t10", 1.0), ("t20", 2.0), ("t30", 3.0)):
    print("  d > %.1f mm at t = %s   (never: %d)"
          % (thr, q(k), sum(1 for r in rows if r[k] is None)))
print("  max separation       ", q("sep_max"))
print("  ECA length measured  ", q("eca_len"))
post = [r["eca_len"] - (r["t20"] if r["t20"] else r["eca_len"]) for r in rows]
print("  ECA length past the 2 mm separation point: min %.2f med %.2f max %.2f"
      % (min(post), np.median(post), max(post)))

print("\n=== LUMEN AT / PAST THE FORK (exact signed distance, positive=inside) ===")
for lbl, k in (("RECA clearance t=1 mm", "cE1"), ("RECA clearance t=2 mm", "cE2"),
               ("RECA clearance t=5 mm", "cE5"), ("RECA clearance t=10 mm", "cE10"),
               ("RECA min whole branch", "cE_min"), ("RECA min past 2mm-sep", "cE_post"),
               ("RECA median branch", "cE_med"),
               ("RCCA min [fork-1,fork+20]", "cR_min"), ("RCCA median that window", "cR_med"),
               ("RCCA median whole branch", "cRall_med"),
               ("declared r RECA t=2 mm", "decl_rE2"), ("declared r RCCA fork+2", "decl_rR")):
    print("  %-26s %s" % (lbl, q(k, "%.3f")))

print("\n=== DEVICE FIT (radii: guidewire 0.18, catheter 0.35; SOFA contactDistance 0.30) ===")
for lbl, k in (("RECA t=2 mm", "cE2"), ("RECA t=5 mm", "cE5"),
               ("RECA min past 2mm-sep", "cE_post"), ("RECA min whole", "cE_min"),
               ("RCCA fork window min", "cR_min")):
    v = np.array([r[k] for r in rows if r[k] is not None], float); v = v[~np.isnan(v)]
    print("  %-22s >=0.18 %3d/%d  >=0.35 %3d/%d  >=0.48 %3d/%d  >=0.65 %3d/%d"
          % (lbl, (v >= 0.18).sum(), len(v), (v >= 0.35).sum(), len(v),
             (v >= 0.48).sum(), len(v), (v >= 0.65).sum(), len(v)))
print("  RECA anatomies with >=1 point outside its own wall: %d, worst %d pts"
      % (sum(1 for r in rows if r["cE_out"] > 0), max(r["cE_out"] for r in rows)))

print("\n=== OUTLIERS ===")


def worst(key, n=6, rev=False):
    v = [r for r in rows if r[key] is not None and not np.isnan(r[key])]
    v.sort(key=lambda r: r[key], reverse=rev)
    for r in v[:n]:
        print("   %-46s %s=%.3f  s_fork=%.1f a5=%s t20=%s cE2=%.3f cEmin=%.3f eca=%.1f"
              % (r["name"], key, r[key], r["s_fork"],
                 ("%.1f" % r["a5"]) if r["a5"] else "na",
                 ("%.2f" % r["t20"]) if r["t20"] else "never",
                 r["cE2"] if r["cE2"] is not None else NAN, r["cE_min"], r["eca_len"]))


print(" smallest divergence angle (5 mm chord):"); worst("a5")
print(" latest 2 mm separation:");                 worst("t20", rev=True)
print(" tightest RECA clearance at t=2 mm:");      worst("cE2")
print(" tightest RECA min clearance:");            worst("cE_min")
print(" tightest RCCA clearance near the fork:");  worst("cR_min")
print(" most proximal fork:");                     worst("s_fork")

print("\n=== JOINT VERDICT COUNTS ===")


def ok(r):
    return (r["t20"] is not None and r["t20"] <= 6.0
            and r["a5"] is not None and r["a5"] >= 20.0
            and r["cE_post"] >= 0.65 and r["cR_min"] >= 0.65
            and r["s_fork"] > r["host_cut"] and r["eca_len"] >= 15.0)


print("  separates to 2 mm within 6 mm of ECA arclength: %d/%d"
      % (sum(1 for r in rows if r["t20"] is not None and r["t20"] <= 6.0), len(rows)))
print("  divergence angle (5 mm chord) >= 20 deg: %d/%d"
      % (sum(1 for r in rows if r["a5"] and r["a5"] >= 20), len(rows)))
print("  divergence angle (5 mm chord) >= 30 deg: %d/%d"
      % (sum(1 for r in rows if r["a5"] and r["a5"] >= 30), len(rows)))
print("  ECA admits catheter+contact (0.65) past separation: %d/%d"
      % (sum(1 for r in rows if r["cE_post"] >= 0.65), len(rows)))
print("  ECA admits guidewire+contact (0.48) past separation: %d/%d"
      % (sum(1 for r in rows if r["cE_post"] >= 0.48), len(rows)))
print("  RCCA admits catheter+contact at the fork window: %d/%d"
      % (sum(1 for r in rows if r["cR_min"] >= 0.65), len(rows)))
print("  fork distal to seam 1: %d/%d"
      % (sum(1 for r in rows if r["s_fork"] > r["host_cut"]), len(rows)))
print("  ALL of the above: %d/%d" % (sum(1 for r in rows if ok(r)), len(rows)))
print("  failing:", [r["name"] for r in rows if not ok(r)][:24])
