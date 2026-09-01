"""CHECK 8b, pass 2 -- separate MID-VESSEL from TERMINAL in the ECA (HANDOFF 12.2 #6),
and measure how deep each device can actually commit into the wrong branch.
"""
import glob, os, sys, json
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

DS = 0.25
NAN = float("nan")


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def densify(p, r=None, step=DS):
    s = arclen(p)
    g = np.arange(0.0, s[-1] + 1e-9, step)
    q = np.column_stack([np.interp(g, s, p[:, k]) for k in range(3)])
    return q, g, (np.interp(g, s, r) if r is not None else None)


def pt_seg_dist(q, A, B):
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB); L2[L2 == 0] = 1e-12
    d = np.empty(len(q))
    for i, x in enumerate(q):
        t = np.clip(np.einsum("ij,ij->i", x - A, AB) / L2, 0.0, 1.0)
        d[i] = np.linalg.norm(A + t[:, None] * AB - x, axis=1).min()
    return d


def depth(t, c, thr, t0):
    """last arclength reachable from t0 with clearance >= thr the whole way."""
    m = t >= t0
    tt, cc = t[m], c[m]
    bad = np.nonzero(cc < thr)[0]
    return float(tt[bad[0]] - t0) if len(bad) else float(tt[-1] - t0)


CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
cm = pv.read(os.path.join(CTRL, "vessel_architecture_collision.obj")).triangulate().clean()
cb = [b for b in load_branches(os.path.join(CTRL, "Centrelines_comb"))
      if "RCCA" in str(b.name).upper()][0]
f0 = vtk.vtkImplicitPolyDataDistance(); f0.SetInput(cm)
cs = np.array([f0.EvaluateFunction(p) for p in np.asarray(cb.coordinates, float)])
print("CONTROL topcow_mr_001 RCCA median %.3f min %.3f -> positive=inside OK\n" % (np.median(cs), cs.min()))
assert np.median(cs) > 0

rows = []
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    try:
        brs = {str(b.name).upper(): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
        P = np.asarray(brs[[k for k in brs if "RCCA" in k][0]].coordinates, float)
        E = np.asarray(brs[[k for k in brs if "RECA" in k][0]].coordinates, float)
        Pd, Ps, _ = densify(P)
        Ed, Es, _ = densify(E)
        dsep = pt_seg_dist(Ed, Pd[:-1], Pd[1:])
        k2 = np.nonzero(dsep > 2.0)[0]
        t20 = float(Es[k2[0]]) if len(k2) else NAN
        m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
        f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
        cE = np.array([f.EvaluateFunction(q) for q in Ed])
        cR = np.array([f.EvaluateFunction(q) for q in Pd])
        L = float(Es[-1])
        # where does the ECA minimum sit -- distance from the distal tip
        j = int(np.argmin(cE))
        # mid-vessel = everything except the last 3 mm (the tip cap)
        mid = Es <= L - 3.0
        midpost = mid & (Es >= t20)
        # RCCA window minimum, and where relative to the fork
        s_fork_i = int(np.argmin(np.linalg.norm(Pd - E[0], axis=1)))
        s_fork = float(Ps[s_fork_i])
        wR = (Ps >= s_fork - 1.0) & (Ps <= s_fork + 20.0)
        jr = int(np.argmin(cR[wR]))
        rows.append(dict(
            name=name, eca_len=L, t20=t20, s_fork=s_fork,
            cE_min_all=float(cE.min()), cE_argmin_from_tip=float(L - Es[j]),
            cE_min_mid=float(cE[midpost].min()) if midpost.any() else NAN,
            cE_tip3=float(cE[Es > L - 3.0].min()),
            d018=depth(Es, cE, 0.18, t20), d035=depth(Es, cE, 0.35, t20),
            d048=depth(Es, cE, 0.48, t20), d065=depth(Es, cE, 0.65, t20),
            cR_min_win=float(cR[wR].min()), cR_argmin_off=float(Ps[wR][jr] - s_fork),
            cR_at_fork=float(np.interp(s_fork, Ps, cR)),
            cR_min_dist=float(cR[Ps >= s_fork].min()),
            cR_min_prox=float(cR[Ps <= s_fork].min()),
        ))
    except Exception as e:
        print("FAIL %s: %s: %s" % (name, type(e).__name__, e))

print("measured %d\n" % len(rows))


def q(key, fmt="%.3f"):
    v = np.array([r[key] for r in rows], float); v = v[~np.isnan(v)]
    return (("n=%d min " + fmt + " p10 " + fmt + " med " + fmt + " p90 " + fmt + " max " + fmt)
            % (len(v), v.min(), np.percentile(v, 10), np.median(v), np.percentile(v, 90), v.max()))


print("=== WHERE THE ECA MINIMUM IS ===")
print("  min over whole ECA          ", q("cE_min_all"))
print("  arclength of that min from the DISTAL TIP", q("cE_argmin_from_tip", "%.2f"))
print("  anatomies whose ECA min lies in the last 3 mm: %d/%d"
      % (sum(1 for r in rows if r["cE_argmin_from_tip"] <= 3.0), len(rows)))
print("  min in the last 3 mm (tip cap)", q("cE_tip3"))
print("  MID-VESSEL min, from the 2 mm separation point to tip-3mm", q("cE_min_mid"))

print("\n=== HOW DEEP EACH DEVICE CAN COMMIT INTO THE ECA (mm past the 2 mm separation point) ===")
for lbl, k in (("guidewire r=0.18", "d018"), ("catheter r=0.35", "d035"),
               ("guidewire+contact 0.48", "d048"), ("catheter+contact 0.65", "d065")):
    print("  %-24s %s" % (lbl, q(k, "%.2f")))
for lbl, k in (("guidewire 0.18", "d018"), ("catheter 0.35", "d035"),
               ("gw+contact 0.48", "d048"), ("cath+contact 0.65", "d065")):
    v = np.array([r[k] for r in rows])
    print("  %-20s >=5 mm %3d/%d   >=10 mm %3d/%d   >=15 mm %3d/%d"
          % (lbl, (v >= 5).sum(), len(v), (v >= 10).sum(), len(v), (v >= 15).sum(), len(v)))

print("\n=== MID-VESSEL FIT (tip cap excluded) ===")
v = np.array([r["cE_min_mid"] for r in rows])
for thr in (0.18, 0.35, 0.48, 0.65):
    print("  ECA mid-vessel min >= %.2f : %d/%d" % (thr, (v >= thr).sum(), len(v)))

print("\n=== RCCA (the correct branch) AROUND THE FORK ===")
print("  clearance AT the fork        ", q("cR_at_fork"))
print("  min in [fork-1, fork+20]     ", q("cR_min_win"))
print("  offset of that min from fork ", q("cR_argmin_off", "%.2f"))
print("  min over the whole distal RCCA", q("cR_min_dist"))
print("  min over the whole proximal RCCA", q("cR_min_prox"))
v = np.array([r["cR_min_win"] for r in rows])
for thr in (0.18, 0.35, 0.48, 0.65):
    print("  RCCA fork-window min >= %.2f : %d/%d" % (thr, (v >= thr).sum(), len(v)))

print("\n=== OUTLIERS (mid-vessel ECA) ===")
s = sorted(rows, key=lambda r: r["cE_min_mid"])
for r in s[:8]:
    print("   %-46s midmin=%.3f  d035=%.1f d065=%.1f eca=%.1f tipmin=%.3f"
          % (r["name"], r["cE_min_mid"], r["d035"], r["d065"], r["eca_len"], r["cE_tip3"]))
print(" shallowest catheter penetration:")
for r in sorted(rows, key=lambda r: r["d035"])[:8]:
    print("   %-46s d035=%.2f d018=%.2f d065=%.2f midmin=%.3f eca=%.1f t20=%.2f"
          % (r["name"], r["d035"], r["d018"], r["d065"], r["cE_min_mid"], r["eca_len"], r["t20"]))
print(" tightest RCCA fork window:")
for r in sorted(rows, key=lambda r: r["cR_min_win"])[:8]:
    print("   %-46s cRmin=%.3f at fork%+.2f mm  atfork=%.3f"
          % (r["name"], r["cR_min_win"], r["cR_argmin_off"], r["cR_at_fork"]))
