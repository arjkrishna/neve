"""CHECK 8b, pass 3 -- combined verdict tally with the terminal-taper correction applied."""
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
    return np.column_stack([np.interp(g, s, p[:, k]) for k in range(3)]), g, (
        np.interp(g, s, r) if r is not None else None)


def pt_seg_dist(q, A, B):
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB); L2[L2 == 0] = 1e-12
    d = np.empty(len(q))
    for i, x in enumerate(q):
        t = np.clip(np.einsum("ij,ij->i", x - A, AB) / L2, 0.0, 1.0)
        d[i] = np.linalg.norm(A + t[:, None] * AB - x, axis=1).min()
    return d


def ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def at(p, s, t):
    return np.array([np.interp(t, s, p[:, k]) for k in range(3)])


def depth(t, c, thr, t0):
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
print("CONTROL topcow_mr_001 median %.3f min %.3f -> positive=inside\n" % (np.median(cs), cs.min()))
assert np.median(cs) > 0

rows = []
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d):
        continue
    prov = json.load(open(os.path.join(d, "provenance.json")))
    brs = {str(b.name).upper(): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    P = np.asarray(brs[[k for k in brs if "RCCA" in k][0]].coordinates, float)
    E = np.asarray(brs[[k for k in brs if "RECA" in k][0]].coordinates, float)
    Pd, Ps, _ = densify(P)
    Ed, Es, _ = densify(E)
    dsep = pt_seg_dist(Ed, Pd[:-1], Pd[1:])
    k2 = np.nonzero(dsep > 2.0)[0]
    t20 = float(Es[k2[0]]) if len(k2) else NAN
    s_fork = float(Ps[int(np.argmin(np.linalg.norm(Pd - E[0], axis=1)))])
    a5 = ang(at(Pd, Ps, s_fork + 5.0) - at(Pd, Ps, s_fork), at(Ed, Es, 5.0) - Ed[0])
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    cE = np.array([f.EvaluateFunction(q) for q in Ed])
    cR = np.array([f.EvaluateFunction(q) for q in Pd])
    L = float(Es[-1])
    rows.append(dict(name=os.path.basename(d), lower=prov["lower"], s_fork=s_fork,
                     host_cut=prov["host_cut_mm"], a5=a5, t20=t20, eca_len=L,
                     cR_fork=float(np.interp(s_fork, Ps, cR)),
                     d048=depth(Es, cE, 0.48, t20), d065=depth(Es, cE, 0.65, t20),
                     d018=depth(Es, cE, 0.18, t20), d035=depth(Es, cE, 0.35, t20)))

n = len(rows)
print("measured %d, distinct lower donors (= distinct fork geometries) %d"
      % (n, len(set(r["lower"] for r in rows))))
print("uses per lower donor: min %d med %.1f max %d" % tuple([
    min(np.bincount(np.unique([r["lower"] for r in rows], return_inverse=True)[1])),
    float(np.median(np.bincount(np.unique([r["lower"] for r in rows], return_inverse=True)[1]))),
    max(np.bincount(np.unique([r["lower"] for r in rows], return_inverse=True)[1]))]))

C = {
 "A fork distal to seam 1 (donor-varying)":      lambda r: r["s_fork"] > r["host_cut"],
 "B 2 mm separation within 6 mm of ECA arclen":  lambda r: r["t20"] <= 6.0,
 "B' 2 mm separation within 8 mm":               lambda r: r["t20"] <= 8.0,
 "B'' 2 mm separation within 12 mm":             lambda r: r["t20"] <= 12.0,
 "C divergence angle (5 mm chord) >= 15 deg":    lambda r: r["a5"] >= 15.0,
 "C' angle >= 15 deg OR 2 mm sep within 5 mm":   lambda r: (r["a5"] >= 15.0) or (r["t20"] <= 5.0),
 "D RCCA at fork admits cath+contact 0.65":      lambda r: r["cR_fork"] >= 0.65,
 "E ECA admits guidewire 0.18 for >= 10 mm":     lambda r: r["d018"] >= 10.0,
 "F ECA admits gw+contact 0.48 for >= 10 mm":    lambda r: r["d048"] >= 10.0,
 "G ECA admits cath+contact 0.65 for >= 10 mm":  lambda r: r["d065"] >= 10.0,
 "G' ECA admits cath+contact 0.65 for >= 5 mm":  lambda r: r["d065"] >= 5.0,
 "H ECA length >= 17 mm":                        lambda r: r["eca_len"] >= 17.0,
}
for k, fn in C.items():
    print("  %-46s %3d/%d" % (k, sum(1 for r in rows if fn(r)), n))

STRICT = ["A fork distal to seam 1 (donor-varying)", "B' 2 mm separation within 8 mm",
          "C' angle >= 15 deg OR 2 mm sep within 5 mm", "D RCCA at fork admits cath+contact 0.65",
          "G ECA admits cath+contact 0.65 for >= 10 mm", "H ECA length >= 17 mm"]
LOOSE = ["A fork distal to seam 1 (donor-varying)", "B'' 2 mm separation within 12 mm",
         "D RCCA at fork admits cath+contact 0.65", "F ECA admits gw+contact 0.48 for >= 10 mm",
         "H ECA length >= 17 mm"]
for lbl, keys in (("STRICT (catheter-grade fork)", STRICT), ("PERMISSIVE (guidewire-grade fork)", LOOSE)):
    ok = [r for r in rows if all(C[k](r) for k in keys)]
    print("\n%s: %d/%d anatomies, %d/%d distinct lower donors"
          % (lbl, len(ok), n, len(set(r["lower"] for r in ok)), len(set(r["lower"] for r in rows))))
    fail = [r for r in rows if r not in ok]
    from collections import Counter
    cnt = Counter()
    for r in fail:
        for k in keys:
            if not C[k](r):
                cnt[k] += 1
    for k, v in cnt.most_common():
        print("   fails %-46s %d" % (k, v))
    bad = sorted(set(r["lower"] for r in fail))
    print("   failing lower donors (%d): %s" % (len(bad), ", ".join(bad)))
