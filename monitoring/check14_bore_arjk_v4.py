"""CHECK 14 addendum 3 -- is the optimism a CONSTANT offset (absorbable) or
PROPORTIONAL to the declared radius (calibre-dependent, so it varies wherever the
vessel narrows)? Pooled over a subsample of set B, per segment.
"""
import glob, json, os, sys
import numpy as np, pyvista as pv, vtk

sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

SEAM = 130.0


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(p, r, step=0.25):
    s = arclen(p)
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1), np.interp(t, s, r), t


S = {k: {"c": [], "r": []} for k in ("host", "lower", "siphon", "reca")}
per = []
dirs = [d for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")) if os.path.isdir(d)]
for d in dirs[::4]:
    prov = json.load(open(os.path.join(d, "provenance.json")))
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    rec = {"name": os.path.basename(d)}
    for want in ("RCCA", "RECA"):
        k = [x for x in brs if want in x.upper()]
        if not k:
            continue
        b = brs[k[0]]
        p = np.asarray(b.coordinates, float); r = np.asarray(b.radii, float)
        q, rr, t = densify(p, r)
        c = np.array([f.EvaluateFunction(pt) for pt in q])
        if want == "RECA":
            S["reca"]["c"].append(c); S["reca"]["r"].append(rr)
            rec["reca_r"] = float(np.median(rr)); rec["reca_d"] = float(np.median(c - rr))
            continue
        hc = prov["host_cut_mm"]
        for lab, msk in (("host", t < hc), ("lower", (t >= hc) & (t < SEAM)),
                         ("siphon", t >= SEAM)):
            if msk.sum():
                S[lab]["c"].append(c[msk]); S[lab]["r"].append(rr[msk])
                rec[lab + "_r"] = float(np.median(rr[msk]))
                rec[lab + "_d"] = float(np.median(c[msk] - rr[msk]))
    per.append(rec)

print("subsample n = %d anatomies" % len(per))
print()
print("=== POOLED: exact bore vs declared radius, per segment ===")
print("%-8s %9s %9s %9s %9s %9s %9s %9s" %
      ("segment", "n_pts", "slope", "intercept", "r2", "resid_sd", "const_sd", "ratio_sd"))
for k in ("host", "lower", "siphon", "reca"):
    c = np.concatenate(S[k]["c"]); r = np.concatenate(S[k]["r"])
    A = np.stack([r, np.ones_like(r)], axis=1)
    sl, ic = np.linalg.lstsq(A, c, rcond=None)[0]
    pred = A @ np.array([sl, ic])
    ss = 1 - np.var(c - pred) / np.var(c)
    d = c - r
    print("%-8s %9d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f"
          % (k, len(c), sl, ic, ss, np.std(c - pred), np.std(d), np.std(c / r)))
print()
print("const_sd = sd of (bore - declared r): spread left by a CONSTANT-offset model")
print("ratio_sd = sd of (bore / declared r): spread left by a PROPORTIONAL model")

print()
print("=== per-segment declared r and bore, pooled medians ===")
print("%-8s %10s %10s %10s %10s" % ("segment", "decl med", "bore med", "delta med", "ratio med"))
for k in ("host", "lower", "siphon", "reca"):
    c = np.concatenate(S[k]["c"]); r = np.concatenate(S[k]["r"])
    print("%-8s %10.3f %10.3f %10.3f %10.3f"
          % (k, np.median(r), np.median(c), np.median(c - r), np.median(c / r)))

print()
print("=== ACROSS ANATOMIES: does segment delta track segment calibre? ===")
for k in ("host", "lower", "siphon", "reca"):
    x = np.array([p[k + "_r"] for p in per if k + "_r" in p])
    y = np.array([p[k + "_d"] for p in per if k + "_d" in p])
    print("   %-7s pearson(declared r, delta) = %+.3f   (n=%d, declared r range %.2f-%.2f)"
          % (k, np.corrcoef(x, y)[0, 1], len(x), x.min(), x.max()))

print()
print("=== what a policy sees: obs 47/48/49 use declared r; error at the tight end ===")
c = np.concatenate([np.concatenate(S[k]["c"]) for k in ("host", "lower", "siphon")])
r = np.concatenate([np.concatenate(S[k]["r"]) for k in ("host", "lower", "siphon")])
print("%-16s %8s %9s %9s %9s" % ("declared r bin", "n", "bore med", "delta med", "ratio med"))
for lo, hi in [(1.5, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 2.7), (2.7, 3.0), (3.0, 4.0), (4.0, 6.0)]:
    m = (r >= lo) & (r < hi)
    if m.sum() < 50:
        continue
    print("%5.1f - %-8.1f %8d %9.3f %9.3f %9.3f"
          % (lo, hi, m.sum(), np.median(c[m]), np.median(c[m] - r[m]), np.median(c[m] / r[m])))
