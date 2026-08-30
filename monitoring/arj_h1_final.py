#!/usr/bin/env python3
"""H1 part 4: randomization test on the EPISODE-LEVEL logistic, envelope-exit
tally, direction-of-departure decomposition, teacher/heuristic union bound."""
import json, math, os, pickle
import numpy as np

M = "D:/Arjun/workspace/neve/monitoring"
S = pickle.load(open(os.path.join(M, "_h1_stats.pkl"), "rb"))
G = pickle.load(open(os.path.join(M, "_h1_geom.pkl"), "rb"))
rows, per, E = S["rows"], S["per"], S["E"]
host, proc, coh = G["host"], G["proc"], G["coh"]
ANAT = sorted(per)
RNG = np.random.default_rng(7)
SEAM = 133.6


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - h), min(1., c + h)


def irls(X, y, it=40):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + 1e-7 * np.eye(X.shape[1])
        st = np.linalg.solve(H, X.T @ (y - p))
        b = b + st
        if np.max(np.abs(st)) < 1e-9: break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    return b, ll


print("=== (A) teacher / heuristic union bound on the 124 grafted episodes ===")
tt = np.array([e["T"] for e in E]); hh = np.array([e["H"] for e in E])
n11 = int(((tt == 1) & (hh == 1)).sum()); n10 = int(((tt == 1) & (hh == 0)).sum())
n01 = int(((tt == 0) & (hh == 1)).sum()); n00 = int(((tt == 0) & (hh == 0)).sum())
u = n11 + n10 + n01
print("  teacher %d/124 = %.1f%%   heuristic %d/124 = %.1f%%" %
      (tt.sum(), 100 * tt.mean(), hh.sum(), 100 * hh.mean()))
print("  solved by AT LEAST ONE: %d/124 = %.1f%% (Wilson %.1f-%.1f)" %
      (u, 100 * u / 124, 100 * wilson(u, 124)[0], 100 * wilson(u, 124)[1]))
print("  solved by NEITHER:      %d/124 = %.1f%% (Wilson %.1f-%.1f)  <- upper bound on"
      " episodes plausibly attributable to intrinsic difficulty" %
      (n00, 100 * n00 / 124, 100 * wilson(n00, 124)[0], 100 * wilson(n00, 124)[1]))
print("  of the %d teacher FAILURES, %d (%.1f%%) were solved by the plain heuristic on the"
      " identical anatomy+target+seed" % (n01 + n00, n01, 100 * n01 / (n01 + n00)))
print("  of the %d heuristic failures, %d (%.1f%%) were solved by the teacher" %
      (n10 + n00, n10, 100 * n10 / (n10 + n00)))

print()
print("  per-anatomy: n, T, H, solved-by-neither")
byan = {}
for e in E:
    a = e["a"]; byan.setdefault(a, []).append((e["T"], e["H"]))
print("  %-12s %3s %3s %3s %8s" % ("anat", "n", "T", "H", "neither"))
for a in sorted(byan):
    v = byan[a]
    print("  %-12s %3d %3d %3d %8d" % (a, len(v), sum(x[0] for x in v),
                                       sum(x[1] for x in v),
                                       sum(1 for x in v if x == (0, 0))))

print()
print("=== (B) RANDOMIZATION test on the EPISODE-LEVEL logistic ===")
print("  model: T ~ 1 + path_len + OOD(anatomy-level); OOD labels permuted across the")
print("  anatomies (n and outcomes fixed); statistic = |beta_OOD|. 20000 perms.")
NP = 20000
for kl, keep in (("all22", set(ANAT)), ("excl 025", set(ANAT) - {"topcowmr025"})):
    ee = [e for e in E if e["a"] in keep]
    aa = sorted(set(e["a"] for e in ee))
    y = np.array([e["T"] for e in ee], float)
    pl = np.array([e["pl"] for e in ee]); pl = (pl - pl.mean()) / pl.std()
    ai = np.array([aa.index(e["a"]) for e in ee])
    obs = {}
    for k in ["maha", "znn", "shape_nn", "dev_med", "dev_max", "dev_d40"]:
        v0 = np.array([rows[a][k] for a in aa])
        v0 = (v0 - v0.mean()) / v0.std()
        X = np.column_stack([np.ones(len(y)), pl, v0[ai]])
        b, _ = irls(X, y)
        idx = np.argsort(RNG.random((NP, len(aa))), axis=1)
        cnt = 0
        for j in range(NP):
            vp = v0[idx[j]]
            Xp = np.column_stack([np.ones(len(y)), pl, vp[ai]])
            bp, _ = irls(Xp, y, it=25)
            if abs(bp[2]) >= abs(b[2]) - 1e-12: cnt += 1
        obs[k] = (b[2], cnt / NP)
        print("  [%s] %-9s beta=%+7.3f  randomization p=%.4f" % (kl, k, b[2], cnt / NP))
    best = max(obs, key=lambda k: abs(obs[k][0]))
    print("  [%s] best OOD metric = %s (|beta|=%.3f, p=%.4f); 6 metrics tried -> "
          "Bonferroni-adjusted p = %.3f" % (kl, best, abs(obs[best][0]), obs[best][1],
                                            min(1.0, 6 * obs[best][1])))

print()
print("=== (C) how far outside the procedural envelope is each anatomy? ===")
KEYS = ["graft_len", "tort", "Rc_min", "kap_p95", "bend_max", "turn_cum",
        "turn_per_mm", "r_min", "r_med"]
P = {k: np.array([f[k] for f in proc]) for k in KEYS}
print("  feature      proc[min,max] (208 draws)     host      n_cohort_OUTSIDE  direction")
for k in KEYS:
    lo, hi = P[k].min(), P[k].max()
    nb = sum(1 for f in coh if f[k] < lo); na = sum(1 for f in coh if f[k] > hi)
    d = ("%d below / %d above" % (nb, na))
    print("  %-12s [%9.3f,%9.3f]  %9.3f   %2d/22 outside   %s"
          % (k, lo, hi, host[k], nb + na, d))

print()
print("  per-anatomy count of the 9 features outside the procedural [min,max]:")
for f in sorted(coh, key=lambda z: z["tag"]):
    out = [k for k in KEYS if f[k] < P[k].min() or f[k] > P[k].max()]
    print("    %-16s %d/9  %s" % (f["tag"], len(out), ",".join(out)))

print()
print("=== (D) direction of departure ===")
hq, ht = host["_q"], host["_t"]
HG = hq[ht >= SEAM]
tan = hq[-1] - hq[0]; tan = tan / np.linalg.norm(tan)   # the axis the generator uses
print("  the generator displaces ONLY in the plane perpendicular to the overall RCCA")
print("  chord (perpendicular_basis(coords[-1]-coords[0])): the along-chord component")
print("  of a procedural displacement is identically 0.")


def depart(f):
    q, t = f["_q"], f["_t"]
    g = t >= SEAM
    Q = q[g]
    A = HG[:-1]; B = HG[1:]; AB = B - A
    den = np.maximum((AB * AB).sum(1), 1e-12)
    vec = np.empty_like(Q)
    for i, p in enumerate(Q):
        s = np.clip(((p - A) * AB).sum(1) / den, 0, 1)
        pr = A + s[:, None] * AB
        d = np.linalg.norm(pr - p, axis=1)
        j = int(d.argmin())
        vec[i] = p - pr[j]
    return vec


pa = []
for f in proc:
    v = depart(f)
    n = np.linalg.norm(v, axis=1)
    pa.append(float(np.abs((v * tan).sum(1))[n > 1e-6].mean()))
print("  procedural: mean |along-chord| component of the departure vector = %.3f mm"
      " (should be ~0 by construction; residual is polyline projection only)"
      % float(np.mean(pa)))
print("  %-16s %8s %8s %8s   %s" % ("anat", "|along|", "|perp|", "along/tot", "mean unit dir (vessel CS)"))
dirs = {}
for f in sorted(coh, key=lambda z: z["tag"]):
    v = depart(f)
    al = (v * tan).sum(1)
    pe = v - al[:, None] * tan
    mu = v.mean(0); mu = mu / max(np.linalg.norm(mu), 1e-9)
    dirs[f["tag"]] = mu
    print("  %-16s %8.2f %8.2f %8.3f   [%+.2f %+.2f %+.2f]"
          % (f["tag"], np.abs(al).mean(), np.linalg.norm(pe, axis=1).mean(),
             np.abs(al).mean() / max(np.linalg.norm(v, axis=1).mean(), 1e-9),
             mu[0], mu[1], mu[2]))
D = np.array([dirs[f["tag"]] for f in sorted(coh, key=lambda z: z["tag"])])
C = D @ D.T
iu = np.triu_indices(len(D), 1)
print("  pairwise cos(angle) between the 22 mean departure directions:"
      " mean %.3f  median %.3f  min %.3f  max %.3f" %
      (C[iu].mean(), np.median(C[iu]), C[iu].min(), C[iu].max()))
print("  fraction of pairs with cos>0.5 (same general direction): %.2f" % (C[iu] > 0.5).mean())
