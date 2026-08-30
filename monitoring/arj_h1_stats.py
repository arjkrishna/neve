#!/usr/bin/env python3
"""H1: OOD distance vs grafted success. Heterogeneity first, then episode-level.
Pure numpy (no scipy on this host): chi2 sf via regularized incomplete gamma,
logistic regression via IRLS with anatomy-clustered sandwich SEs.
"""
import json, math, os, pickle, itertools
import numpy as np

ROOT = "D:/Arjun/workspace/neve"
M = os.path.join(ROOT, "monitoring")
GRAFT_PL = 166.91
SEAM = 133.6

# ---------------- distributions ----------------
def gammainc_p(a, x):
    if x < 0 or a <= 0: return float("nan")
    if x == 0: return 0.0
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(2000):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15: break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c = x + 1.0 - a, 1e300
    d = 1.0 / b; h = d
    for i in range(1, 2000):
        an = -i * (i - a); b += 2.0
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < 1e-15: break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

def chi2_sf(x, df): return max(0.0, min(1.0, 1.0 - gammainc_p(df / 2.0, x / 2.0)))
def norm_sf(z): return 0.5 * math.erfc(z / math.sqrt(2.0))
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # average ties
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); i = 0
        s = np.sort(v)
        while i < len(v):
            j = i
            while j + 1 < len(v) and s[j + 1] == s[i]: j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    return pearson(ra, rb)

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))

def wpearson(a, b, w):
    a = np.asarray(a, float); b = np.asarray(b, float); w = np.asarray(w, float)
    ma = (w * a).sum() / w.sum(); mb = (w * b).sum() / w.sum()
    ca = a - ma; cb = b - mb
    return float((w * ca * cb).sum() / math.sqrt((w * ca * ca).sum() * (w * cb * cb).sum()))

def partial(a, b, c):
    """partial corr of a,b given the columns of c (list of arrays)."""
    C = np.column_stack([np.ones(len(a))] + [np.asarray(x, float) for x in c])
    def resid(y):
        y = np.asarray(y, float)
        beta, *_ = np.linalg.lstsq(C, y, rcond=None)
        return y - C @ beta
    return pearson(resid(a), resid(b))

# ---------------- logistic with cluster-robust SE ----------------
def logit(X, y, groups, maxit=200):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ b; p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10: break
    eta = X @ b; p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    W = np.maximum(p * (1 - p), 1e-9)
    Hinv = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))
    u = X * (y - p)[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gg in set(groups):
        m = np.array([q == gg for q in groups])
        s = u[m].sum(0)
        meat += np.outer(s, s)
        G = len(set(groups)); n = len(y)
    G = len(set(groups)); n = len(y); k = X.shape[1]
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = Hinv @ (adj * meat) @ Hinv
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    return b, np.sqrt(np.diag(V)), ll, np.sqrt(np.diag(Hinv))

# ---------------- data ----------------
eps = json.load(open(os.path.join(M, "arj_t3_merged.json")))
G = pickle.load(open(os.path.join(M, "_h1_geom.pkl"), "rb"))
DEV = pickle.load(open(os.path.join(M, "_h1_dev.pkl"), "rb"))
host, proc, coh = G["host"], G["proc"], G["coh"]
short = {f["tag"]: "topcowmr" + f["tag"][-3:] for f in coh}
inv = {v: k for k, v in short.items()}

FEAT = ["graft_len", "tort", "kap_p95", "turn_per_mm", "r_med", "r_min"]
DEGEN = ["Rc_min", "bend_max", "Rc_min_d40", "bend_max_d40"]

P = np.array([[f[k] for k in FEAT] for f in proc])
mu, sd = P.mean(0), P.std(0, ddof=1)
Z = (P - mu) / sd
S = np.cov(Z.T) + 1e-6 * np.eye(len(FEAT))
Si = np.linalg.inv(S)
print("procedural feature-space condition number: %.1f" % np.linalg.cond(S))

def pt2curve(Pp, C):
    A = C[:-1]; B = C[1:]; AB = B - A
    den = np.maximum((AB * AB).sum(1), 1e-12)
    out = np.empty(len(Pp))
    for i, p in enumerate(Pp):
        t = np.clip(((p - A) * AB).sum(1) / den, 0.0, 1.0)
        out[i] = np.linalg.norm(A + t[:, None] * AB - p, axis=1).min()
    return out

pg = [f["_q"][f["_t"] >= SEAM] for f in proc]

rows = {}
for f in coh:
    a = short[f["tag"]]
    z = (np.array([f[k] for k in FEAT]) - mu) / sd
    d = z - Z.mean(0)
    maha = float(math.sqrt(max(d @ Si @ d, 0)))
    znn = float(np.min(np.linalg.norm(Z - z[None, :], axis=1)))
    q = f["_q"][f["_t"] >= SEAM]
    shp = float(min(np.median(pt2curve(q, c)) for c in pg))
    rows[a] = dict(anat=a, maha=maha, znn=znn, shape_nn=shp,
                   dev_med=DEV[f["tag"]]["dev_med"], dev_max=DEV[f["tag"]]["dev_max"],
                   dev_d40=DEV[f["tag"]]["dev_d40"],
                   **{k: f[k] for k in FEAT + DEGEN})
    rows[a]["z_" + "tort"] = float(z[FEAT.index("tort")])

# intra-procedural calibration for shape_nn
sub = list(range(0, len(pg), 4))
intra = []
for i in sub:
    dd = [float(np.median(pt2curve(pg[i], pg[j]))) for j in range(len(pg)) if j != i]
    intra.append(min(dd))
print("intra-procedural shape NN (median pt-to-curve, mm) over %d probes: "
      "mean %.3f  p50 %.3f  max %.3f" % (len(sub), np.mean(intra), np.median(intra), max(intra)))

# host itself, placed in the family
zh = (np.array([host[k] for k in FEAT]) - mu) / sd
dh = zh - Z.mean(0)
print("HOST maha=%.2f  znn=%.3f  shape_nn=%.3f mm"
      % (math.sqrt(dh @ Si @ dh), np.min(np.linalg.norm(Z - zh[None, :], axis=1)),
         min(np.median(pt2curve(host["_q"][host["_t"] >= SEAM], c)) for c in pg)))

# ---------------- episode table ----------------
E = [e for e in eps if e["pl"] > GRAFT_PL]
for e in E:
    e["a"] = e["anat"]
print()
print("grafted episodes n=%d  teacher %d  heuristic %d" %
      (len(E), sum(e["T"] for e in E), sum(e["H"] for e in E)))

ANAT = sorted(set(e["a"] for e in E))
per = {a: dict(n=0, T=0, H=0) for a in ANAT}
for e in E:
    per[e["a"]]["n"] += 1; per[e["a"]]["T"] += e["T"]; per[e["a"]]["H"] += e["H"]

# ---------------- (1) heterogeneity ----------------
def het(key, keep=None):
    aa = [a for a in ANAT if (keep is None or a in keep)]
    n = np.array([per[a]["n"] for a in aa], float)
    k = np.array([per[a][key] for a in aa], float)
    p0 = k.sum() / n.sum()
    exp = n * p0
    chi = float(((k - exp) ** 2 / np.maximum(exp, 1e-9) +
                 ((n - k) - (n - exp)) ** 2 / np.maximum(n - exp, 1e-9)).sum())
    ll1 = 0.0
    for ki, ni in zip(k, n):
        pi = ki / ni
        if 0 < pi < 1: ll1 += ki * math.log(pi) + (ni - ki) * math.log(1 - pi)
    ll0 = k.sum() * math.log(p0) + (n.sum() - k.sum()) * math.log(1 - p0)
    lr = 2 * (ll1 - ll0)
    df = len(aa) - 1
    return dict(g=len(aa), p0=p0, chi=chi, chi_p=chi2_sf(chi, df),
                lr=lr, lr_p=chi2_sf(lr, df), df=df, disp=chi / df)

print()
print("=== (1) HETEROGENEITY: does the per-anatomy spread exceed binomial noise? ===")
for key in ("T", "H"):
    for lbl, keep in (("all22", None), ("excl mr_025", set(ANAT) - {"topcowmr025"}),
                      ("excl 025+024", set(ANAT) - {"topcowmr025", "topcowmr024"})):
        r = het(key, keep)
        print("%-9s %-13s g=%2d pooled p=%.3f  chi2=%6.2f df=%2d p=%.3f | LRT=%6.2f p=%.3f | dispersion=%.2f"
              % ("teacher" if key == "T" else "heuristic", lbl, r["g"], r["p0"], r["chi"],
                 r["df"], r["chi_p"], r["lr"], r["lr_p"], r["disp"]))

print()
print("per-anatomy grafted rates + Wilson 95%% CI (teacher | heuristic)")
print("%-12s %3s  %-18s   %-18s  %6s" % ("anat", "n", "teacher", "heuristic", "T-H"))
for a in sorted(ANAT, key=lambda x: -per[x]["T"] / per[x]["n"]):
    d = per[a]; lo, hi = wilson(d["T"], d["n"]); lo2, hi2 = wilson(d["H"], d["n"])
    print("%-12s %3d  %d/%d %5.1f%% [%4.1f,%5.1f]  %d/%d %5.1f%% [%4.1f,%5.1f] %+6d"
          % (a, d["n"], d["T"], d["n"], 100 * d["T"] / d["n"], 100 * lo, 100 * hi,
             d["H"], d["n"], 100 * d["H"] / d["n"], 100 * lo2, 100 * hi2, d["T"] - d["H"]))

pickle.dump(dict(rows=rows, per=per, E=E), open(os.path.join(M, "_h1_stats.pkl"), "wb"))

# ---------------- (2) anatomy-level correlations ----------------
print()
print("=== (2) anatomy-level (n=22, weak) OOD / difficulty vs grafted success ===")
KEEPS = [("all22", ANAT), ("excl mr_025", [a for a in ANAT if a != "topcowmr025"]),
         ("excl 025+024", [a for a in ANAT if a not in ("topcowmr025", "topcowmr024")])]
METR = ["maha", "znn", "shape_nn", "dev_med", "dev_max", "dev_d40",
        "tort", "graft_len", "Rc_min", "bend_max", "r_min", "r_med", "kap_p95", "turn_per_mm"]
for lbl, aa in KEEPS:
    rT = np.array([per[a]["T"] / per[a]["n"] for a in aa])
    rH = np.array([per[a]["H"] / per[a]["n"] for a in aa])
    w = np.array([per[a]["n"] for a in aa], float)
    print("-- %s (n=%d anatomies) --" % (lbl, len(aa)))
    print("%-12s %8s %8s %8s %8s" % ("metric", "wr_T", "sp_T", "wr_H", "wr_T-H"))
    for k in METR:
        v = np.array([rows[a][k] for a in aa])
        print("%-12s %8.3f %8.3f %8.3f %8.3f" %
              (k, wpearson(v, rT, w), spearman(v, rT), wpearson(v, rH, w),
               wpearson(v, rT - rH, w)))

# ---------------- (3) episode-level models ----------------
print()
print("=== (3) EPISODE-LEVEL logistic, anatomy-clustered robust SE (n=%d) ===" % len(E))
def build(keep, cols, dep="T"):
    ee = [e for e in E if e["a"] in keep]
    y = np.array([e[dep] for e in ee], float)
    grp = [e["a"] for e in ee]
    X = [np.ones(len(ee))]
    names = ["const"]
    for nm, fn in cols:
        v = np.array([fn(e) for e in ee], float)
        v = (v - v.mean()) / (v.std() + 1e-12)
        X.append(v); names.append(nm)
    return np.column_stack(X), y, grp, names

def rep(keep, cols, dep="T", tag=""):
    X, y, grp, names = build(keep, cols, dep)
    b, se, ll, se_naive = logit(X, y, grp)
    print("  [%s] n=%d  dep=%s" % (tag, len(y), dep))
    for nm, bi, si in zip(names, b, se):
        z = bi / si if si > 0 else float("nan")
        print("     %-14s beta=%+7.3f  cl.se=%.3f  z=%+6.2f  p=%.4f" %
              (nm, bi, si, z, 2 * norm_sf(abs(z))))
    return b, se, ll

keep_all = set(ANAT)
keep_no25 = keep_all - {"topcowmr025"}
f_pl = lambda e: e["pl"]
for oodk in ["maha", "shape_nn", "dev_med", "znn"]:
    f_ood = (lambda k: (lambda e: rows[e["a"]][k]))(oodk)
    print()
    print(" >>> OOD metric = %s" % oodk)
    rep(keep_no25, [("path_len", f_pl)], "T", "depth only")
    rep(keep_no25, [("path_len", f_pl), ("OOD", f_ood)], "T", "depth+OOD")
    rep(keep_no25, [("path_len", f_pl), ("OOD", f_ood),
                    ("Rc_min", lambda e: rows[e["a"]]["Rc_min"]),
                    ("tort", lambda e: rows[e["a"]]["tort"]),
                    ("r_min", lambda e: rows[e["a"]]["r_min"])], "T", "depth+OOD+difficulty")
    rep(keep_no25, [("path_len", f_pl), ("OOD", f_ood), ("Hsucc", lambda e: e["H"])],
        "T", "depth+OOD+heuristic-outcome (policy-specific test)")
    rep(keep_no25, [("path_len", f_pl), ("OOD", f_ood)], "H", "heuristic control")
