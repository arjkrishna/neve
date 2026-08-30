#!/usr/bin/env python3
"""H1 part 3: exact Monte-Carlo nulls (heterogeneity + anatomy-label permutation),
paired teacher-vs-heuristic decomposition, and the H1/H2 confound matrix."""
import json, math, os, pickle
import numpy as np

M = "D:/Arjun/workspace/neve/monitoring"
S = pickle.load(open(os.path.join(M, "_h1_stats.pkl"), "rb"))
rows, per, E = S["rows"], S["per"], S["E"]
ANAT = sorted(per)
RNG = np.random.default_rng(20260829)
NSIM = 50000

FEATS = ["maha", "znn", "shape_nn", "dev_med", "dev_max", "dev_d40"]
DIFF = ["tort", "graft_len", "Rc_min", "bend_max", "r_min", "r_med", "kap_p95", "turn_per_mm"]


def stats_from(counts, ns):
    p0 = counts.sum(-1) / ns.sum()
    exp = p0[..., None] * ns
    chi = (((counts - exp) ** 2) / np.maximum(exp, 1e-9) +
           (((ns - counts) - (ns - exp)) ** 2) / np.maximum(ns - exp, 1e-9)).sum(-1)
    p = np.clip(counts / ns, 1e-12, 1 - 1e-12)
    ll1 = (counts * np.log(p) + (ns - counts) * np.log(1 - p)).sum(-1)
    pp = np.clip(p0, 1e-12, 1 - 1e-12)
    ll0 = counts.sum(-1) * np.log(pp) + (ns.sum() - counts.sum(-1)) * np.log(1 - pp)
    return chi, 2 * (ll1 - ll0)


def mc_het(keep, key):
    aa = [a for a in ANAT if a in keep]
    ns = np.array([per[a]["n"] for a in aa], float)
    k = np.array([per[a][key] for a in aa], float)
    p0 = k.sum() / ns.sum()
    chi_o, lr_o = stats_from(k, ns)
    sim = RNG.binomial(ns.astype(int)[None, :].repeat(NSIM, 0), p0).astype(float)
    chi_s, lr_s = stats_from(sim, ns)
    return dict(chi=chi_o, chi_p=float((chi_s >= chi_o - 1e-9).mean()),
                lr=lr_o, lr_p=float((lr_s >= lr_o - 1e-9).mean()),
                var_obs=float(np.var(k / ns)),
                var_p=float((np.var(sim / ns, axis=1) >= np.var(k / ns) - 1e-12).mean()),
                g=len(aa), p0=p0)


print("=== EXACT MONTE-CARLO NULL for per-anatomy heterogeneity (%d sims) ===" % NSIM)
print("null: every anatomy shares the pooled rate; ns held at the observed values")
for key, lbl in (("T", "teacher"), ("H", "heuristic")):
    for kl, keep in (("all22", set(ANAT)), ("excl 025", set(ANAT) - {"topcowmr025"}),
                     ("excl 025+024", set(ANAT) - {"topcowmr025", "topcowmr024"})):
        r = mc_het(keep, key)
        print("%-9s %-13s g=%2d p0=%.3f | Pearson chi2=%6.2f MC-p=%.4f |"
              " LRT=%6.2f MC-p=%.4f | var(rates)=%.4f MC-p=%.4f"
              % (lbl, kl, r["g"], r["p0"], r["chi"], r["chi_p"], r["lr"], r["lr_p"],
                 r["var_obs"], r["var_p"]))

# ---- paired teacher vs heuristic, episode level ----
print()
print("=== PAIRED teacher vs heuristic on the SAME 124 seeds ===")
tt = np.array([e["T"] for e in E]); hh = np.array([e["H"] for e in E])
n11 = int(((tt == 1) & (hh == 1)).sum()); n10 = int(((tt == 1) & (hh == 0)).sum())
n01 = int(((tt == 0) & (hh == 1)).sum()); n00 = int(((tt == 0) & (hh == 0)).sum())
print("  T=1,H=1 %d   T=1,H=0 %d   T=0,H=1 %d   T=0,H=0 %d" % (n11, n10, n01, n00))
b, c = n10, n01
mcn = (abs(b - c) - 1) ** 2 / max(b + c, 1)
print("  McNemar chi2(cc)=%.3f  b=%d c=%d  -> teacher %+d vs heuristic overall" % (mcn, b, c, n11 + n10 - (n11 + n01)))
phi = float(np.corrcoef(tt, hh)[0, 1])
print("  episode-level phi(T,H) = %+.3f  (chance-level agreement would be ~0 given both ~0.56)" % phi)
print("  agreement %d/%d = %.1f%%" % (n11 + n00, len(tt), 100 * (n11 + n00) / len(tt)))

# ---- anatomy-label permutation test for anatomy-level covariates ----
print()
print("=== ANATOMY-LABEL PERMUTATION test (%d perms) ===" % NSIM)
print("null: the covariate is unrelated to the anatomy's outcomes; the 22 covariate")
print("values are permuted across anatomies, outcomes and ns held fixed. Statistic =")
print("n-weighted Pearson r between covariate and per-anatomy rate. Exact under H0.")


def perm_test(keep, key, dep):
    aa = [a for a in ANAT if a in keep]
    ns = np.array([per[a]["n"] for a in aa], float)
    rate = np.array([per[a][dep] / per[a]["n"] for a in aa])
    v = np.array([rows[a][key] for a in aa])

    def wr(x):
        mx = (ns * x).sum() / ns.sum(); my = (ns * rate).sum() / ns.sum()
        cx = x - mx; cy = rate - my
        return (ns * cx * cy).sum() / math.sqrt((ns * cx * cx).sum() * (ns * cy * cy).sum())
    o = wr(v)
    idx = np.argsort(RNG.random((NSIM, len(aa))), axis=1)
    Vp = v[idx]
    mx = (ns * Vp).sum(1) / ns.sum()
    my = (ns * rate).sum() / ns.sum()
    cx = Vp - mx[:, None]; cy = rate - my
    num = (ns * cx * cy).sum(1)
    den = np.sqrt((ns * cx * cx).sum(1) * (ns * cy * cy).sum())
    s = num / den
    return o, float((np.abs(s) >= abs(o) - 1e-12).mean()), s


for kl, keep in (("all22", set(ANAT)), ("excl 025", set(ANAT) - {"topcowmr025"})):
    print("-- %s --" % kl)
    print("%-12s %8s %8s | %8s %8s | %8s %8s" %
          ("covariate", "wr_T", "p_T", "wr_H", "p_H", "wr_diff", "p_diff"))
    allsim = []
    for k in FEATS + DIFF:
        oT, pT, sT = perm_test(keep, k, "T")
        oH, pH, _ = perm_test(keep, k, "H")
        aa = [a for a in ANAT if a in keep]
        ns = np.array([per[a]["n"] for a in aa], float)
        d = np.array([(per[a]["T"] - per[a]["H"]) / per[a]["n"] for a in aa])
        v = np.array([rows[a][k] for a in aa])
        mx = (ns * v).sum() / ns.sum(); my = (ns * d).sum() / ns.sum()
        cx = v - mx; cy = d - my
        od = (ns * cx * cy).sum() / math.sqrt((ns * cx * cx).sum() * (ns * cy * cy).sum())
        idx = np.argsort(RNG.random((NSIM, len(aa))), axis=1)
        Vp = v[idx]
        mxp = (ns * Vp).sum(1) / ns.sum()
        cxp = Vp - mxp[:, None]
        sd_ = (ns * cxp * cy).sum(1) / np.sqrt((ns * cxp * cxp).sum(1) * (ns * cy * cy).sum())
        pd_ = float((np.abs(sd_) >= abs(od) - 1e-12).mean())
        print("%-12s %8.3f %8.4f | %8.3f %8.4f | %8.3f %8.4f" % (k, oT, pT, oH, pH, od, pd_))
        if k in FEATS:
            allsim.append(np.abs(sT))
            if k == FEATS[0]:
                obs_max = []
            obs_max.append(abs(oT))
    A = np.max(np.vstack(allsim), axis=0)
    print("  family-wise (max |wr_T| over the %d OOD metrics) obs=%.3f  MC-p=%.4f"
          % (len(FEATS), max(obs_max), float((A >= max(obs_max) - 1e-12).mean())))

# ---- H1 / H2 confound matrix ----
print()
print("=== H1-H2 CONFOUND: correlation of OOD metrics with difficulty proxies (n=22) ===")
V = {k: np.array([rows[a][k] for a in ANAT]) for k in FEATS + DIFF}
print("%-12s" % "" + "".join("%9s" % d[:8] for d in DIFF))
for f in FEATS:
    print("%-12s" % f + "".join("%9.3f" % np.corrcoef(V[f], V[d])[0, 1] for d in DIFF))

print()
print("=== OOD metric values per anatomy (with grafted rates) ===")
print("%-12s %3s %6s %6s | %7s %7s %8s %8s %8s %8s" %
      ("anat", "n", "T", "H", "maha", "znn", "shape_nn", "dev_med", "dev_max", "dev_d40"))
for a in sorted(ANAT, key=lambda x: rows[x]["dev_med"]):
    r = rows[a]; d = per[a]
    print("%-12s %3d %6.2f %6.2f | %7.1f %7.2f %8.2f %8.2f %8.2f %8.2f"
          % (a, d["n"], d["T"] / d["n"], d["H"] / d["n"], r["maha"], r["znn"],
             r["shape_nn"], r["dev_med"], r["dev_max"], r["dev_d40"]))
