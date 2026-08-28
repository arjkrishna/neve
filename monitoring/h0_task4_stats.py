import json, os, math, collections, itertools
M = r"D:\Arjun\workspace\neve\monitoring"
OFF = 33.314
SEAM_PL = 166.91          # path_len of the graft seam (s_RCCA = 133.6)

# ---- H0 episodes (from worker STEP logs; outcomes authoritative) ----
traj = json.load(open(os.path.join(M, "h0_traj.json")))
H0 = [dict(arm="H0", mesh=r["mesh"], seed=r["seed"], succ=r["succ"],
           path_len=r["path_len"], s=r["path_len"] - OFF) for r in traj]

# ---- Teacher 98-episode run on the same 4 anatomies (success from jsonl) ----
tea = json.load(open(os.path.join(M, "h0_teacher98.json")))
JL = (r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben"
      r"\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints"
      r"\eval_anatomies_checkpoint2002292\episodes_official_20260828_045651.jsonl")
jl = {}
for line in open(JL):
    line = line.strip()
    if line:
        d = json.loads(line); jl[d["seed"]] = d
TE = [dict(arm="TEACHER", mesh=r["mesh"], seed=r["seed"],
           succ=int(bool(jl[r["seed"]]["success"])),
           path_len=r["path_len"], s=r["path_len"] - OFF) for r in tea]

# ---- reachability exclusion, measured (h0_task4_geo.out @@@JSON@@@) ----
out = open(os.path.join(M, "h0_task4_geo.out"), encoding="utf-8", errors="replace").read()
GJ = json.loads(out.split("@@@JSON@@@")[1].strip())
GEO = GJ["geo"]
SB = {k: {t: GEO[k][t]["s_block"] for t in ("wire_0.18", "sofa_0.30", "cath_0.35")}
      for k in GEO}
print("=== measured s_block (first NON-TERMINAL sub-threshold run start, mm along RCCA) ===")
for k in sorted(GEO):
    g = GEO[k]
    print("  %-12s L=%.1f nOUT=%d min_clear=%.3f  wire=%s sofa=%s cath=%s  s_last_inside=%.1f"
          % (k, g["L"], g["nout"], g["min_d"], SB[k]["wire_0.18"], SB[k]["sofa_0.30"],
             SB[k]["cath_0.35"], g["s_last_inside"]))

def excluded(ep, thr):
    b = SB[ep["mesh"]][thr]
    return b is not None and ep["s"] > b

def rate(eps):
    n = len(eps); k = sum(e["succ"] for e in eps)
    return k, n, (100.0 * k / n if n else float("nan"))

def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]]"""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    def p(x):
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)
    p0 = p(a)
    lo = max(0, c1 - r2); hi = min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-12))

print()
print("=== 1. RAW RATES, the two runs as measured ===")
for nm, E in (("H0 (ckpt0 scripted heuristic)", H0), ("TEACHER ckpt2002292", TE)):
    k, n, r = rate(E); print("  %-32s %3d/%3d = %5.1f%%" % (nm, k, n, r))
    cnt = collections.Counter(e["mesh"] for e in E)
    sc = collections.Counter(e["mesh"] for e in E if e["succ"])
    print("      per-anatomy:", "  ".join("%s %d/%d" % (m.replace("topcowmr", "mr"), sc[m], cnt[m]) for m in sorted(cnt)))
    past = [e for e in E if e["path_len"] > SEAM_PL]
    print("      past-seam share %.1f%%  (%d/%d)" % (100.0 * len(past) / len(E), len(past), len(E)))

print()
print("=== 2. REACHABILITY EXCLUSION applied to each arm ===")
for thr in ("wire_0.18", "sofa_0.30", "cath_0.35"):
    print("  threshold %s:" % thr)
    for nm, E in (("H0", H0), ("TEACHER", TE)):
        ex = [e for e in E if excluded(e, thr)]
        keep = [e for e in E if not excluded(e, thr)]
        k, n, r = rate(keep); k0, n0, r0 = rate(E)
        print("     %-8s excluded %d (%s)  ->  %3d/%3d = %5.1f%%   (was %d/%d = %5.1f%%)"
              % (nm, len(ex),
                 ", ".join("%s pl=%.1f s=%.1f succ=%d" % (e["mesh"].replace("topcowmr", "mr"), e["path_len"], e["s"], e["succ"]) for e in ex) or "-",
                 k, n, r, k0, n0, r0))

print()
print("=== 3. PAST-SEAM (path_len > %.2f mm) TEACHER vs H0, same 4 anatomies ===" % SEAM_PL)
for thr in ("none", "wire_0.18", "sofa_0.30"):
    def sel(E):
        return [e for e in E if e["path_len"] > SEAM_PL and (thr == "none" or not excluded(e, thr))]
    th, hh = sel(TE), sel(H0)
    kt, nt, rt = rate(th); kh, nh, rh = rate(hh)
    p = fisher(kt, nt - kt, kh, nh - kh)
    print("  correction=%-10s TEACHER %2d/%2d = %5.1f%%   H0 %2d/%2d = %5.1f%%   diff %+5.1f pp   Fisher p = %.4f"
          % (thr, kt, nt, rt, kh, nh, rh, rt - rh, p))

print()
print("=== 3b. same, PROXIMAL (path_len <= seam) and ALL ===")
for lab, f in (("proximal", lambda e: e["path_len"] <= SEAM_PL), ("all", lambda e: True)):
    th = [e for e in TE if f(e)]; hh = [e for e in H0 if f(e)]
    kt, nt, rt = rate(th); kh, nh, rh = rate(hh)
    print("  %-9s TEACHER %2d/%2d = %5.1f%%   H0 %2d/%2d = %5.1f%%   Fisher p = %.4f"
          % (lab, kt, nt, rt, kh, nh, rh, fisher(kt, nt - kt, kh, nh - kh)))

print()
print("=== 4. STANDARDISED to a common anatomy x depth mix ===")
BANDS = [(0, SEAM_PL, "prox"), (SEAM_PL, 200, "seam-200"), (200, 240, "200-240"), (240, 1e9, ">240")]
def band(pl):
    for lo, hi, nm in BANDS:
        if lo < pl <= hi: return nm
    return "prox"
def strat(E):
    d = collections.defaultdict(list)
    for e in E: d[(e["mesh"], band(e["path_len"]))].append(e)
    return d
ST, SH = strat(TE), strat(H0)
keys = sorted(set(ST) | set(SH))
print("  stratum                nT  succT   nH  succH")
common = []
for k in keys:
    t, h = ST.get(k, []), SH.get(k, [])
    print("   %-12s %-9s %3d %5s  %3d %5s" % (k[0].replace("topcowmr", "mr"), k[1],
          len(t), (sum(e["succ"] for e in t) if t else "-"),
          len(h), (sum(e["succ"] for e in h) if h else "-")))
    if t and h: common.append(k)
print("  strata present in BOTH arms:", len(common), "of", len(keys))

def direct_std(E_strat, common, weights):
    num = den = 0.0
    for k in common:
        e = E_strat[k]; p = sum(x["succ"] for x in e) / len(e)
        num += weights[k] * p; den += weights[k]
    return 100.0 * num / den
W = {k: len(ST[k]) + len(SH[k]) for k in common}
nT = sum(len(ST[k]) for k in common); nH = sum(len(SH[k]) for k in common)
print("  common-stratum episode counts: TEACHER %d, H0 %d (pooled weight n=%d)" % (nT, nH, sum(W.values())))
print("  DIRECT-STANDARDISED (pooled weights, all depths):  TEACHER %5.1f%%   H0 %5.1f%%   diff %+5.1f pp"
      % (direct_std(ST, common, W), direct_std(SH, common, W), direct_std(ST, common, W) - direct_std(SH, common, W)))
cp = [k for k in common if k[1] != "prox"]
if cp:
    Wp = {k: W[k] for k in cp}
    print("  DIRECT-STANDARDISED (past-seam strata only, %d strata): TEACHER %5.1f%%  H0 %5.1f%%  diff %+5.1f pp"
          % (len(cp), direct_std(ST, cp, Wp), direct_std(SH, cp, Wp), direct_std(ST, cp, Wp) - direct_std(SH, cp, Wp)))

# Mantel-Haenszel on past-seam strata (anatomy x band), + MH chi2
def mh(commonk):
    num = den = 0.0; O = 0.0; Ee = 0.0; V = 0.0
    for k in commonk:
        t, h = ST[k], SH[k]
        a = sum(e["succ"] for e in t); b = len(t) - a
        c = sum(e["succ"] for e in h); d = len(h) - c
        n = a + b + c + d
        num += a * d / n; den += b * c / n
        O += a; Ee += (a + b) * (a + c) / n
        if n > 1:
            V += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    orr = num / den if den > 0 else float("inf")
    chi = (abs(O - Ee) - 0.5) ** 2 / V if V > 0 else float("nan")
    from math import erfc, sqrt
    p = erfc(math.sqrt(chi / 2.0)) if chi == chi else float("nan")
    return orr, chi, p
if cp:
    orr, chi, p = mh(cp)
    print("  MANTEL-HAENSZEL (anatomy x depth strata, past-seam): OR = %.3f  chi2_cc = %.3f  p = %.4f" % (orr, chi, p))
orr, chi, p = mh(common)
print("  MANTEL-HAENSZEL (all common strata):                 OR = %.3f  chi2_cc = %.3f  p = %.4f" % (orr, chi, p))

print()
print("=== 5. depth-band breakdown, both arms, 4 anatomies ===")
print("  band        TEACHER            H0")
for lo, hi, nm in BANDS:
    t = [e for e in TE if lo < e["path_len"] <= hi]; h = [e for e in H0 if lo < e["path_len"] <= hi]
    kt, nt, rt = rate(t); kh, nh, rh = rate(h)
    print("  %-10s %3d/%3d = %5.1f%%   %3d/%3d = %5.1f%%" % (nm, kt, nt, rt, kh, nh, rh))
