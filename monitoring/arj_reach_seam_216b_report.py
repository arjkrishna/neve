import json, numpy as np

txt = open(r"d:\Arjun\workspace\neve\monitoring\_arj_reach216b.out", "r",
           encoding="utf-8", errors="replace").read()
R = json.loads(txt.split("@@@JSON@@@", 1)[1].strip())
names = sorted(R)
THR = ["wire_0.18", "sofa_0.30", "cath_0.35", "sofa_wire_0.48", "sofa_cath_0.65"]
pool_n = np.array([R[n]["pool_n"] for n in names])
TOT = int(pool_n.sum())


def q(a, label, fmt="%.2f"):
    a = np.asarray(a, float)
    print(("  %-34s n=%3d min " + fmt + " p05 " + fmt + " med " + fmt +
           " p95 " + fmt + " max " + fmt) % (label, len(a), a.min(),
          np.percentile(a, 5), np.median(a), np.percentile(a, 95), a.max()))


print("=== SEAM VALIDATION (matched arclength) ===")
q([R[n]["seam1_pre_max"] for n in names], "pre-seam1 max deviation (mm)", "%.5f")
s1 = np.array([R[n]["seam1_m_1mm"] for n in names], float)
s1a = np.array([R[n]["seam1_m_01mm"] for n in names], float)
s2 = np.array([R[n]["seam2_m_1mm"] for n in names], float)
s2m = np.array([R[n]["seam2_m_med"] for n in names], float)
bif = np.array([R[n]["s_bif"] for n in names], float)
q(s1, "SEAM1 first >1.0mm from host")
q(s1a, "SEAM1 first >0.1mm from host")
q(s2, "SEAM2 first >1.0mm from sibling(min)")
q(s2m, "SEAM2 (median over siblings)")
q(bif, "RCCA/RECA bifurcation s (mm)")
q([R[n]["reca_min_gap"] for n in names], "min RCCA-RECA centerline gap", "%.3f")
print("  n distinct seam1 values: %d ; n distinct lowers: %d"
      % (len(set(np.round(s1, 2))), len(set(R[n]["lower"] for n in names))))
print("  seam1 spread WITHIN a lower donor (max-min), max over donors: ", end="")
g = {}
for n in names:
    g.setdefault(R[n]["lower"], []).append(R[n]["seam1_m_1mm"])
print("%.2f mm" % max(max(v) - min(v) for v in g.values()))

print("\n=== CHECK 7  CEILING ===")
for k in THR:
    raw = np.array([1.0 - R[n][k]["nblk_raw"] / float(R[n]["pool_n"]) for n in names])
    real = np.array([1.0 - R[n][k]["nblk_real"] / float(R[n]["pool_n"]) for n in names])
    br = sum(R[n][k]["nblk_raw"] for n in names)
    bl = sum(R[n][k]["nblk_real"] for n in names)
    print("\n  %s" % k)
    print("    RAW (centerline-anchored, same estimator as the 49-set):")
    print("      anatomies w/ mid-vessel run %3d/216 | ceiling min %.3f p05 %.3f med %.3f mean %.4f | ==1.0 %d"
          % (sum(1 for n in names if R[n][k]["n_mid"]), raw.min(),
             np.percentile(raw, 5), np.median(raw), raw.mean(), (raw >= 1).sum()))
    print("      POOL-WEIGHTED COHORT CEILING %.4f  (%d/%d unreachable)" % (1 - br / TOT, br, TOT))
    print("    ADJUDICATED (re-centred / maximal inscribed sphere):")
    print("      anatomies w/ real blockage  %3d/216 | ceiling min %.3f p05 %.3f med %.3f mean %.4f | ==1.0 %d"
          % (sum(1 for n in names if R[n][k]["n_mid_real"]), real.min(),
             np.percentile(real, 5), np.median(real), real.mean(), (real >= 1).sum()))
    print("      POOL-WEIGHTED COHORT CEILING %.4f  (%d/%d unreachable)" % (1 - bl / TOT, bl, TOT))
    if k == "cath_0.35":
        w = sorted(names, key=lambda n: 1.0 - R[n][k]["nblk_real"] / float(R[n]["pool_n"]))
        print("      worst anatomies (adjudicated):")
        for n in w[:12]:
            c = 1.0 - R[n][k]["nblk_real"] / float(R[n]["pool_n"])
            if c >= 1.0:
                break
            rr = [x for x in R[n][k]["runs"] if x["still_blocked"]]
            x = rr[0]
            print("        %-46s ceil %.3f s_block %6.1f (bif+%5.1f) blk %3d/%3d  min_d %.3f -> recentred %.3f  declr %.2f"
                  % (n, c, x["s0"], x["d_from_bif"], R[n][k]["nblk_real"],
                     R[n]["pool_n"], x["min_d"], x["recentred_min"], x["decl_r"]))

print("\n=== BLOCKAGE ADJUDICATION DETAIL (cath_0.35) ===")
allr = []
for n in names:
    for x in R[n]["cath_0.35"]["runs"]:
        allr.append((n, x))
print("  raw mid-vessel runs %d across %d anatomies"
      % (len(allr), sum(1 for n in names if R[n]["cath_0.35"]["n_mid"])))
surv = [x for _, x in allr if x["still_blocked"]]
gone = [x for _, x in allr if not x["still_blocked"]]
print("  survive re-centring: %d ; dissolve: %d" % (len(surv), len(gone)))
prox = [(n, x) for n, x in allr if x["d_from_bif"] is not None and -5 < x["d_from_bif"] < 25]
print("  runs within [bif-5, bif+25] mm: %d ; of those surviving: %d"
      % (len(prox), sum(1 for _, x in prox if x["still_blocked"])))
if prox:
    q([x["d_from_bif"] for _, x in prox], "  their s_block - s_bif (mm)")
    q([x["min_d"] for _, x in prox], "  their raw min_d (mm)", "%.3f")
    q([x["recentred_min"] for _, x in prox], "  their recentred min_d (mm)", "%.3f")
    q([x["decl_r"] for _, x in prox], "  their declared radius (mm)", "%.3f")
    q([x["max_off"] for _, x in prox], "  re-centring offset used (mm)", "%.3f")
dist = [(n, x) for n, x in allr if x["d_from_bif"] is None or x["d_from_bif"] >= 25]
print("  runs distal (>bif+25mm): %d ; surviving: %d"
      % (len(dist), sum(1 for _, x in dist if x["still_blocked"])))
if dist:
    q([x["s0"] for _, x in dist], "  their s0 (mm)", "%.1f")
    q([x["min_d"] for _, x in dist], "  their raw min_d (mm)", "%.3f")
    q([x["recentred_min"] for _, x in dist], "  their recentred min_d (mm)", "%.3f")
rej = sum(x["reca_rejects"] for _, x in allr)
print("  re-centred candidates rejected for drifting toward RECA: %d" % rej)

print("\n=== CHECK 12  SHARED-COURSE FRACTION ===")
n1 = n2 = 0
f1 = []
f2 = []
for n in names:
    ps = np.array(R[n]["pool_s"])
    a = R[n]["seam1_m_1mm"]
    b = R[n]["seam2_m_1mm"]
    n1 += int((ps < a).sum())
    n2 += int((ps < b).sum())
    f1.append(100.0 * (ps < a).mean())
    f2.append(100.0 * (ps < b).mean())
q(f1, "% pool proximal to own seam1")
q(f2, "% pool proximal to own seam2")
print("  POOL-WEIGHTED  proximal to SEAM1 : %d/%d = %.2f%%" % (n1, TOT, 100.0 * n1 / TOT))
print("  POOL-WEIGHTED  proximal to SEAM2 : %d/%d = %.2f%%" % (n2, TOT, 100.0 * n2 / TOT))
print("  anatomies with 0%% proximal to seam1: %d" % sum(1 for x in f1 if x == 0))
print("  cohort-min seam1 %.2f -> pool points below it: %d"
      % (s1.min(), sum(int((np.array(R[n]["pool_s"]) < s1.min()).sum()) for n in names)))

print("\n=== CHECK 12  REQUIRED min_arclength_from_start ===")
print("  past EVERY seam1 : max_a seam1 = %.2f mm" % s1.max())
print("  past EVERY seam2 : max_a seam2 = %.2f mm" % s2.max())
print("  (95th pct seam1 %.2f, seam2 %.2f)" % (np.percentile(s1, 95), np.percentile(s2, 95)))
for MA in [40.0, 50.0, 63.0, 64.0, 100.0, 130.0, 140.0, 142.0, 150.0]:
    keep = np.array([int((np.array(R[n]["pool_s"]) >= MA).sum()) for n in names])
    p1 = sum(int(((np.array(R[n]["pool_s"]) >= MA) &
                  (np.array(R[n]["pool_s"]) >= R[n]["seam1_m_1mm"])).sum()) for n in names)
    p2 = sum(int(((np.array(R[n]["pool_s"]) >= MA) &
                  (np.array(R[n]["pool_s"]) >= R[n]["seam2_m_1mm"])).sum()) for n in names)
    blk = 0
    tp = 0
    for n in names:
        ps = np.array(R[n]["pool_s"])
        ps = ps[ps >= MA]
        sb = R[n]["cath_0.35"]["s_block_real"]
        tp += len(ps)
        blk += int((ps >= sb).sum()) if sb is not None else 0
    print("  min_arc %5.1f: pool %5d (med %3.0f, min %3d, empty %d)  past-seam1 %.1f%%  past-seam2 %.1f%%  ceiling %.4f"
          % (MA, keep.sum(), np.median(keep), keep.min(), (keep == 0).sum(),
             100.0 * p1 / max(keep.sum(), 1), 100.0 * p2 / max(keep.sum(), 1),
             1 - blk / float(max(tp, 1))))
