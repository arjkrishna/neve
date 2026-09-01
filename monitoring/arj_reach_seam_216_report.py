import json, os, numpy as np

OUT = r"d:\Arjun\workspace\neve\monitoring\_arj_reach216.out"
txt = open(OUT, "r", encoding="utf-8", errors="replace").read()
J = txt.split("@@@JSON@@@", 1)[1].strip()
R = json.loads(J)
names = sorted(R)
print("N anatomies %d" % len(names))

THR = ["wire_0.18", "sofa_0.30", "cath_0.35", "sofa_wire_0.48", "sofa_cath_0.65"]


def q(a, label, fmt="%.3f"):
    a = np.asarray(a, float)
    print(("  %-28s n=%3d  min " + fmt + "  p05 " + fmt + "  med " + fmt +
           "  p95 " + fmt + "  max " + fmt + "  mean " + fmt)
          % (label, len(a), a.min(), np.percentile(a, 5), np.median(a),
             np.percentile(a, 95), a.max(), a.mean()))


print("\n=== SANITY / SIGN CONTROL ===")
print("  flipped anatomies: %d of %d" % (sum(R[n]["flipped"] for n in names), len(names)))
ctrls = [(n, R[n]["ctrl"]) for n in names if R[n]["ctrl"]]
for n, c in ctrls:
    print("  control %-40s sd>0 %.3f enc %.3f agree %.2f%%"
          % (n, c["frac_sd_pos"], c["frac_enclosed"], c["agree_pct"]))
q([R[n]["min_clear"] for n in names], "min clearance (mm)")
q([R[n]["med_clear"] for n in names], "median clearance (mm)")
nout = np.array([R[n]["n_outside"] for n in names])
print("  anatomies with >=1 densified pt outside wall: %d ; total pts %d ; max/anat %d"
      % ((nout > 0).sum(), nout.sum(), nout.max()))
q([R[n]["L"] for n in names], "RCCA length (mm)", "%.1f")
q([R[n]["nstat"] for n in names], "RCCA stations", "%.0f")

print("\n=== POOL (CenterlineRandom threshold=5, branches=[RCCA], min_arc=40) ===")
pool_n = np.array([R[n]["pool_n"] for n in names])
q(pool_n, "pool size (stations)", "%.0f")
q([R[n]["n_ge40"] for n in names], "stations with s>=40", "%.0f")
q([R[n]["n_dropped_excl"] for n in names], "dropped by excluded-branch mask", "%.0f")
print("  total pool points across cohort: %d" % pool_n.sum())
q([min(R[n]["pool_s"]) for n in names], "pool s_min (mm)", "%.2f")
q([max(R[n]["pool_s"]) for n in names], "pool s_max (mm)", "%.2f")

print("\n=== CHECK 7  REACHABILITY CEILING ===")
for k in THR:
    ceil = np.array([R[n][k]["ceiling"] for n in names], float)
    nmid = np.array([R[n][k]["n_mid"] for n in names])
    blk = np.array([R[n][k]["n_pool_blocked"] for n in names])
    cohort = 1.0 - blk.sum() / float(pool_n.sum())
    print("\n  threshold %s  (device half-width / gap, mm)" % k)
    print("    anatomies with >=1 MID-VESSEL blockage : %d / %d" % ((nmid > 0).sum(), len(names)))
    print("    anatomies with ceiling == 100%%         : %d" % (ceil >= 1.0).sum())
    print("    ceiling  min %.4f  p05 %.4f  med %.4f  mean %.4f"
          % (ceil.min(), np.percentile(ceil, 5), np.median(ceil), ceil.mean()))
    print("    ceiling <0.99 : %d   <0.90 : %d   <0.50 : %d   ==0 : %d"
          % ((ceil < 0.99).sum(), (ceil < 0.90).sum(), (ceil < 0.50).sum(), (ceil == 0).sum()))
    print("    POOL-WEIGHTED COHORT CEILING = %.4f  (%d of %d pool points unreachable)"
          % (cohort, blk.sum(), pool_n.sum()))
    worst = sorted(names, key=lambda n: R[n][k]["ceiling"])[:8]
    for n in worst:
        r = R[n][k]
        if r["ceiling"] >= 1.0:
            break
        mr = r["mid_runs"][0] if r["mid_runs"] else None
        print("      %-46s ceil %.3f  s_block %.1f  nblk %3d/%3d  worst_run s%.1f-%.1f min_d %.3f"
              % (n, r["ceiling"], r["s_block"], r["n_pool_blocked"], R[n]["pool_n"],
                 mr["s0"], mr["s1"], mr["min_d"]))

print("\n  --- where mid-vessel blockages sit (cath_0.35) ---")
sb = [R[n]["cath_0.35"]["s_block"] for n in names if R[n]["cath_0.35"]["s_block"] is not None]
if sb:
    q(sb, "s_block (mm)", "%.1f")
    print("    s_block < 40 mm (proximal to pool): %d" % sum(1 for x in sb if x < 40))

print("\n=== CHECK 12  SEAMS (measured) ===")
s1 = np.array([R[n]["seam1_1mm"] if R[n]["seam1_1mm"] is not None else 0.0 for n in names])
s1a = np.array([R[n]["seam1_05mm"] if R[n]["seam1_05mm"] is not None else 0.0 for n in names])
s1b = np.array([R[n]["seam1_01mm"] if R[n]["seam1_01mm"] is not None else 0.0 for n in names])
s2 = np.array([R[n]["seam2_1mm"] if R[n].get("seam2_1mm") is not None else np.nan for n in names])
q(s1, "SEAM1 vs host, >1.0 mm (mm)", "%.2f")
q(s1a, "SEAM1 vs host, >0.5 mm (mm)", "%.2f")
q(s1b, "SEAM1 vs host, >0.1 mm (mm)", "%.2f")
ok2 = ~np.isnan(s2)
q(s2[ok2], "SEAM2 vs siblings, >1.0mm (mm)", "%.2f")
print("  anatomies with no sibling (singleton lower): %d" % (~ok2).sum())
print("  min seam1 across cohort = %.2f  (cohort-shared host prefix)" % s1.min())
print("  max seam1 = %.2f   max seam2 = %.2f" % (s1.max(), np.nanmax(s2)))
lo5 = sorted(names, key=lambda n: R[n]["seam1_1mm"])[:6]
print("  earliest seam1:")
for n in lo5:
    print("    %-46s seam1 %.2f  seam2 %s" % (n, R[n]["seam1_1mm"], R[n].get("seam2_1mm")))
hi5 = sorted(names, key=lambda n: -R[n]["seam1_1mm"])[:6]
print("  latest seam1:")
for n in hi5:
    print("    %-46s seam1 %.2f  seam2 %s" % (n, R[n]["seam1_1mm"], R[n].get("seam2_1mm")))
hi2 = sorted([n for n in names if R[n].get("seam2_1mm") is not None],
             key=lambda n: -R[n]["seam2_1mm"])[:6]
print("  latest seam2:")
for n in hi2:
    print("    %-46s seam2 %.2f (n_sib %d, med %.2f)  seam1 %.2f"
          % (n, R[n]["seam2_1mm"], R[n]["seam2_n_sib"], R[n]["seam2_med"], R[n]["seam1_1mm"]))

print("\n=== CHECK 12  SHARED-COURSE FRACTION OF THE TARGET POOL ===")
fr1 = []
fr1_glob = []
fr2 = []
for n in names:
    ps = np.array(R[n]["pool_s"])
    a = R[n]["seam1_1mm"]
    fr1.append((ps < a).mean())
    fr1_glob.append((ps < s1.min()).mean())
    b = R[n].get("seam2_1mm")
    fr2.append((ps < b).mean() if b is not None else np.nan)
fr1 = np.array(fr1); fr2 = np.array(fr2); fr1_glob = np.array(fr1_glob)
q(100 * fr1, "%% pool proximal to OWN seam1", "%.2f")
q(100 * fr1_glob, "%% pool proximal to cohort-min seam1", "%.2f")
q(100 * fr2[~np.isnan(fr2)], "%% pool proximal to OWN seam2", "%.2f")
tot = pool_n.sum()
n1 = sum(int((np.array(R[n]["pool_s"]) < R[n]["seam1_1mm"]).sum()) for n in names)
n2 = sum(int((np.array(R[n]["pool_s"]) < R[n]["seam2_1mm"]).sum())
         for n in names if R[n].get("seam2_1mm") is not None)
tot2 = sum(R[n]["pool_n"] for n in names if R[n].get("seam2_1mm") is not None)
print("  POOL-WEIGHTED: %d/%d = %.2f%% of all admissible targets are PROXIMAL to seam1 (host geometry)"
      % (n1, tot, 100.0 * n1 / tot))
print("  POOL-WEIGHTED: %d/%d = %.2f%% are proximal to seam2 (host arch + lower donor)"
      % (n2, tot2, 100.0 * n2 / tot2))
print("  anatomies with ZERO pool points proximal to seam1: %d" % (fr1 == 0).sum())

print("\n=== CHECK 12  REQUIRED min_arclength_from_start ===")
print("  to force EVERY target past seam1: min_arc = max_a seam1 = %.2f mm" % s1.max())
print("  to force EVERY target past seam2: min_arc = max_a seam2 = %.2f mm" % np.nanmax(s2))
for MA in [40.0, 50.0, 60.0, float(np.ceil(s1.max())), 100.0, 120.0, 130.0,
           140.0, float(np.ceil(np.nanmax(s2))), 160.0]:
    keep = []
    empt = 0
    past1 = 0
    past2 = 0
    for n in names:
        ps = np.array(R[n]["pool_s"])
        k = (ps >= MA).sum()
        keep.append(k)
        if k == 0:
            empt += 1
        past1 += int(((ps >= MA) & (ps >= R[n]["seam1_1mm"])).sum())
        b = R[n].get("seam2_1mm")
        if b is not None:
            past2 += int(((ps >= MA) & (ps >= b)).sum())
    keep = np.array(keep)
    print("  min_arc %6.1f : pool total %5d (med/anat %3.0f, min %3d)  empty-pool anatomies %d  "
          "past-seam1 %5d (%.1f%%)  past-seam2 %5d (%.1f%%)"
          % (MA, keep.sum(), np.median(keep), keep.min(), empt,
             past1, 100.0 * past1 / max(keep.sum(), 1),
             past2, 100.0 * past2 / max(keep.sum(), 1)))

print("\n=== CEILING RECOMPUTED UNDER PAST-SEAM SAMPLERS (cath_0.35) ===")
for MA in [40.0, float(np.ceil(s1.max())), float(np.ceil(np.nanmax(s2)))]:
    tp = 0
    tb = 0
    worst = []
    for n in names:
        ps = np.array(R[n]["pool_s"])
        ps = ps[ps >= MA]
        if len(ps) == 0:
            continue
        sbk = R[n]["cath_0.35"]["s_block"]
        b = int((ps >= sbk).sum()) if sbk is not None else 0
        tp += len(ps)
        tb += b
        worst.append((1.0 - b / float(len(ps)), n))
    worst.sort()
    print("  min_arc %6.1f : cohort ceiling %.4f  (%d/%d blocked)   worst anat %s %.3f"
          % (MA, 1.0 - tb / float(tp), tb, tp, worst[0][1], worst[0][0]))
