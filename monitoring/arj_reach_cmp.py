import json, numpy as np
from collections import Counter

THR = ["wire_0.18", "sofa_0.30", "cath_0.35", "sofa_wire_0.48", "sofa_cath_0.65"]


def load(p):
    t = open(p, "r", encoding="utf-8", errors="replace").read()
    return json.loads(t.split("@@@JSON@@@", 1)[1].strip())


A = load(r"d:\Arjun\workspace\neve\monitoring\_arj_reach216b.out")
B = load(r"d:\Arjun\workspace\neve\monitoring\_arj_reach49.out")


def ceilings(R, tag):
    names = sorted(R)
    tot = sum(R[n]["pool_n"] for n in names)
    print("\n--- %s  (%d anatomies, %d pool points, med pool %.0f) ---"
          % (tag, len(names), tot, np.median([R[n]["pool_n"] for n in names])))
    for k in THR:
        raw = np.array([1.0 - R[n][k]["nblk_raw"] / float(R[n]["pool_n"]) for n in names])
        rl = np.array([1.0 - R[n][k]["nblk_real"] / float(R[n]["pool_n"]) for n in names])
        br = sum(R[n][k]["nblk_raw"] for n in names)
        bl = sum(R[n][k]["nblk_real"] for n in names)
        print("  %-15s RAW coh %.4f  per-anat [%.3f..1.000] n<1 %2d | ADJ coh %.4f  per-anat [%.3f..1.000] n<1 %2d"
              % (k, 1 - br / tot, raw.min(), (raw < 1).sum(),
                 1 - bl / tot, rl.min(), (rl < 1).sum()))


ceilings(B, "49-set TopBrain reference (re-measured with identical code)")
ceilings(A, "216-set three-source carotid")

print("\n=== 216: adjudicated per-anatomy ceiling bands (cath_0.35) ===")
names = sorted(A)
rl = np.array([1.0 - A[n]["cath_0.35"]["nblk_real"] / float(A[n]["pool_n"]) for n in names])
for lo, hi in [(1.0, 1.01), (0.99, 1.0), (0.95, 0.99), (0.90, 0.95), (0.80, 0.90), (0.0, 0.80)]:
    m = (rl >= lo) & (rl < hi)
    print("  ceiling [%.2f,%.2f): %3d anatomies" % (lo, hi, m.sum()))
w = np.array([1.0 - A[n]["wire_0.18"]["nblk_real"] / float(A[n]["pool_n"]) for n in names])
print("  wire_0.18 adjudicated: n<1.0 = %d, min %.3f, cohort %.4f"
      % ((w < 1).sum(), w.min(),
         1 - sum(A[n]["wire_0.18"]["nblk_real"] for n in names) /
         float(sum(A[n]["pool_n"] for n in names))))

print("\n=== which SIPHON donors carry the surviving blockages (cath_0.35) ===")
bad = [n for n in names if A[n]["cath_0.35"]["n_mid_real"] > 0]
c = Counter(A[n]["siphon"] for n in bad)
tot_by_siphon = Counter(A[n]["siphon"] for n in names)
for s, k in sorted(c.items(), key=lambda x: -x[1]):
    print("  %-18s %d of %d anatomies using this siphon blocked" % (s, k, tot_by_siphon[s]))
print("  distinct siphon donors implicated: %d of %d" % (len(c), len(tot_by_siphon)))
print("  lower donors implicated: %d of %d"
      % (len(set(A[n]["lower"] for n in bad)), len(set(A[n]["lower"] for n in names))))

print("\n=== SHARING STRUCTURE ===")
lowc = Counter(A[n]["lower"] for n in names)
sipc = Counter(A[n]["siphon"] for n in names)
print("  distinct lower donors %d (group sizes %s)"
      % (len(lowc), dict(Counter(lowc.values()))))
print("  distinct siphon donors %d (group sizes %s)"
      % (len(sipc), dict(Counter(sipc.values()))))
s1 = np.array([A[n]["seam1_m_1mm"] for n in names])
s2 = np.array([A[n]["seam2_m_1mm"] for n in names])
tot = sum(A[n]["pool_n"] for n in names)


def frac_below(cut_fn):
    k = 0
    for n in names:
        ps = np.array(A[n]["pool_s"])
        k += int((ps < cut_fn(n)).sum())
    return k, 100.0 * k / tot


print("  pool proximal to seam1 (shared with HOST, 1 geometry)      : %d = %.2f%%"
      % frac_below(lambda n: A[n]["seam1_m_1mm"]))
print("  pool proximal to seam1_0.1mm (byte-identical to host)      : %d = %.2f%%"
      % frac_below(lambda n: A[n]["seam1_m_01mm"]))
print("  pool proximal to seam2 (shared with 1-4 siblings, 47 geoms): %d = %.2f%%"
      % frac_below(lambda n: A[n]["seam2_m_1mm"]))
print("  pool distal to seam2   (siphon-determined, 44 geoms)       : %d = %.2f%%"
      % (tot - frac_below(lambda n: A[n]["seam2_m_1mm"])[0],
         100 - frac_below(lambda n: A[n]["seam2_m_1mm"])[1]))

print("\n=== exact min_arclength requirements ===")
print("  max seam1 = %.2f mm  (p95 %.2f, med %.2f, min %.2f)"
      % (s1.max(), np.percentile(s1, 95), np.median(s1), s1.min()))
print("  max seam2 = %.2f mm  (p95 %.2f, med %.2f, min %.2f)"
      % (s2.max(), np.percentile(s2, 95), np.median(s2), s2.min()))
for MA in [40.0, 65.5, 66.0, 140.75, 141.0]:
    keep = np.array([int((np.array(A[n]["pool_s"]) >= MA).sum()) for n in names])
    bad1 = sum(1 for n in names if ((np.array(A[n]["pool_s"]) >= MA) &
                                    (np.array(A[n]["pool_s"]) < A[n]["seam1_m_1mm"])).any())
    bad2 = sum(1 for n in names if ((np.array(A[n]["pool_s"]) >= MA) &
                                    (np.array(A[n]["pool_s"]) < A[n]["seam2_m_1mm"])).any())
    blk = 0
    tp = 0
    for n in names:
        ps = np.array(A[n]["pool_s"])
        ps = ps[ps >= MA]
        sb = A[n]["cath_0.35"]["s_block_real"]
        tp += len(ps)
        blk += int((ps >= sb).sum()) if sb is not None else 0
    print("  min_arc %6.2f : pool %5d (med %3.0f min %3d)  anatomies still sampling pre-seam1 %d / pre-seam2 %d  adj-ceiling %.4f"
          % (MA, keep.sum(), np.median(keep), keep.min(), bad1, bad2,
             1 - blk / float(max(tp, 1))))

print("\n=== 49-set seam reproduction (method validation) ===")
bn = sorted(B)
print("  seam1 (vs host, >1mm): %s" % sorted(set(round(B[n]["seam1_m_1mm"], 2) for n in bn)))
print("  seam1 (>0.1mm)       : %s" % sorted(set(round(B[n]["seam1_m_01mm"], 2) for n in bn)))
print("  pool proximal to seam1: %.2f%%"
      % (100.0 * sum(int((np.array(B[n]["pool_s"]) < B[n]["seam1_m_1mm"]).sum()) for n in bn)
         / sum(B[n]["pool_n"] for n in bn)))
