#!/usr/bin/env python3
"""H1 part 2: shape deviation from the HOST siphon + non-degenerate OOD metric."""
import os, pickle
import numpy as np

ROOT = "D:/Arjun/workspace/neve"
G = pickle.load(open(os.path.join(ROOT, "monitoring", "_h1_geom.pkl"), "rb"))
host, proc, coh = G["host"], G["proc"], G["coh"]
SEAM = 133.6


def pt2curve(P, C):
    """min distance from each point of P to the polyline C (segment-exact)."""
    A = C[:-1]; B = C[1:]
    AB = B - A
    denom = np.maximum((AB * AB).sum(1), 1e-12)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        t = np.clip(((p - A) * AB).sum(1) / denom, 0.0, 1.0)
        proj = A + t[:, None] * AB
        out[i] = np.linalg.norm(proj - p, axis=1).min()
    return out


hq, ht = host["_q"], host["_t"]
HG = hq[ht >= SEAM]          # host graft-region polyline (1 mm stations)
HL = float(ht[-1])

print("HOST graft region: s %.1f..%.1f  (%d stations)" % (SEAM, HL, len(HG)))
hk = host["_q"]
# where is the host's tightest bend?
from arj_h1_ood_geom import menger, bend_deg, STEN
kap = menger(HG); ben = bend_deg(HG)
s_k = ht[ht >= SEAM][STEN:len(HG) - STEN]
print("HOST graft argmax curvature at s=%.1f (Rc=%.3f mm); argmax bend at s=%.1f (%.1f deg)"
      % (s_k[kap.argmax()], 1.0 / kap.max(), s_k[ben.argmax()], ben.max()))
print("HOST: perturbation w=0 for s>=%.1f (distal_anchor 25 + ramp 15 => full pin from L-25=%.1f,"
      " attenuation begins L-40=%.1f)" % (HL - 25, HL - 25, HL - 40))
print()


def dev_profile(f):
    q, t = f["_q"], f["_t"]
    g = t >= SEAM
    d = pt2curve(q[g], HG)
    return t[g], d


rows = []
for f in proc:
    t, d = dev_profile(f)
    rows.append((f["tag"], float(np.median(d)), float(d.max()),
                 float(np.percentile(d, 95)), float(d[t >= HL - 40].max()),
                 f["tort_param"]))
pm = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print("PROCEDURAL deviation from the HOST curve, graft region (mm)")
print("            median      max      p95   max_over_distal40")
print("  mean   %8.3f %8.3f %8.3f %8.3f" % tuple(pm.mean(0)))
print("  sd     %8.3f %8.3f %8.3f %8.3f" % tuple(pm.std(0, ddof=1)))
print("  min    %8.3f %8.3f %8.3f %8.3f" % tuple(pm.min(0)))
print("  max    %8.3f %8.3f %8.3f %8.3f" % tuple(pm.max(0)))
print()

crow = {}
print("COHORT deviation from the HOST curve, graft region (mm)")
print("%-8s %8s %8s %8s %8s %10s %10s" %
      ("anat", "median", "max", "p95", "d40max", "s@dev>2mm", "s@dev>10mm"))
for f in coh:
    t, d = dev_profile(f)
    L = f["L"]
    i2 = np.argmax(d > 2.0) if (d > 2.0).any() else -1
    i10 = np.argmax(d > 10.0) if (d > 10.0).any() else -1
    crow[f["tag"]] = dict(dev_med=float(np.median(d)), dev_max=float(d.max()),
                          dev_p95=float(np.percentile(d, 95)),
                          dev_d40=float(d[t >= L - 40].max()),
                          s_gt2=float(t[i2]) if i2 >= 0 else float("nan"),
                          s_gt10=float(t[i10]) if i10 >= 0 else float("nan"))
    c = crow[f["tag"]]
    print("%-8s %8.2f %8.2f %8.2f %8.2f %10.1f %10.1f" %
          (f["tag"][-6:], c["dev_med"], c["dev_max"], c["dev_p95"], c["dev_d40"],
           c["s_gt2"], c["s_gt10"]))
pickle.dump(crow, open(os.path.join(ROOT, "monitoring", "_h1_dev.pkl"), "wb"))
print()
print("procedural max-deviation over ALL 208 draws: %.3f mm" % pm[:, 1].max())
print("cohort min median-deviation: %.2f mm (%s)" %
      (min(c["dev_med"] for c in crow.values()),
       min(crow.items(), key=lambda kv: kv[1]["dev_med"])[0]))
