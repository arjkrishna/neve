"""CHECK 11, part 3 -- DECLARED RADII, independently of the coordinates.

Question from the 49-set: there the radii diverged 31 mm PROXIMAL to the
coordinate seam, because a smoothstep ramp overwrote declared calibre in a
stretch whose COORDINATES were byte-identical to the host. Does either seam of
the 216 do the same, and by how much?

Both comparisons here are legitimate on an arclength grid: proximal to each
seam the two curves being compared are resamplings of the SAME polyline, and
the point-to-curve floor measured in check11_geom_arjk.py is 0.01-0.10 mm.
"""
import json
import os
import numpy as np
from collections import defaultdict

ANAT = "carotid_data/anatomies"
HOSTD = "eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA = "Centerline curve - RCCA.mrk.json"
DS = 0.05


def read_curve(path):
    m = json.load(open(path))["markups"][0]
    P = np.array([c["position"] for c in m["controlPoints"]], float)
    R = None
    for meas in m.get("measurements", []):
        if meas["name"] == "Radius" and meas.get("controlPointValues"):
            R = np.array(meas["controlPointValues"], float)
    return P, R


def arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def summ(vals, f="%7.3f"):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return "none"
    fmt = "min " + f + "  p05 " + f + "  med " + f + "  p95 " + f + "  max " + f + "  n=%d"
    return fmt % (a.min(), np.percentile(a, 5), np.median(a),
                  np.percentile(a, 95), a.max(), a.size)


hp, hr = read_curve(os.path.join(HOSTD, RCCA))
hs = arclen(hp)
names = sorted(n for n in os.listdir(ANAT)
               if os.path.isdir(os.path.join(ANAT, n)))
cur, prov = {}, {}
for n in names:
    cur[n] = read_curve(os.path.join(ANAT, n, "Centrelines_comb", RCCA))
    prov[n] = json.load(open(os.path.join(ANAT, n, "provenance.json")))

print("ostium radius r(s=0): %d distinct value(s), %s"
      % (len(set(round(cur[n][1][0], 9) for n in names)),
         sorted(set(round(cur[n][1][0], 6) for n in names))[:3]))
print("host r(s=0) = %.6f" % hr[0])

# ---------- SEAM 1: composite radius vs HOST radius ----------
print()
print("=== SEAM 1: |r_composite - r_host| by arclength band (mm of radius) ===")
bands = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)]
tab = defaultdict(list)
lead = []
maxprox = []
for n in names:
    P, R = cur[n]
    s = arclen(P)
    hc = prov[n]["host_cut_mm"]
    g = np.arange(0.0, min(hs[-1], s[-1]), DS)
    dr = np.abs(np.interp(g, s, R) - np.interp(g, hs, hr))
    for lo, hi in bands:
        m = (g >= lo) & (g < hi)
        if m.any():
            tab[(lo, hi)].append(float(dr[m].max()))
    m = g <= hc
    maxprox.append(float(dr[m].max()))
    idx = np.where(dr > 0.02)[0]
    hold = int(1.0 / DS)
    st = None
    for i in idx:
        if i + hold < len(dr) and (dr[i:i + hold] > 0.02).all():
            st = float(g[i]); break
    lead.append(None if st is None else hc - st)
for lo, hi in bands:
    print("  %3d-%3d mm : %s" % (lo, hi, summ(tab[(lo, hi)])))
print("  MAX |dr| in the stretch where coordinates are byte-identical to the host"
      " (s <= host_cut): %s" % summ(maxprox))
print("  radius departure is PROXIMAL to the coordinate seam by (mm): %s"
      % summ(lead, "%6.2f"))
print("  provenance blend_mm : %s"
      % summ([prov[n]["blend_mm"] for n in names], "%6.2f"))

# ---------- SEAM 2: shared-lower pairs ----------
print()
print("=== SEAM 2: |r_A - r_B| for pairs sharing a LOWER donor, by band ===")
idx = defaultdict(list)
for n in names:
    idx[n.split("__")[0]].append(n)
bands2 = [(0, 60), (60, 90), (90, 100), (100, 105), (105, 110),
          (110, 120), (120, 130), (130, 140)]
tab2 = defaultdict(list)
lead2, maxprox2, first2 = [], [], []
early = []
for k, mem in sorted(idx.items()):
    mem = sorted(mem)
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            a, b = mem[i], mem[j]
            Pa, Ra = cur[a]
            Pb, Rb = cur[b]
            sa, sb = arclen(Pa), arclen(Pb)
            g = np.arange(0.0, min(sa[-1], sb[-1]), DS)
            dr = np.abs(np.interp(g, sa, Ra) - np.interp(g, sb, Rb))
            for lo, hi in bands2:
                m = (g >= lo) & (g < hi)
                if m.any():
                    tab2[(lo, hi)].append(float(dr[m].max()))
            m = g <= 130.0
            maxprox2.append(float(dr[m].max()))
            hold = int(1.0 / DS)
            st = None
            for q in np.where(dr > 0.02)[0]:
                if q + hold < len(dr) and (dr[q:q + hold] > 0.02).all():
                    st = float(g[q]); break
            first2.append(st)
            if st is not None:
                lead2.append(130.0 - st)
                if st < 104.0:
                    early.append((st, k, a.split("__")[1], b.split("__")[1],
                                  float(dr[g < 104.0].max())))
for lo, hi in bands2:
    print("  %3d-%3d mm : %s" % (lo, hi, summ(tab2[(lo, hi)])))
print("  first |dr| > 0.02 mm at s = %s" % summ(first2, "%6.2f"))
print("  PROXIMAL to the 130 mm coordinate seam by (mm): %s" % summ(lead2, "%6.2f"))
print("  MAX |dr| where the coordinates are identical (s <= 130): %s"
      % summ(maxprox2))
print("  pairs whose radii differ before s=104 mm: %d of %d"
      % (len(early), len(first2)))
for e in sorted(early)[:8]:
    print("     s=%6.2f  %-22s %-14s vs %-14s  max|dr| below 104 = %.3f"
          % (e[0], e[1], e[2], e[3], e[4]))
