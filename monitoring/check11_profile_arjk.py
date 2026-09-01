"""CHECK 11, part 4 -- where exactly does the radius overwrite open?

Part 3 showed the 0.02 mm threshold sits BELOW the interpolation floor of the
comparison (0.036 mm vs host in the 5-10 mm ostium flare, 0.067 mm between
shared-lower pairs). So this reports the |dr| envelope at fixed arclengths,
where a floor is visible rather than mistaken for a seam, plus a first-crossing
at 0.10 mm which is above both floors.
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


def first_cross(g, v, thr, hold=20):
    for i in np.where(v > thr)[0]:
        if i + hold < len(v) and (v[i:i + hold] > thr).all():
            return float(g[i])
    return None


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

S1 = [2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45]
prof = defaultdict(list)
cross, pred, resid = [], [], []
for n in names:
    P, R = cur[n]
    s = arclen(P)
    hc = prov[n]["host_cut_mm"]
    g = np.arange(0.0, min(hs[-1], s[-1]), DS)
    dr = np.abs(np.interp(g, s, R) - np.interp(g, hs, hr))
    for q in S1:
        prof[q].append(float(np.interp(q, g, dr)))
    c = first_cross(g, dr, 0.10)
    cross.append(c)
    p = max(10.0, hc - 25.0)
    pred.append(p)
    if c is not None:
        resid.append(c - p)
print("=== SEAM 1: |r_comp - r_host| at fixed arclength, across all 216 ===")
for q in S1:
    a = np.array(prof[q])
    print("  s=%4d mm : med %6.3f  p95 %6.3f  max %6.3f" %
          (q, np.median(a), np.percentile(a, 95), a.max()))
print("  first |dr| > 0.10 mm (above the 0.036 mm floor) at s = %s"
      % summ(cross, "%6.2f"))
print("  predicted ramp-window start max(10, host_cut-25) = %s"
      % summ(pred, "%6.2f"))
print("  measured crossing - predicted window start = %s" % summ(resid, "%6.2f"))
print("  lead over the coordinate seam (host_cut - crossing) = %s"
      % summ([prov[n]["host_cut_mm"] - c for n, c in zip(names, cross)
              if c is not None], "%6.2f"))

idx = defaultdict(list)
for n in names:
    idx[n.split("__")[0]].append(n)
S2 = [60, 80, 95, 100, 103, 105, 107, 110, 113, 115, 120, 125, 129]
prof2 = defaultdict(list)
cross2 = []
for k, mem in sorted(idx.items()):
    mem = sorted(mem)
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            Pa, Ra = cur[mem[i]]
            Pb, Rb = cur[mem[j]]
            sa, sb = arclen(Pa), arclen(Pb)
            g = np.arange(0.0, min(sa[-1], sb[-1]), DS)
            dr = np.abs(np.interp(g, sa, Ra) - np.interp(g, sb, Rb))
            for q in S2:
                prof2[q].append(float(np.interp(q, g, dr)))
            cross2.append(first_cross(g, dr, 0.10))
print()
print("=== SEAM 2: |r_A - r_B| at fixed arclength, 399 shared-lower pairs ===")
for q in S2:
    a = np.array(prof2[q])
    print("  s=%4d mm : med %6.3f  p95 %6.3f  max %6.3f" %
          (q, np.median(a), np.percentile(a, 95), a.max()))
print("  first |dr| > 0.10 mm (above the 0.067 mm floor) at s = %s"
      % summ(cross2, "%6.2f"))
print("  lead over the 130 mm coordinate seam = %s"
      % summ([130.0 - c for c in cross2 if c is not None], "%6.2f"))
print("  pairs never crossing 0.10 mm before the coordinate seam: %d of %d"
      % (sum(1 for c in cross2 if c is None or c >= 130.0), len(cross2)))
