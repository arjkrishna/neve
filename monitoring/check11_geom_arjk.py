"""CHECK 11, part 2 -- parameterization-free seam location.

The arclength-matched metric in check11_seams_arjk.py carries a drift floor:
the composites are a 1.0 mm resampling of the host polyline, and chord
shortcutting makes composite-arclength run slightly short of host-arclength, so
"deviation at equal s" grows tangentially even where the two curves are the SAME
curve. Measured floor there: ~0.25 mm by 45 mm, which is the same size as the
smallest threshold asked for.

This script removes that: exact minimum distance from each NATIVE control point
of curve A to the POLYLINE of curve B (point-to-segment, closed form), so a
point lying on the other curve reads 0 no matter how it is parameterized.
Resolution in s is the native station spacing, ~1.0 mm.

CONTROLS: host points vs host polyline must read 0; a composite vs itself 0.
"""
import json
import os
import numpy as np
from collections import defaultdict

ANAT = "carotid_data/anatomies"
HOSTD = "eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA = "Centerline curve - RCCA.mrk.json"


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


def pt_to_polyline(Q, B):
    """Exact min distance from each row of Q to the polyline through B."""
    A0 = B[:-1]
    D = B[1:] - A0
    dd = np.einsum("ij,ij->i", D, D)
    dd[dd == 0] = 1e-12
    out = np.empty(len(Q))
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]
        w = q[:, None, :] - A0[None, :, :]
        t = np.clip(np.einsum("kij,ij->ki", w, D) / dd, 0.0, 1.0)
        proj = A0[None, :, :] + t[:, :, None] * D[None, :, :]
        out[i:i + 256] = np.linalg.norm(q[:, None, :] - proj, axis=2).min(axis=1)
    return out


def first_at(s, v, thr, hold_pts=2):
    idx = np.where(v > thr)[0]
    for i in idx:
        if i + hold_pts <= len(v) and (v[i:i + hold_pts] > thr).all():
            return float(s[i])
    return float(s[idx[0]]) if len(idx) else None


def summ(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return "none"
    return ("min %7.2f  p05 %7.2f  med %7.2f  p95 %7.2f  max %7.2f  n=%d"
            % (a.min(), np.percentile(a, 5), np.median(a),
               np.percentile(a, 95), a.max(), a.size))


hp, hr = read_curve(os.path.join(HOSTD, RCCA))
hs = arclen(hp)
print("host RCCA %d pts, %.2f mm" % (len(hp), hs[-1]))
print("CONTROL host pts vs host polyline: max %.3e mm"
      % pt_to_polyline(hp, hp).max())

names = sorted(n for n in os.listdir(ANAT)
               if os.path.isdir(os.path.join(ANAT, n)))
cur, prov = {}, {}
for n in names:
    cur[n] = read_curve(os.path.join(ANAT, n, "Centrelines_comb", RCCA))
    prov[n] = json.load(open(os.path.join(ANAT, n, "provenance.json")))
print("CONTROL composite vs itself: max %.3e mm"
      % pt_to_polyline(cur[names[0]][0], cur[names[0]][0]).max())

rows = []
for n in names:
    P, R = cur[n]
    s = arclen(P)
    d = pt_to_polyline(P, hp)
    r = dict(name=n)
    for t in (0.25, 1.0, 2.0):
        r["c%s" % t] = first_at(s, d, t)
    hc = prov[n]["host_cut_mm"]
    m = s <= hc
    r["floor_below_hostcut"] = float(d[m].max()) if m.any() else None
    r["n_below"] = int(m.sum())
    rows.append(r)

print()
print("=== SEAM 1 (host arch -> lower), point-to-curve, mm arclength ===")
for t in (0.25, 1.0, 2.0):
    print("  first > %4.2f mm : %s" % (t, summ([r["c%s" % t] for r in rows])))
print("  provenance host_cut_mm : %s"
      % summ([prov[r["name"]]["host_cut_mm"] for r in rows]))
print("  (measured 0.25) - host_cut_mm : %s"
      % summ([r["c0.25"] - prov[r["name"]]["host_cut_mm"] for r in rows
              if r["c0.25"] is not None]))
print("  TRUE NOISE FLOOR, max dist at s <= host_cut : %s"
      % summ([r["floor_below_hostcut"] for r in rows]))

idx = defaultdict(list)
for r in rows:
    idx[r["name"].split("__")[0]].append(r["name"])
pairs = []
for k, mem in sorted(idx.items()):
    mem = sorted(mem)
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            a, b = mem[i], mem[j]
            Pa, Ra = cur[a]
            Pb, Rb = cur[b]
            sa = arclen(Pa)
            d = pt_to_polyline(Pa, Pb)
            rec = dict(key=k, a=a, b=b)
            for t in (0.25, 1.0, 2.0):
                rec["c%s" % t] = first_at(sa, d, t)
            nom = (prov[a]["host_cut_mm"] + prov[a]["cca_mm"] + prov[a]["ica_mm"])
            rec["nom"] = nom
            m = sa <= nom - 2.0
            rec["floor"] = float(d[m].max()) if m.any() else None
            pairs.append(rec)

print()
print("=== SEAM 2 (lower -> siphon), shared-lower pairs, point-to-curve ===")
print("  %d groups, %d pairs" % (len(idx), len(pairs)))
for t in (0.25, 1.0, 2.0):
    print("  first > %4.2f mm : %s" % (t, summ([p["c%s" % t] for p in pairs])))
print("  nominal host_cut+cca+ica : %s" % summ([p["nom"] for p in pairs]))
print("  (measured 0.25) - nominal : %s"
      % summ([p["c0.25"] - p["nom"] for p in pairs if p["c0.25"] is not None]))
print("  TRUE NOISE FLOOR, max dist at s <= nominal-2 : %s"
      % summ([p["floor"] for p in pairs]))

per_group = defaultdict(list)
for p in pairs:
    if p["c0.25"] is not None:
        per_group[p["key"]].append(p["c0.25"])
spread = [max(v) - min(v) for v in per_group.values() if len(v) > 1]
print("  within-group spread of the seam-2 crossing : %s" % summ(spread))

print()
print("=== seam-2 outliers (latest 0.25 crossings) ===")
for p in sorted(pairs, key=lambda p: -(p["c0.25"] or 0))[:6]:
    print("  %-24s %-14s vs %-14s  c025 %6.2f  c1 %6.2f  nominal %6.2f"
          % (p["key"], p["a"].split("__")[1], p["b"].split("__")[1],
             p["c0.25"], p["c1.0"] or -1, p["nom"]))
