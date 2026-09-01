"""CHECK 11 -- the two seams, measured. Arclength-resampled curve comparison.

Coordinates and DECLARED RADII are compared independently: against the HOST's
own RCCA, and (within a shared-lower-donor group) against each other.
No mesh is touched, so traps 11.1/11.2 do not apply. Everything is resampled
onto a 0.05 mm arclength grid before any comparison: the host RCCA opens with a
5.32 mm segment while the composites are uniform ~1.0 mm, so index-wise
comparison is meaningless here (point counts 235 vs 250 as well).
CONTROLS: host-vs-host must read 0; noise floor proximal to each crossing is
reported so a "departure" cannot be a resampling artifact in disguise.
"""
import json
import os
import numpy as np
from collections import defaultdict

ANAT = "carotid_data/anatomies"
HOSTD = "eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA = "Centerline curve - RCCA.mrk.json"
DS = 0.05
HOLD_MM = 1.0   # a crossing must persist this far to count


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


def on_grid(P, R, g):
    s = arclen(P)
    Q = np.column_stack([np.interp(g, s, P[:, i]) for i in range(3)])
    Rg = np.interp(g, s, R) if R is not None else None
    return Q, Rg


def first_cross(g, v, thr):
    hold = int(round(HOLD_MM / DS))
    idx = np.where(v > thr)[0]
    if not len(idx):
        return None
    for i in idx:
        if i + hold >= len(v):
            break
        if (v[i:i + hold] > thr).all():
            return float(g[i])
    return float(g[idx[0]])


def summ(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return "none"
    return ("min %7.2f  p05 %7.2f  med %7.2f  p95 %7.2f  max %7.2f  n=%d"
            % (a.min(), np.percentile(a, 5), np.median(a),
               np.percentile(a, 95), a.max(), a.size))


def measure():
    hp, hr = read_curve(os.path.join(HOSTD, RCCA))
    hL = arclen(hp)[-1]
    names = sorted(n for n in os.listdir(ANAT)
                   if os.path.isdir(os.path.join(ANAT, n)))
    print("anatomies %d   host RCCA len %.2f mm  n=%d" % (len(names), hL, len(hp)))

    g = np.arange(0.0, hL, DS)
    A, RA = on_grid(hp, hr, g)
    B, RB = on_grid(hp, hr, g)
    print("CONTROL host-vs-host: max coord %.3e mm  max radius %.3e mm"
          % (np.abs(A - B).max(), np.abs(RA - RB).max()))

    cur, prov = {}, {}
    for n in names:
        cur[n] = read_curve(os.path.join(ANAT, n, "Centrelines_comb", RCCA))
        prov[n] = json.load(open(os.path.join(ANAT, n, "provenance.json")))

    rows = []
    for n in names:
        P, R = cur[n]
        L = min(hL, arclen(P)[-1])
        g = np.arange(0.0, L, DS)
        Q, Rq = on_grid(P, R, g)
        H, Rh = on_grid(hp, hr, g)
        d = np.linalg.norm(Q - H, axis=1)
        dr = np.abs(Rq - Rh)
        r = dict(name=n, L=float(arclen(P)[-1]), n_pts=len(P))
        for t in (0.25, 1.0, 2.0):
            r["c%s" % t] = first_cross(g, d, t)
        for t in (0.02, 0.05, 0.10, 0.25):
            r["r%s" % t] = first_cross(g, dr, t)
        s1 = r["c0.25"]
        if s1 is not None:
            m = g < s1
            r["coord_noise"] = float(d[m].max()) if m.any() else 0.0
            r["rad_at_c025"] = float(dr[m].max()) if m.any() else 0.0
        rows.append(r)
    return rows, prov, cur


def group_pairs(rows, cur, key):
    idx = defaultdict(list)
    for r in rows:
        lower, siphon = r["name"].split("__")
        idx[lower if key == "lower" else siphon].append(r["name"])
    out = []
    for k, members in sorted(idx.items()):
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                Pa, Ra = cur[a]
                Pb, Rb = cur[b]
                L = min(arclen(Pa)[-1], arclen(Pb)[-1])
                g = np.arange(0.0, L, DS)
                Qa, Rga = on_grid(Pa, Ra, g)
                Qb, Rgb = on_grid(Pb, Rb, g)
                d = np.linalg.norm(Qa - Qb, axis=1)
                dr = np.abs(Rga - Rgb)
                rec = dict(key=k, a=a, b=b)
                for t in (0.25, 1.0, 2.0):
                    rec["c%s" % t] = first_cross(g, d, t)
                for t in (0.02, 0.05, 0.10, 0.25):
                    rec["r%s" % t] = first_cross(g, dr, t)
                s1 = rec["c0.25"]
                if s1 is not None:
                    m = g < s1
                    rec["coord_noise"] = float(d[m].max()) if m.any() else 0.0
                out.append(rec)
    return out, {k: len(v) for k, v in idx.items()}


if __name__ == "__main__":
    rows, prov, cur = measure()
    print()
    print("=== A. COORDINATE DEPARTURE FROM HOST RCCA (seam 1), mm arclength ===")
    for t in (0.25, 1.0, 2.0):
        print("  first > %4.2f mm : %s" % (t, summ([r["c%s" % t] for r in rows])))
    print("  provenance host_cut_mm : %s"
          % summ([prov[r["name"]]["host_cut_mm"] for r in rows]))
    dif = [r["c0.25"] - prov[r["name"]]["host_cut_mm"]
           for r in rows if r["c0.25"] is not None]
    print("  (measured 0.25 crossing) - host_cut_mm : %s" % summ(dif))
    print("  noise floor: max coord dev proximal to crossing : %s"
          % summ([r.get("coord_noise") for r in rows]))
    print("  n with no 0.25 crossing: %d"
          % sum(1 for r in rows if r["c0.25"] is None))

    print()
    print("=== B. DECLARED-RADIUS DEPARTURE FROM HOST RCCA, mm arclength ===")
    for t in (0.02, 0.05, 0.10, 0.25):
        print("  first > %4.2f mm : %s" % (t, summ([r["r%s" % t] for r in rows])))
    lead = [r["c0.25"] - r["r0.02"] for r in rows
            if r["c0.25"] is not None and r["r0.02"] is not None]
    print("  RADIUS LEADS COORDINATE by (coord0.25 - rad0.02) mm : %s" % summ(lead))
    print("  provenance blend_mm : %s"
          % summ([prov[r["name"]]["blend_mm"] for r in rows]))
    print("  max |dr| proximal to the coordinate crossing : %s"
          % summ([r.get("rad_at_c025") for r in rows]))

    for key, label in (("lower", "SEAM 2 -- shared LOWER donor, siphon varies"),
                       ("siphon", "CROSS-CHECK -- shared SIPHON, lower varies")):
        pairs, sizes = group_pairs(rows, cur, key)
        print()
        print("=== %s: %d groups (sizes %s), %d pairs ==="
              % (label, len(sizes), sorted(set(sizes.values())), len(pairs)))
        for t in (0.25, 1.0, 2.0):
            print("  coord  > %4.2f mm : %s" % (t, summ([p["c%s" % t] for p in pairs])))
        for t in (0.02, 0.05, 0.10, 0.25):
            print("  radius > %4.2f mm : %s" % (t, summ([p["r%s" % t] for p in pairs])))
        lead = [p["c0.25"] - p["r0.02"] for p in pairs
                if p["c0.25"] is not None and p["r0.02"] is not None]
        print("  RADIUS LEADS COORDINATE by mm : %s" % summ(lead))
        print("  noise floor proximal : %s"
              % summ([p.get("coord_noise") for p in pairs]))
        if key == "lower":
            ex = sorted(pairs, key=lambda p: (p["c0.25"] is None, p["c0.25"] or 1e9))
            print("  earliest 3 pairs:")
            for p in ex[:3]:
                print("    %-42s | %-14s vs %-14s coord %6.2f rad %6.2f"
                      % (p["key"], p["a"].split("__")[1], p["b"].split("__")[1],
                         p["c0.25"] or -1, p["r0.02"] or -1))
            print("  latest 3 pairs:")
            for p in ex[-3:]:
                print("    %-42s | %-14s vs %-14s coord %6.2f rad %6.2f"
                      % (p["key"], p["a"].split("__")[1], p["b"].split("__")[1],
                         p["c0.25"] or -1, p["r0.02"] or -1))

    print()
    print("=== OUTLIERS on seam 1 (coordinate), 5 earliest / 5 latest ===")
    rs = sorted(rows, key=lambda r: (r["c0.25"] is None, r["c0.25"] or 1e9))
    for r in rs[:5] + rs[-5:]:
        p = prov[r["name"]]
        print("  %-46s c025 %6.2f c1 %6.2f c2 %6.2f r002 %6.2f host_cut %5.1f blend %5.1f"
              % (r["name"], r["c0.25"] or -1, r["c1.0"] or -1, r["c2.0"] or -1,
                 r["r0.02"] or -1, p["host_cut_mm"], p["blend_mm"]))
