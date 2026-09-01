#!/usr/bin/env python3
"""Chirality audit v2 -- DONOR level (the composites are replicates).

host_cut = 130 - (cca + ica) is fixed by the lower donor, so every composite
sharing a lower donor carries a byte-identical lower band up to a proper
rotation. The independent unit is the DONOR (47 lowers, 44 siphons), not the
composite (216).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = "D:/Arjun/workspace/neve/carotid_data/anatomies"
HOST = "D:/Arjun/workspace/neve/eve_bench/data/dualdevicenav/Centrelines_comb"
SEAM2 = 130.0
UP = np.array([0.0, 0.0, 1.0])          # LPS +z = superior


def read_pos(path):
    m = json.load(open(path, encoding="utf-8"))["markups"][0]
    return np.array([cp["position"] for cp in m["controlPoints"]], float)


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample_uniform(p, lo, hi, step):
    s = arclen(p)
    n = max(int(round((hi - lo) / step)) + 1, 4)
    t = np.linspace(lo, hi, n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def dist_to_polyline(q, P):
    """Min distance from each point q to polyline P (segment-wise)."""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2[L2 < 1e-12] = 1e-12
    out = np.empty(len(q))
    for i, x in enumerate(q):
        t = np.clip(np.einsum("ij,ij->i", x - a, ab) / L2, 0, 1)
        proj = a + t[:, None] * ab
        out[i] = np.linalg.norm(proj - x, axis=1).min()
    return out


def integrated_torsion(q):
    e = np.diff(q, axis=0)
    b, ok = [], []
    for i in range(len(e) - 1):
        c = np.cross(e[i], e[i + 1])
        nc = np.linalg.norm(c)
        b.append(c / nc if nc > 1e-9 else np.zeros(3))
        ok.append(nc > 1e-9)
    b, ok = np.array(b), np.array(ok, bool)
    last = None
    for i in range(len(b)):
        if ok[i]:
            last = b[i]
        elif last is not None:
            b[i], ok[i] = last, True
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return 0.0
    tot = 0.0
    for a_, c_ in zip(idx[:-1], idx[1:]):
        b0, b1 = b[a_], b[c_]
        cr = np.cross(b0, b1)
        ang = np.arctan2(np.linalg.norm(cr), float(np.clip(b0 @ b1, -1, 1)))
        tot += ang * (1.0 if float(cr @ unit(e[c_])) >= 0 else -1.0)
    return float(tot)


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return dict(n=0)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                pos=int((v > 0).sum()), lo=float(v.min()), hi=float(v.max()))


def line(tag, d):
    if not d["n"]:
        print("  %-20s n=0")
        return
    print("  %-20s n=%3d mean %+8.3f (se %5.3f) median %+8.3f sd %7.3f "
          "pos %3d/%3d range [%+8.2f,%+8.2f]"
          % (tag, d["n"], d["mean"], d["se"], d["median"], d["sd"], d["pos"],
             d["n"], d["lo"], d["hi"]))


def compare(name, L, R):
    dl, dr = stats(L), stats(R)
    print("")
    print("%s" % name)
    line("LEFT group", dl)
    line("RIGHT group", dr)
    if dl["n"] and dr["n"]:
        a = dl["mean"] - dr["mean"]
        f = -dl["mean"] - dr["mean"]
        print("    mean diff AS-IS   %+8.3f   |  LEFT-SIGN-FLIPPED %+8.3f   -> %s"
              % (a, f, "AS-IS closer (means agree)" if abs(a) < abs(f)
                 else "FLIPPED closer (means are opposite = NOT mirrored)"))
    return dl, dr


def main():
    hp = read_pos(os.path.join(HOST, "Centerline curve - RCCA.mrk.json"))

    # ---------------- estimator self-tests --------------------------------
    tt = np.linspace(0, 6 * np.pi, 400)
    hel = np.stack([np.cos(tt) * 10, np.sin(tt) * 10, tt * 3], 1)
    print("SELF-TEST helix torsion: tau %+.4f  x-reflected %+.4f  z-reflected %+.4f"
          % (integrated_torsion(hel),
             integrated_torsion(hel * [-1., 1, 1]),
             integrated_torsion(hel * [1., 1, -1])))

    # ---------------- load ------------------------------------------------
    recs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        pv = json.load(open(os.path.join(d, "provenance.json"), encoding="utf-8"))
        rp = read_pos(os.path.join(d, "Centrelines_comb", "Centerline curve - RCCA.mrk.json"))
        ep = read_pos(os.path.join(d, "Centrelines_comb", "Centerline curve - RECA.mrk.json"))
        recs.append(dict(name=os.path.basename(d), pv=pv, rp=rp, ep=ep, rs=arclen(rp)))

    # ---------------- seam 1, measured against the host polyline ----------
    print("")
    print("=============== SEAM 1 / BAND ===============")
    meas, prov = [], []
    for r in recs:
        dd = dist_to_polyline(r["rp"], hp)
        k = np.where(dd > 0.10)[0]
        meas.append(float(r["rs"][k[0]]) if len(k) else float(r["rs"][-1]))
        prov.append(r["pv"]["host_cut_mm"])
    meas, prov = np.array(meas), np.array(prov)
    print("provenance host_cut_mm : %.1f - %.1f mm (median %.1f)"
          % (prov.min(), prov.max(), np.median(prov)))
    print("measured departure     : %.1f - %.1f mm (median %.1f)"
          % (meas.min(), meas.max(), np.median(meas)))
    dev = meas - prov
    print("measured - provenance  : median %+.2f mm, mean %+.2f, |max| %.2f, "
          "within 3 mm: %d/216" % (np.median(dev), dev.mean(), np.abs(dev).max(),
                                   int((np.abs(dev) <= 3).sum())))
    band = SEAM2 - prov
    print("LOWER band length      : %.1f - %.1f mm (median %.1f); "
          "n<40mm %d, n<60mm %d, n<70mm %d"
          % (band.min(), band.max(), np.median(band), int((band < 40).sum()),
             int((band < 60).sum()), int((band < 70).sum())))
    tot = np.array([r["rs"][-1] for r in recs])
    print("SIPHON band length     : %.1f - %.1f mm (median %.1f)"
          % ((tot - 130).min(), (tot - 130).max(), np.median(tot - 130)))

    # ---------------- donor-level dedup -----------------------------------
    low_by, sip_by = defaultdict(list), defaultdict(list)
    for r in recs:
        low_by[r["pv"]["lower"]].append(r)
        sip_by[r["pv"]["siphon"]].append(r)
    print("")
    print("replication: %d composites from %d lower donors (%d-%d uses each) and "
          "%d siphons" % (len(recs), len(low_by),
                          min(len(v) for v in low_by.values()),
                          max(len(v) for v in low_by.values()), len(sip_by)))

    # ---------------- A: torsion, lower band, donor level -----------------
    print("")
    print("=============== A. INTEGRATED TORSION, LOWER BAND (donor level) ===============")
    lower_tau = {}
    for step in (2.0, 3.0, 4.0, 6.0, 8.0):
        L, R, spread = [], [], []
        per = {}
        for nm, rs_ in low_by.items():
            vals = []
            for r in rs_:
                lo, hi = r["pv"]["host_cut_mm"], SEAM2
                vals.append(integrated_torsion(resample_uniform(r["rp"], lo, hi, step)))
            v = float(np.mean(vals))
            spread.append(float(np.max(vals) - np.min(vals)))
            per[nm] = v
            (L if "_left" in nm else R).append(v)
        lower_tau[step] = per
        print("")
        print("--- step %.1f mm (within-donor spread of replicates: max %.3f rad) ---"
              % (step, max(spread)))
        compare("lower band torsion", L, R)

    # ---------------- B: siphon control, donor level ----------------------
    print("")
    print("=============== B. CONTROL: SIPHON BAND, KNOWN MIRROR (donor level) ===============")
    sip_tau = {}
    for step in (2.0, 3.0, 4.0, 6.0, 8.0):
        L, R, spread = [], [], []
        per = {}
        for nm, rs_ in sip_by.items():
            vals = [integrated_torsion(resample_uniform(r["rp"], SEAM2, r["rs"][-1], step))
                    for r in rs_]
            v = float(np.mean(vals))
            spread.append(float(np.max(vals) - np.min(vals)))
            per[nm] = v
            (L if nm.endswith("_L") else R).append(v)
        sip_tau[step] = per
        print("")
        print("--- step %.1f mm (within-siphon spread: max %.3f rad) ---" % (step, max(spread)))
        compare("siphon band torsion", L, R)

    # ---------------- paired within-patient (the TopBrain precedent) ------
    print("")
    print("=============== PAIRED WITHIN-PATIENT (same subject, both sides) ===============")
    for step in (3.0, 4.0, 6.0, 8.0):
        per = sip_tau[step]
        pairs = [(k, per[k + "_L"], per[k]) for k in per if k + "_L" in per and not k.endswith("_L")]
        if not pairs:
            continue
        dif = np.array([l - r for _, l, r in pairs])
        sam = np.array([l + r for _, l, r in pairs])
        opp = sum(1 for _, l, r in pairs if l * r < 0)
        print("SIPHON step %.1f: %d paired subjects | opposite-sign L vs R: %d/%d | "
              "mean(L-R) %+.3f  mean(L+R) %+.3f"
              % (step, len(pairs), opp, len(pairs), dif.mean(), sam.mean()))
    for step in (3.0, 4.0, 6.0, 8.0):
        per = lower_tau[step]
        stems = defaultdict(dict)
        for k, v in per.items():
            stems[k.rsplit("_", 1)[0]][k.rsplit("_", 1)[1]] = v
        pairs = [(k, d["left"], d["right"]) for k, d in stems.items() if len(d) == 2]
        if not pairs:
            continue
        dif = np.array([l - r for _, l, r in pairs])
        sam = np.array([l + r for _, l, r in pairs])
        opp = sum(1 for _, l, r in pairs if l * r < 0)
        print("LOWER  step %.1f: %d paired subjects | opposite-sign L vs R: %d/%d | "
              "mean(L-R) %+.3f  mean(L+R) %+.3f"
              % (step, len(pairs), opp, len(pairs), dif.mean(), sam.mean()))

    # ---------------- C/D: fork chirality, no Frenet ----------------------
    print("")
    print("=============== C. FORK CHIRALITY (no Frenet frame) ===============")
    trip, azi, medi = defaultdict(list), defaultdict(list), defaultdict(list)
    within = defaultdict(list)
    for r in recs:
        rp, ep, rs = r["rp"], r["ep"], r["rs"]
        es = arclen(ep)
        k = int(np.argmin(np.linalg.norm(rp - ep[0], axis=1)))
        s_f = float(rs[k])
        d = 10.0
        pm = np.stack([np.interp([s_f - d, s_f, s_f + d], rs, rp[:, c]) for c in range(3)], 1)
        u_cca, u_ica = unit(pm[1] - pm[0]), unit(pm[2] - pm[1])
        pe = np.array([np.interp(min(d, es[-1]), es, ep[:, c]) for c in range(3)])
        u_eca = unit(pe - ep[0])
        t = float(np.dot(np.cross(u_cca, u_ica), u_eca))
        v = unit(u_eca - u_ica * float(u_eca @ u_ica))
        ref = unit(UP - u_ica * float(UP @ u_ica))
        az = float(np.degrees(np.arctan2(float(v @ np.cross(u_ica, ref)), float(v @ ref))))
        within[r["pv"]["lower"]].append((t, az, float(v[0])))
    # donor level
    for nm, vs in within.items():
        side = "left" if "_left" in nm else "right"
        arr = np.array(vs)
        trip[side].append(float(np.mean(arr[:, 0])))
        c, s = np.cos(np.radians(arr[:, 1])), np.sin(np.radians(arr[:, 1]))
        azi[side].append(float(np.degrees(np.arctan2(s.mean(), c.mean()))))
        medi[side].append(float(np.mean(arr[:, 2])))
    sp = max(float(np.array(v)[:, 0].max() - np.array(v)[:, 0].min()) for v in within.values())
    spa = max(float(np.abs(np.diff(np.array(v)[:, 1])).max()) if len(v) > 1 else 0.0
              for v in within.values())
    print("within-donor spread: triple product %.4f, azimuth %.2f deg  "
          "(0 = pure proper rotation, as expected)" % (sp, spa))
    compare("triple product det[u_cca,u_ica,u_eca]", trip["left"], trip["right"])
    compare("ECA azimuth about route tangent, superior ref (deg)", azi["left"], azi["right"])
    compare("ECA perp component along LPS +x (patient left)", medi["left"], medi["right"])

    # sign test on the triple product at donor level
    tl = np.array(trip["left"])
    tr = np.array(trip["right"])
    p1, p2 = (tl > 0).mean(), (tr > 0).mean()
    pp = (int((tl > 0).sum()) + int((tr > 0).sum())) / (len(tl) + len(tr))
    se = np.sqrt(pp * (1 - pp) * (1 / len(tl) + 1 / len(tr)))
    print("")
    print("triple-product sign, DONOR level: left %d/%d positive (%.0f%%), "
          "right %d/%d positive (%.0f%%), two-proportion z = %.2f"
          % ((tl > 0).sum(), len(tl), 100 * p1, (tr > 0).sum(), len(tr), 100 * p2,
             (p1 - p2) / se if se > 0 else 0.0))

    # per-donor azimuth listing
    print("")
    print("per-donor ECA azimuth (deg, superior-referenced):")
    for side in ("left", "right"):
        vals = sorted(azi[side])
        print("  %-6s %s" % (side, " ".join("%+.0f" % x for x in vals)))


if __name__ == "__main__":
    main()
