#!/usr/bin/env python3
"""Independent chirality audit of the 216 three-source carotid composites.

Instruments:
  A) integrated torsion (binormal-rotation estimator) over the lower-donor
     band [seam1, 130 mm] of the RCCA route
  B) same estimator on the siphon band (>130 mm) -- KNOWN-MIRROR CONTROL
  C) frame-free triple product det[u_cca, u_ica, u_eca] at the fork
  D) world-frame medial component of the ECA departure (LPS +x = patient left)
"""
import glob
import json
import os

import numpy as np

ROOT = "D:/Arjun/workspace/neve/carotid_data/anatomies"
HOST = "D:/Arjun/workspace/neve/eve_bench/data/dualdevicenav/Centrelines_comb"
SEAM2 = 130.0


def read_pos(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["markups"][0]
    return (np.array([cp["position"] for cp in m["controlPoints"]], float),
            m.get("coordinateSystem"))


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_uniform(p, lo, hi, step):
    s = arclen(p)
    n = max(int(round((hi - lo) / step)) + 1, 4)
    t = np.linspace(lo, hi, n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def integrated_torsion(q):
    """Sum of signed rotation of the binormal about the tangent, in radians.

    Rotation invariant; sign-flips under reflection.
    """
    e = np.diff(q, axis=0)
    b, ok = [], []
    for i in range(len(e) - 1):
        c = np.cross(e[i], e[i + 1])
        nc = np.linalg.norm(c)
        if nc < 1e-9:
            b.append(np.zeros(3))
            ok.append(False)
        else:
            b.append(c / nc)
            ok.append(True)
    b = np.array(b)
    ok = np.array(ok)
    last = None
    for i in range(len(b)):
        if ok[i]:
            last = b[i]
        elif last is not None:
            b[i] = last
            ok[i] = True
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return 0.0, 0
    tot = 0.0
    for a, c in zip(idx[:-1], idx[1:]):
        b0, b1 = b[a], b[c]
        t = unit(e[c])
        cr = np.cross(b0, b1)
        ang = np.arctan2(np.linalg.norm(cr), float(np.clip(b0 @ b1, -1, 1)))
        s = float(cr @ t)
        tot += ang * (1.0 if s >= 0 else -1.0)
    return float(tot), len(idx)


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return dict(n=0)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                pos=int((v > 0).sum()), neg=int((v < 0).sum()),
                lo=float(v.min()), hi=float(v.max()))


def line(tag, d):
    if not d["n"]:
        print("  %-22s n=0" % tag)
        return
    print("  %-22s n=%3d  mean %+9.3f  median %+9.3f  sd %8.3f  pos %3d/%3d"
          "  range [%+9.2f, %+9.2f]"
          % (tag, d["n"], d["mean"], d["median"], d["sd"], d["pos"], d["n"],
             d["lo"], d["hi"]))


def main():
    hp, hcs = read_pos(os.path.join(HOST, "Centerline curve - RCCA.mrk.json"))
    hs = arclen(hp)
    print("host RCCA: %d pts, %.1f mm, coordinateSystem=%s" % (len(hp), hs[-1], hcs))

    tt = np.linspace(0, 6 * np.pi, 400)
    helix = np.stack([np.cos(tt) * 10, np.sin(tt) * 10, tt * 3], 1)
    a, _ = integrated_torsion(helix)
    bmir, _ = integrated_torsion(helix * np.array([-1.0, 1.0, 1.0]))
    th = 0.7
    R1 = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.]])
    R2 = np.array([[1, 0, 0], [0, np.cos(.4), -np.sin(.4)], [0, np.sin(.4), np.cos(.4)]])
    R = R1 @ R2
    c, _ = integrated_torsion(helix @ R.T)
    print("SELF-TEST helix: tau=%+.4f  reflected=%+.4f  rotated=%+.4f "
          "(want reflected=-tau, rotated=tau)" % (a, bmir, c))

    dirs = sorted(glob.glob(os.path.join(ROOT, "*")))
    rows = []
    csys = set()
    for d in dirs:
        name = os.path.basename(d)
        pv = json.load(open(os.path.join(d, "provenance.json"), encoding="utf-8"))
        rp, cs1 = read_pos(os.path.join(d, "Centrelines_comb",
                                        "Centerline curve - RCCA.mrk.json"))
        ep, cs2 = read_pos(os.path.join(d, "Centrelines_comb",
                                        "Centerline curve - RECA.mrk.json"))
        csys |= {cs1, cs2}
        rs = arclen(rp)
        m = min(len(rp), len(hp))
        dd = np.linalg.norm(rp[:m] - hp[:m], axis=1)
        j = int(np.argmax(dd > 0.05)) if bool((dd > 0.05).any()) else m - 1
        seam1_meas = float(rs[j])
        rows.append(dict(
            name=name, lower=pv["lower"], siphon=pv["siphon"],
            side="left" if "_left" in pv["lower"] else "right",
            sip_L=pv["siphon"].endswith("_L"),
            host_cut=pv["host_cut_mm"], seam1_meas=seam1_meas,
            cca=pv["cca_mm"], ica=pv["ica_mm"], total=float(rs[-1]),
            rp=rp, ep=ep, rs=rs))

    print("coordinateSystem values seen: %s" % sorted(str(x) for x in csys))
    print("n anatomies: %d" % len(rows))

    hc = np.array([r["host_cut"] for r in rows])
    sm = np.array([r["seam1_meas"] for r in rows])
    print("")
    print("SEAM 1: provenance host_cut_mm  %.1f-%.1f mm (median %.1f)"
          % (hc.min(), hc.max(), np.median(hc)))
    print("        measured departure      %.1f-%.1f mm (median %.1f)"
          % (sm.min(), sm.max(), np.median(sm)))
    print("        |measured - provenance| max %.3f mm, median %.3f mm"
          % (np.abs(sm - hc).max(), np.median(np.abs(sm - hc))))
    band = SEAM2 - hc
    print("BAND (lower donor) length: %.1f-%.1f mm, median %.1f, n<40mm=%d n<60mm=%d"
          % (band.min(), band.max(), np.median(band),
             int((band < 40).sum()), int((band < 60).sum())))
    seg = np.array([r["cca"] + r["ica"] for r in rows])
    print("        cross-check cca+ica vs 130-host_cut: max err %.4f mm"
          % np.abs(seg - band).max())
    tl = np.array([r["total"] for r in rows])
    print("SIPHON band (>130 mm) length: %.1f-%.1f mm, median %.1f"
          % ((tl - 130).min(), (tl - 130).max(), np.median(tl - 130)))

    for step in (2.0, 4.0, 6.0, 8.0):
        print("")
        print("==================== resample step %.1f mm ====================" % step)
        for label, key in (("LOWER band [seam1,130]", "low"),
                           ("SIPHON band [130,end]  (KNOWN MIRROR CONTROL)", "sip"),
                           ("ICA sub-band [fork,130]", "ica")):
            L, Rr = [], []
            for r in rows:
                if key == "low":
                    lo, hi = r["host_cut"], SEAM2
                    grp = r["side"] == "left"
                elif key == "sip":
                    lo, hi = SEAM2, r["total"]
                    grp = r["sip_L"]
                else:
                    lo, hi = r["host_cut"] + r["cca"], SEAM2
                    grp = r["side"] == "left"
                if hi - lo < 4 * step:
                    continue
                q = resample_uniform(r["rp"], lo, hi, step)
                v, _ = integrated_torsion(q)
                (L if grp else Rr).append(v)
            gl = "_L / left group" if key == "sip" else "_left group"
            gr = "plain / right group" if key == "sip" else "_right group"
            dl, dr = stats(L), stats(Rr)
            print("")
            print("%s" % label)
            line(gl, dl)
            line(gr, dr)
            if dl["n"] and dr["n"]:
                print("    mean diff as-is        %+9.3f rad" % (dl["mean"] - dr["mean"]))
                print("    mean diff LEFT flipped %+9.3f rad" % (-dl["mean"] - dr["mean"]))
                print("    median diff as-is      %+9.3f | flipped %+9.3f"
                      % (dl["median"] - dr["median"], -dl["median"] - dr["median"]))

    print("")
    print("==================== FORK CHIRALITY (no Frenet) ====================")
    trip = {"left": [], "right": []}
    med = {"left": [], "right": []}
    azi = {"left": [], "right": []}
    forkerr = []
    for r in rows:
        rp, ep, rs = r["rp"], r["ep"], r["rs"]
        es = arclen(ep)
        k = int(np.argmin(np.linalg.norm(rp - ep[0], axis=1)))
        forkerr.append((float(rs[k]) - (r["host_cut"] + r["cca"]),
                        float(np.linalg.norm(rp[k] - ep[0]))))
        s_f = float(rs[k])
        d = 10.0
        if s_f - d < r["host_cut"] or s_f + d > SEAM2 or es[-1] < d:
            continue
        pm = np.stack([np.interp([s_f - d, s_f, s_f + d], rs, rp[:, c]) for c in range(3)], 1)
        u_cca = unit(pm[1] - pm[0])
        u_ica = unit(pm[2] - pm[1])
        pe = np.array([np.interp(d, es, ep[:, c]) for c in range(3)])
        u_eca = unit(pe - ep[0])
        t = float(np.dot(np.cross(u_cca, u_ica), u_eca))
        v = unit(u_eca - u_ica * float(u_eca @ u_ica))
        trip[r["side"]].append(t)
        med[r["side"]].append(float(v[0]))
        ref = unit(np.array([1.0, 0, 0]) - u_ica * float(u_ica[0]))
        azi[r["side"]].append(
            float(np.degrees(np.arctan2(float(v @ np.cross(u_ica, ref)), float(v @ ref)))))
    fe = np.array(forkerr)
    print("fork located: |s_fork-(host_cut+cca)| max %.2f mm; route-ECA origin gap max %.3f mm"
          % (np.abs(fe[:, 0]).max(), fe[:, 1].max()))
    for nm, dct in (("triple product det[u_cca,u_ica,u_eca]", trip),
                    ("ECA perp unit vector, LPS +x (medial for a right carotid)", med),
                    ("ECA azimuth about route tangent (deg)", azi)):
        print("")
        print("%s" % nm)
        dl, dr = stats(dct["left"]), stats(dct["right"])
        line("_left donors", dl)
        line("_right donors", dr)
        if dl["n"] and dr["n"]:
            print("    mean diff as-is        %+9.3f" % (dl["mean"] - dr["mean"]))
            print("    mean diff LEFT flipped %+9.3f" % (-dl["mean"] - dr["mean"]))

    print("")
    print("==================== counts ====================")
    nl = sum(1 for r in rows if r["side"] == "left")
    print("composites: lower _left %d, lower _right %d" % (nl, len(rows) - nl))
    ns = sum(1 for r in rows if r["sip_L"])
    print("composites: siphon _L %d, siphon plain %d" % (ns, len(rows) - ns))
    ld = sorted({r["lower"] for r in rows})
    print("distinct lower donors: %d (_left %d, _right %d)"
          % (len(ld), sum("_left" in x for x in ld), sum("_right" in x for x in ld)))
    sd = sorted({r["siphon"] for r in rows})
    print("distinct siphons: %d (_L %d, plain %d)"
          % (len(sd), sum(x.endswith("_L") for x in sd),
             sum(not x.endswith("_L") for x in sd)))


if __name__ == "__main__":
    main()
