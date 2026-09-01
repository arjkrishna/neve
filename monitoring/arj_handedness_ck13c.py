# CHECK 13, round 3. Circular-safe chirality (sin of the bifurcation dihedral),
# plus an ANATOMICAL read of which way the ECA forks, with an LPS axis control
# taken from the host's own RCCA vs LCCA. ASCII only. Read-only on carotid_data/.
import json, os, glob, math
import numpy as np

ANAT = "carotid_data/anatomies"
HOST = "eve_bench/data/dualdevicenav/Centrelines_comb"
RCCA = "Centerline curve - RCCA.mrk.json"
RECA = "Centerline curve - RECA.mrk.json"
MIR = np.array([-1.0, 1.0, 1.0])
RNG = np.random.default_rng(0)


def read_curve(path):
    m = json.load(open(path, encoding="utf-8"))["markups"][0]
    return np.array([c["position"] for c in m["controlPoints"]], float), m["coordinateSystem"]


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def seg(p, a, b):
    s = arclen(p)
    return p[(s >= a) & (s <= b)]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def chord(p, a, b):
    q = seg(p, a, b)
    return unit(q[-1] - q[0]) if len(q) >= 2 else np.array([np.nan] * 3)


# ---------- CONTROL A: which way is +x in these files?
hp, cs = read_curve(os.path.join(HOST, RCCA))
hl, _ = read_curve(os.path.join(HOST, "Centerline curve - LCCA.mrk.json"))
print("CONTROL A  host frame, coordinateSystem=%s" % cs)
print("   host RCCA mean x %+8.2f   host LCCA mean x %+8.2f   -> LCCA is %s of RCCA in x"
      % (hp[:, 0].mean(), hl[:, 0].mean(), "POSITIVE" if hl[:, 0].mean() > hp[:, 0].mean() else "NEGATIVE"))
print("   host RCCA z rise ostium->tip %+8.2f mm (superior should be +z)" % (hp[-1, 2] - hp[0, 2]))
print("   => for a RIGHT carotid, MEDIAL (toward midline) = %s x"
      % ("+" if hl[:, 0].mean() > hp[:, 0].mean() else "-"))

rows = []
for dpath in sorted(glob.glob(os.path.join(ANAT, "*"))):
    name = os.path.basename(dpath)
    pv = json.load(open(os.path.join(dpath, "provenance.json"), encoding="utf-8"))
    p, _ = read_curve(os.path.join(dpath, "Centrelines_comb", RCCA))
    e, _ = read_curve(os.path.join(dpath, "Centrelines_comb", RECA))
    hc = pv["host_cut_mm"]
    bif = hc + pv["cca_mm"]
    lo_end = bif + pv["ica_mm"]
    lo, si = name.split("__")
    r = {"name": name, "lower": lo, "siphon": si,
         "lo_tag": "left" if lo.endswith("_left") else "right"}
    for L in (8.0, 12.0, 16.0, 20.0):
        tc = chord(p, max(bif - L, hc), bif)
        di = chord(p, bif, min(bif + L, lo_end))
        de = chord(e, 0.0, L)
        # sin of the signed dihedral of ECA about the CCA axis, measured from ICA.
        # Circular-safe (no wraparound), rotation invariant, negates under reflection.
        pi_ = unit(di - tc * float(np.dot(di, tc)))
        pe = unit(de - tc * float(np.dot(de, tc)))
        r["sin%g" % L] = float(np.dot(np.cross(pi_, pe), tc))
        r["cos%g" % L] = float(np.dot(pi_, pe))
        if L == 12.0:
            # anatomical decomposition of the ECA offset from the ICA, in the
            # plane perpendicular to the CCA axis
            off = pe - pi_ * float(np.dot(pe, pi_))
            r["off_x"] = float(off[0])      # LPS: +x = Left = MEDIAL for a right carotid
            r["off_y"] = float(off[1])      # LPS: +y = Posterior
            r["eca_med"] = float(off[0])
            r["eca_ant"] = float(-off[1])
            # mirror control
            tcm = chord(p * MIR, max(bif - L, hc), bif)
            dim = chord(p * MIR, bif, min(bif + L, lo_end))
            dem = chord(e * MIR, 0.0, L)
            pim = unit(dim - tcm * float(np.dot(dim, tcm)))
            pem = unit(dem - tcm * float(np.dot(dem, tcm)))
            r["sin12_mir"] = float(np.dot(np.cross(pim, pem), tcm))
    rows.append(r)

print("\nCONTROL B  mirrored composite must negate sin(dihedral)")
d = [abs(r["sin12"] + r["sin12_mir"]) for r in rows if np.isfinite(r["sin12"])]
print("   max |sin + sin_mirrored| = %.2e over %d anatomies" % (max(d), len(d)))


def per_donor(field):
    d = {}
    for r in rows:
        d.setdefault(r["lower"], []).append(r[field])
    med = {k: float(np.nanmedian(v)) for k, v in d.items()}
    spr = max(float(np.nanmax(v) - np.nanmin(v)) for v in d.values())
    return med, spr


def perm_p(a, b, n=200000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    sh = pool[np.argsort(RNG.random((n, len(pool))), axis=1)]
    return float((np.abs(sh[:, :na].mean(1) - sh[:, na:].mean(1)) >= obs - 1e-12).mean())


def fisher_2x2(a, b, c, d, n=200000):
    """two-sided permutation on the difference in positive-rate"""
    x = np.array([1] * a + [0] * b + [1] * c + [0] * d)
    na = a + b
    obs = abs(a / na - c / (c + d))
    sh = x[np.argsort(RNG.random((n, len(x))), axis=1)]
    st = np.abs(sh[:, :na].mean(1) - sh[:, na:].mean(1))
    return float((st >= obs - 1e-12).mean())


print("\n===== LOWER donors, per-donor median, grouped by side label =====")
for f in ("sin8", "sin12", "sin16", "sin20", "eca_med", "eca_ant"):
    med, spr = per_donor(f)
    g = {}
    for k, v in med.items():
        g.setdefault("left" if k.endswith("_left") else "right", []).append(v)
    a, b = np.array(g["left"]), np.array(g["right"])
    p = perm_p(a, b)
    pa, pb = int((a > 0).sum()), int((b > 0).sum())
    pf = fisher_2x2(pa, len(a) - pa, pb, len(b) - pb)
    print("  %-8s left  n=%2d mean %+7.3f med %+7.3f   right n=%2d mean %+7.3f med %+7.3f"
          % (f, len(a), a.mean(), np.median(a), len(b), b.mean(), np.median(b)))
    print("           mean diff %+7.3f  perm p=%.4f | if LEFT were mirrored, diff %+7.3f"
          % (a.mean() - b.mean(), p, (-a).mean() - b.mean()))
    print("           positive:  left %2d/%2d (%.0f%%)   right %2d/%2d (%.0f%%)  perm p=%.4f   "
          "within-donor spread max %.4f"
          % (pa, len(a), 100.0 * pa / len(a), pb, len(b), 100.0 * pb / len(b), pf, spr))

# paired same-patient
med, _ = per_donor("sin12")
pairs = [(k, med[k], med[k[:-4] + "right"]) for k in med
         if k.endswith("_left") and (k[:-4] + "right") in med]
x = np.array([q[1] for q in pairs]); y = np.array([q[2] for q in pairs])
print("\n  PAIRED same-patient lowers, sin12  n=%d" % len(pairs))
for k, u, v in pairs:
    print("    %-18s left %+7.3f   right %+7.3f   %s" % (k[:-5], u, v,
          "opposite" if u * v < 0 else "same"))
print("    opposite-sign %d/%d   mean|L-R| shipped %.3f   if L mirrored %.3f"
      % (int((x * y < 0).sum()), len(pairs), np.abs(x - y).mean(), np.abs(-x - y).mean()))

print("\n  per-donor sin12, sorted (left donors marked)")
for k, v in sorted(med.items(), key=lambda kv: kv[1]):
    print("    %-22s %-6s %+7.3f" % (k, "left" if k.endswith("_left") else "right", v))
