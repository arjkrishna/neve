# CHECK 13, round 2. Per-DONOR handedness with a normalised signed dihedral at the
# carotid bifurcation, CCA/ICA windows split, and permutation tests.
# ASCII only. Read-only on carotid_data/.
import json, os, glob, math
import numpy as np

ANAT = "carotid_data/anatomies"
RCCA = "Centerline curve - RCCA.mrk.json"
RECA = "Centerline curve - RECA.mrk.json"
MIR = np.array([-1.0, 1.0, 1.0])
RNG = np.random.default_rng(0)


def read_curve(path):
    m = json.load(open(path, encoding="utf-8"))["markups"][0]
    return np.array([c["position"] for c in m["controlPoints"]], float)


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def seg(p, a, b):
    s = arclen(p)
    return p[(s >= a) & (s <= b)]


def resample(p, step):
    s = arclen(p)
    if len(p) < 4 or s[-1] < 3 * step:
        return None
    t = np.linspace(0.0, s[-1], int(math.floor(s[-1] / step)) + 1)
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def integ_torsion(p):
    d = np.diff(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    b = np.cross(t[:-1], t[1:])
    nb = np.linalg.norm(b, axis=1)
    ok = nb > 1e-9
    bu = b / np.maximum(nb[:, None], 1e-12)
    tot = 0.0
    for i in range(len(bu) - 1):
        if ok[i] and ok[i + 1]:
            tot += math.atan2(float(np.dot(np.cross(bu[i], bu[i + 1]), t[i + 1])),
                              float(np.dot(bu[i], bu[i + 1])))
    return tot


def dihedral(t_axis, a, b):
    """Signed angle (deg) from a to b about t_axis, both projected perpendicular
    to it. Rotation invariant; flips sign under reflection."""
    t = unit(t_axis)
    pa = a - t * float(np.dot(a, t))
    pb = b - t * float(np.dot(b, t))
    na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    pa, pb = pa / na, pb / nb
    return math.degrees(math.atan2(float(np.dot(np.cross(pa, pb), t)),
                                   float(np.dot(pa, pb))))


rows = []
for dpath in sorted(glob.glob(os.path.join(ANAT, "*"))):
    name = os.path.basename(dpath)
    pv = json.load(open(os.path.join(dpath, "provenance.json"), encoding="utf-8"))
    p = read_curve(os.path.join(dpath, "Centrelines_comb", RCCA))
    e = read_curve(os.path.join(dpath, "Centrelines_comb", RECA))
    hc = pv["host_cut_mm"]
    bif = hc + pv["cca_mm"]
    lo_end = bif + pv["ica_mm"]
    tot = arclen(p)[-1]
    lo = name.split("__")[0]
    si = name.split("__")[1]
    r = {"name": name, "lower": lo, "siphon": si,
         "lo_tag": "left" if lo.endswith("_left") else "right",
         "si_tag": "_L" if si.endswith("_L") else "R"}
    for lbl, a, b in (("cca", hc, bif), ("ica", bif, lo_end),
                      ("low", hc, lo_end), ("sip", lo_end, tot)):
        q = resample(seg(p, a, b), 4.0)
        r[lbl + "_t"] = float("nan") if q is None else integ_torsion(q)
    # bifurcation frame, several span choices to show it is not a span artefact
    for L in (8.0, 12.0, 20.0):
        tc = unit(seg(p, max(bif - L, hc), bif)[-1] - seg(p, max(bif - L, hc), bif)[0])
        di = unit(seg(p, bif, min(bif + L, lo_end))[-1] - seg(p, bif, min(bif + L, lo_end))[0])
        de = unit(seg(e, 0.0, L)[-1] - seg(e, 0.0, L)[0])
        r["chi%g" % L] = float(np.dot(tc, np.cross(di, de)))
        r["dih%g" % L] = dihedral(tc, di, de)
    # CONTROL: mirror the whole composite, the dihedral must return the negative
    pm, em = p * MIR, e * MIR
    L = 12.0
    tc = unit(seg(pm, max(bif - L, hc), bif)[-1] - seg(pm, max(bif - L, hc), bif)[0])
    di = unit(seg(pm, bif, min(bif + L, lo_end))[-1] - seg(pm, bif, min(bif + L, lo_end))[0])
    de = unit(seg(em, 0.0, L)[-1] - seg(em, 0.0, L)[0])
    r["dih12_mir"] = dihedral(tc, di, de)
    rows.append(r)

print("CONTROL  mirrored composite must negate the bifurcation dihedral")
d = [abs(r["dih12"] + r["dih12_mir"]) for r in rows if np.isfinite(r["dih12"])]
print("   max |dih + dih_mirrored| = %.2e over %d anatomies" % (max(d), len(d)))


def per_donor(key, field):
    d = {}
    for r in rows:
        d.setdefault(r[key], []).append(r[field])
    med = {k: float(np.nanmedian(v)) for k, v in d.items()}
    spr = max(float(np.nanmax(v) - np.nanmin(v)) for v in d.values())
    return med, spr


def tag_of(key, n):
    if key == "lower":
        return "left" if n.endswith("_left") else "right"
    return "_L" if n.endswith("_L") else "R"


def perm_p(a, b, n=200000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    idx = np.argsort(RNG.random((n, len(pool))), axis=1)
    sh = pool[idx]
    stat = np.abs(sh[:, :na].mean(1) - sh[:, na:].mean(1))
    return float((stat >= obs - 1e-12).mean()), obs


def report(key, fields):
    print("\n===== %s donors =====" % key.upper())
    for f in fields:
        med, spr = per_donor(key, f)
        g = {}
        for k, v in med.items():
            g.setdefault(tag_of(key, k), []).append(v)
        ks = sorted(g)
        a, b = np.array(g[ks[0]]), np.array(g[ks[1]])
        p, obs = perm_p(a, b)
        # is the flipped group closer to the recipient group than the shipped one?
        ref = b if key == "lower" else b        # right / R is the recipient side
        oth = a
        print("  %-7s  %-6s n=%2d mean %+7.2f med %+7.2f  |  %-6s n=%2d mean %+7.2f med %+7.2f"
              % (f, ks[0], len(a), a.mean(), np.median(a), ks[1], len(b), b.mean(), np.median(b)))
        print("            group mean diff %+7.2f (perm p=%.4f)   if that group were mirrored: %+7.2f"
              % (a.mean() - b.mean(), p, (-a).mean() - b.mean()))
        print("            within-donor spread across its composites, max %.3f" % spr)
        print("            sign split  %-6s %d/%d positive    %-6s %d/%d positive"
              % (ks[0], int((a > 0).sum()), len(a), ks[1], int((b > 0).sum()), len(b)))
        # paired same patient
        pairs = []
        for k in med:
            if key == "lower" and k.endswith("_left"):
                o = k[:-4] + "right"
                if o in med:
                    pairs.append((k, med[k], med[o]))
            if key == "siphon" and k.endswith("_L"):
                o = k[:-2]
                if o in med:
                    pairs.append((k, med[k], med[o]))
        if pairs:
            x = np.array([q[1] for q in pairs]); y = np.array([q[2] for q in pairs])
            print("            paired n=%2d  mean|L-R| shipped %6.2f   if L flipped %6.2f   "
                  "opposite-sign %d/%d" % (len(pairs), np.abs(x - y).mean(),
                                           np.abs(-x - y).mean(),
                                           int((x * y < 0).sum()), len(pairs)))


report("lower", ["dih12", "dih8", "dih20", "chi12", "ica_t", "cca_t", "low_t"])
report("siphon", ["sip_t"])

print("\nPER-DONOR bifurcation dihedral (deg), lower donors, sorted")
med, _ = per_donor("lower", "dih12")
for k, v in sorted(med.items(), key=lambda kv: kv[1]):
    print("   %-22s %-6s %+8.1f" % (k, tag_of("lower", k), v))
