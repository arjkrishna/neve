# CHECK 13, round 4. Circular mean of the bifurcation dihedral per group, and a
# paired sign/permutation test for the SIPHON mirror. ASCII only. Read-only.
import json, os, glob, math
import numpy as np

ANAT = "carotid_data/anatomies"
RCCA = "Centerline curve - RCCA.mrk.json"
RECA = "Centerline curve - RECA.mrk.json"
RNG = np.random.default_rng(1)


def read_curve(p):
    m = json.load(open(p, encoding="utf-8"))["markups"][0]
    return np.array([c["position"] for c in m["controlPoints"]], float)


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def seg(p, a, b):
    s = arclen(p)
    return p[(s >= a) & (s <= b)]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def chord(p, a, b):
    q = seg(p, a, b)
    return unit(q[-1] - q[0])


def resample(p, step):
    s = arclen(p)
    if len(p) < 4 or s[-1] < 3 * step:
        return None
    t = np.linspace(0.0, s[-1], int(math.floor(s[-1] / step)) + 1)
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


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


rows = []
for dp in sorted(glob.glob(os.path.join(ANAT, "*"))):
    n = os.path.basename(dp)
    pv = json.load(open(os.path.join(dp, "provenance.json"), encoding="utf-8"))
    p = read_curve(os.path.join(dp, "Centrelines_comb", RCCA))
    e = read_curve(os.path.join(dp, "Centrelines_comb", RECA))
    hc = pv["host_cut_mm"]; bif = hc + pv["cca_mm"]; lo = bif + pv["ica_mm"]
    tot = arclen(p)[-1]
    L = 16.0
    tc = chord(p, max(bif - L, hc), bif)
    di = chord(p, bif, min(bif + L, lo))
    de = chord(e, 0.0, L)
    pi_ = unit(di - tc * float(np.dot(di, tc)))
    pe = unit(de - tc * float(np.dot(de, tc)))
    dih = math.degrees(math.atan2(float(np.dot(np.cross(pi_, pe), tc)), float(np.dot(pi_, pe))))
    q = resample(seg(p, lo, tot), 4.0)
    rows.append({"lower": n.split("__")[0], "siphon": n.split("__")[1], "dih": dih,
                 "sip_t": float("nan") if q is None else integ_torsion(q)})

print("n composites %d" % len(rows))
nl = sum(1 for r in rows if r["lower"].endswith("_left"))
ns = sum(1 for r in rows if r["siphon"].endswith("_L"))
print("composites carrying a _left LOWER  : %d / %d" % (nl, len(rows)))
print("composites carrying an _L SIPHON   : %d / %d" % (ns, len(rows)))


def donor_med(key, field, ang=False):
    d = {}
    for r in rows:
        d.setdefault(r[key], []).append(r[field])
    if ang:
        return {k: math.degrees(math.atan2(np.mean(np.sin(np.radians(v))),
                                           np.mean(np.cos(np.radians(v))))) for k, v in d.items()}
    return {k: float(np.nanmedian(v)) for k, v in d.items()}


def circ_mean_R(deg):
    a = np.radians(np.asarray(deg, float))
    C, S = np.cos(a).mean(), np.sin(a).mean()
    return math.degrees(math.atan2(S, C)), math.hypot(C, S)


print("\n=== LOWER: circular mean of the ICA->ECA dihedral about the CCA axis (16 mm chords) ===")
med = donor_med("lower", "dih", ang=True)
g = {}
for k, v in med.items():
    g.setdefault("left" if k.endswith("_left") else "right", []).append(v)
for k in sorted(g):
    m, R = circ_mean_R(g[k])
    print("   %-6s n=%2d  circular mean %+8.1f deg   resultant R %.3f" % (k, len(g[k]), m, R))
m1, _ = circ_mean_R(g["left"]); m2, _ = circ_mean_R(g["right"])
d = (m1 - m2 + 180) % 360 - 180
dm = (-m1 - m2 + 180) % 360 - 180
print("   separation left vs right %+8.1f deg   |  if LEFT mirrored %+8.1f deg" % (d, dm))

print("\n=== SIPHON: integrated torsion, paired same-patient (mirror was applied to _L) ===")
med = donor_med("siphon", "sip_t")
pairs = [(k[:-2], med[k], med[k[:-2]]) for k in med if k.endswith("_L") and k[:-2] in med]
x = np.array([q[1] for q in pairs]); y = np.array([q[2] for q in pairs])
sh = np.abs(x - y); fl = np.abs(-x - y)
print("   pairs n=%d   mean|shipped_L - R| %.2f   mean|flipped_L - R| %.2f" % (len(pairs), sh.mean(), fl.mean()))
print("   pairs where shipped is closer than flipped: %d/%d" % (int((sh < fl).sum()), len(pairs)))
# permutation: randomly flip the sign of each L and see how often the shipped
# orientation beats the alternative by this much
obs = fl.mean() - sh.mean()
cnt = 0
for _ in range(20000):
    s = RNG.choice([-1.0, 1.0], size=len(x))
    a = np.abs(s * x - y); b = np.abs(-s * x - y)
    if b.mean() - a.mean() >= obs:
        cnt += 1
print("   sign-flip permutation p (shipped orientation is the better one) = %.4f" % (cnt / 20000.0))
print("   group means: shipped _L %+6.2f   R %+6.2f   flipped _L would be %+6.2f"
      % (x.mean(), y.mean(), -x.mean()))
print("\n   per-pair siphon torsion (rad): patient, _L as shipped, R")
for nm, a, b in sorted(pairs):
    print("     %-16s %+7.2f  %+7.2f   %s" % (nm, a, b, "opposite" if a * b < 0 else "same"))
