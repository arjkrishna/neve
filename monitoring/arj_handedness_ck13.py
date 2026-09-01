# CHECK 13 -- handedness of the two donor types in the 216 three-source set.
# Signed integrated torsion (rad): rotation-invariant, flips sign under reflection.
# ASCII only. Read-only on carotid_data/.
import json, os, glob, math
import numpy as np

ANAT = "carotid_data/anatomies"
RCCA = "Centerline curve - RCCA.mrk.json"
RECA = "Centerline curve - RECA.mrk.json"


def read_curve(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["markups"][0]
    return np.array([c["position"] for c in m["controlPoints"]], float)


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def seg(p, a, b):
    s = arclen(p)
    m = (s >= a) & (s <= b)
    return p[m]


def resample(p, step):
    s = arclen(p)
    if s[-1] < 3 * step:
        return None
    n = int(math.floor(s[-1] / step)) + 1
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def integ_torsion(p):
    """Sum of SIGNED rotation angles of the binormal about the tangent, radians.
    Under reflection b -> -Mb and the signed angle -> -angle, so this flips sign
    under reflection and is invariant under proper rotation."""
    d = np.diff(p, axis=0)
    t = unit(d)
    b = np.cross(t[:-1], t[1:])
    nb = np.linalg.norm(b, axis=1)
    ok = nb > 1e-9
    bu = b / np.maximum(nb[:, None], 1e-12)
    tot = 0.0
    for i in range(len(bu) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        ax = t[i + 1]
        c = float(np.dot(bu[i], bu[i + 1]))
        sn = float(np.dot(np.cross(bu[i], bu[i + 1]), ax))
        tot += math.atan2(sn, c)
    return tot


def chirality(p):
    """Sum of normalised scalar triple products of consecutive segments.
    No arccos/atan2 degeneracy; same invariance and antisymmetry."""
    d = np.diff(p, axis=0)
    u = unit(d)
    if len(u) < 3:
        return float("nan")
    return float(np.sum(np.einsum("ij,ij->i", np.cross(u[:-2], u[1:-1]), u[2:])))


def helix(c=0.35, turns=3.0, n=4000):
    th = np.linspace(0, 2 * math.pi * turns, n)
    return np.column_stack([np.cos(th), np.sin(th), c * th])


MIR = np.array([-1.0, 1.0, 1.0])

print("CONTROL 1  right-handed helix, analytic torsion > 0")
h = helix()
hm = h * MIR
for st in (0.05, 0.1):
    a = integ_torsion(resample(h, st))
    b = integ_torsion(resample(hm, st))
    print("   step %.2f  torsion %+8.3f   mirrored %+8.3f   chir %+8.3f / %+8.3f"
          % (st, a, b, chirality(resample(h, st)), chirality(resample(hm, st))))

rows = []
for dpath in sorted(glob.glob(os.path.join(ANAT, "*"))):
    name = os.path.basename(dpath)
    pv = json.load(open(os.path.join(dpath, "provenance.json"), encoding="utf-8"))
    p = read_curve(os.path.join(dpath, "Centrelines_comb", RCCA))
    e = read_curve(os.path.join(dpath, "Centrelines_comb", RECA))
    s = arclen(p)
    hc = pv["host_cut_mm"]
    bif = hc + pv["cca_mm"]
    lo_end = bif + pv["ica_mm"]
    lower_tag = "left" if name.split("__")[0].endswith("_left") else "right"
    sip_tag = "_L" if name.split("__")[1].endswith("_L") else "R"
    r = {"name": name, "lower": name.split("__")[0], "siphon": name.split("__")[1],
         "lower_tag": lower_tag, "sip_tag": sip_tag,
         "hc": hc, "bif": bif, "lo_end": lo_end, "total": s[-1]}
    for lbl, a, b in (("host", 0.0, hc), ("low", hc, lo_end), ("sip", lo_end, s[-1]),
                      ("lowM", hc + 5.0, lo_end - 5.0), ("sipM", lo_end + 5.0, s[-1] - 5.0)):
        q = seg(p, a, b)
        r[lbl + "_len"] = 0.0 if len(q) < 2 else float(arclen(q)[-1])
        for st in (2.0, 4.0, 8.0):
            rs = resample(q, st) if len(q) > 3 else None
            r["%s_t%g" % (lbl, st)] = float("nan") if rs is None else integ_torsion(rs)
            r["%s_c%g" % (lbl, st)] = float("nan") if rs is None else chirality(rs)
        rs = resample(q, 4.0) if len(q) > 3 else None
        r[lbl + "_mir4"] = float("nan") if rs is None else integ_torsion(rs * MIR)

    def dir_at(pp, a, b):
        q = seg(pp, a, b)
        if len(q) < 2:
            return np.array([np.nan] * 3)
        return unit(q[-1] - q[0])

    t_cca = dir_at(p, max(bif - 12.0, hc), bif)
    d_ica = dir_at(p, bif, min(bif + 12.0, lo_end))
    d_eca = dir_at(e, 0.0, 12.0)
    r["bifchi"] = float(np.dot(t_cca, np.cross(d_ica, d_eca)))
    r["bifang_ie"] = float(math.degrees(math.acos(np.clip(float(np.dot(d_ica, d_eca)), -1, 1))))
    rows.append(r)

print("\nn anatomies %d" % len(rows))


def stat(v):
    v = np.array([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return "n=0"
    return ("n=%3d mean %+7.2f  med %+7.2f  sd %6.2f  p10 %+7.2f p90 %+7.2f  pos %d/%d"
            % (len(v), v.mean(), np.median(v), v.std(ddof=1) if len(v) > 1 else 0.0,
               np.percentile(v, 10), np.percentile(v, 90), int((v > 0).sum()), len(v)))


def group(rr, key, field):
    out = {}
    for r in rr:
        out.setdefault(r[key], []).append(r[field])
    return out


print("\nCONTROL 2  mirror of each window must return the exact negative")
for lbl in ("host", "low", "sip"):
    d = [abs(r[lbl + "_t4"] + r[lbl + "_mir4"]) for r in rows if np.isfinite(r[lbl + "_t4"])]
    print("   %-5s max |t + t_mirrored| = %.3e over %d windows" % (lbl, max(d), len(d)))

print("\nCONTROL 3  HOST window is shipped geometry common to all 216 -> groups must not differ")
for key in ("lower_tag", "sip_tag"):
    g = group(rows, key, "host_t4")
    for k in sorted(g):
        print("   host_t4 by %-9s %-6s %s" % (key, k, stat(g[k])))

print("\nsegment lengths (mm)")
for lbl in ("host", "low", "sip"):
    v = np.array([r[lbl + "_len"] for r in rows])
    print("   %-5s min %6.1f med %6.1f max %6.1f" % (lbl, v.min(), np.median(v), v.max()))

print("\n=== LOWER SEGMENT, integrated torsion, grouped by donor side label ===")
for st in (2.0, 4.0, 8.0):
    print("  node step %.0f mm" % st)
    for lbl in ("low", "lowM"):
        g = group(rows, "lower_tag", "%s_t%g" % (lbl, st))
        for k in sorted(g):
            print("    %-5s %-6s %s" % (lbl, k, stat(g[k])))

print("\n=== SIPHON SEGMENT, integrated torsion, grouped by _L / no-suffix ===")
for st in (2.0, 4.0, 8.0):
    print("  node step %.0f mm" % st)
    for lbl in ("sip", "sipM"):
        g = group(rows, "sip_tag", "%s_t%g" % (lbl, st))
        for k in sorted(g):
            print("    %-5s %-6s %s" % (lbl, k, stat(g[k])))

print("\n=== chirality sum (degeneracy-free), step 4 mm ===")
for key, lbl in (("lower_tag", "low"), ("sip_tag", "sip")):
    g = group(rows, key, "%s_c4" % lbl)
    for k in sorted(g):
        print("   %-4s %-6s %s" % (lbl, k, stat(g[k])))

print("\n=== bifurcation chirality  t_CCA . (d_ICA x d_ECA)  by lower side ===")
g = group(rows, "lower_tag", "bifchi")
for k in sorted(g):
    print("   %-6s %s" % (k, stat(g[k])))
g = group(rows, "lower_tag", "bifang_ie")
for k in sorted(g):
    print("   ICA-ECA angle deg %-6s %s" % (k, stat(g[k])))


def donor_table(rr, key, field):
    d = {}
    for r in rr:
        d.setdefault(r[key], []).append(r[field])
    return {k: float(np.nanmedian(v)) for k, v in d.items()}, d


print("\n=== PER-DONOR (deduplicated; each donor reused across many composites) ===")
specs = (("lower", "low", lambda n: "left" if n.endswith("_left") else "right"),
         ("siphon", "sip", lambda n: "_L" if n.endswith("_L") else "R"))
for key, lbl, tagf in specs:
    med, raw = donor_table(rows, key, "%s_t4" % lbl)
    spread = max(np.nanmax(v) - np.nanmin(v) for v in raw.values())
    print("  %s donors: %d, max within-donor spread of %s_t4 across its composites %.4f"
          % (key, len(med), lbl, spread))
    byg = {}
    for k, v in med.items():
        byg.setdefault(tagf(k), []).append(v)
    for k in sorted(byg):
        print("    %-6s %s" % (k, stat(byg[k])))
    pairs = []
    for k in med:
        if key == "lower":
            if k.endswith("_left"):
                o = k[:-4] + "right"
                if o in med:
                    pairs.append((k, med[k], med[o]))
        else:
            if k.endswith("_L"):
                o = k[:-2]
                if o in med:
                    pairs.append((k, med[k], med[o]))
    if pairs:
        a = np.array([q[1] for q in pairs])
        b = np.array([q[2] for q in pairs])
        print("    paired same-patient n=%d : mean(as-shipped L) %+.2f  mean(R) %+.2f"
              % (len(pairs), a.mean(), b.mean()))
        print("       mean|diff| as shipped %.2f   if L sign-flipped %.2f   opposite-sign pairs %d/%d"
              % (np.abs(a - b).mean(), np.abs(-a - b).mean(), int((a * b < 0).sum()), len(pairs)))
    else:
        print("    no same-patient pairs found for %s" % key)

print("\nOUTLIERS, low_t4 per-donor median, 6 each way")
med, _ = donor_table(rows, "lower", "low_t4")
o = sorted(med.items(), key=lambda kv: kv[1])
for k, v in o[:6] + o[-6:]:
    print("   %-22s %+8.2f" % (k, v))
print("OUTLIERS, sip_t4 per-donor median, 6 each way")
med, _ = donor_table(rows, "siphon", "sip_t4")
o = sorted(med.items(), key=lambda kv: kv[1])
for k, v in o[:6] + o[-6:]:
    print("   %-22s %+8.2f" % (k, v))
