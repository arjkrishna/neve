"""CHECK 8 -- INTER-VESSEL FUSION, all 216 carotid anatomies.

Metric: centre distance minus both DECLARED radii, over every branch pair,
centerlines densified to <=0.25 mm.  Topologically adjacent branches share a
junction lumen by construction, so points within T mm ALONG-BRANCH of any
junction contact are masked out; the un-masked minimum is the fusion test.
Also reports the RCCA/RECA bifurcation divergence profile explicitly.
"""
import sys, os, json, hashlib
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from collections import Counter
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
STEP = 0.25
TOUCH = 1.0            # centre distance below which two branches are in contact -> junction
TS = (5.0, 8.0, 12.0)  # along-branch junction exclusion radii tested


def densify(C, R, step=STEP):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(np.ceil(S[-1] / step)) + 1, len(C))
    Sn = np.linspace(0.0, S[-1], n)
    Cn = np.stack([np.interp(Sn, S, C[:, k]) for k in range(3)], axis=1)
    Rn = np.interp(Sn, S, R)
    return Cn, Rn, Sn


def short(n):
    return str(n).replace("Centerline curve ", "").replace(".mrk", "")


def pair_analysis(A, ra, sa, B, rb, sb, chunk=400):
    nA, nB = len(A), len(B)
    dminA = np.full(nA, np.inf)
    dminB = np.full(nB, np.inf)
    for i0 in range(0, nA, chunk):
        blk = A[i0:i0 + chunk]
        D = np.sqrt(((blk[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        dminA[i0:i0 + chunk] = D.min(axis=1)
        np.minimum(dminB, D.min(axis=0), out=dminB)
    contA = sa[dminA < TOUCH]
    contB = sb[dminB < TOUCH]
    out = {}
    for T in TS:
        if contA.size:
            mA = np.min(np.abs(sa[:, None] - contA[None, :]), axis=1) >= T
        else:
            mA = np.ones(nA, bool)
        if contB.size:
            mB = np.min(np.abs(sb[:, None] - contB[None, :]), axis=1) >= T
        else:
            mB = np.ones(nB, bool)
        if not mA.any() or not mB.any():
            out[T] = (float("nan"), float("nan"), float("nan"))
            continue
        Am, ram, sam = A[mA], ra[mA], sa[mA]
        Bm, rbm, sbm = B[mB], rb[mB], sb[mB]
        best = (np.inf, -1, -1)
        for i0 in range(0, len(Am), chunk):
            blk = Am[i0:i0 + chunk]
            rblk = ram[i0:i0 + chunk]
            G = (np.sqrt(((blk[:, None, :] - Bm[None, :, :]) ** 2).sum(-1))
                 - rblk[:, None] - rbm[None, :])
            i, j = np.unravel_index(np.argmin(G), G.shape)
            if G[i, j] < best[0]:
                best = (float(G[i, j]), i0 + i, j)
        out[T] = (best[0], float(sam[best[1]]), float(sbm[best[2]]))
    return out


names = sorted(os.listdir(ROOT))
names = [n for n in names if os.path.isdir(os.path.join(ROOT, n, "Centrelines_comb"))]
print("anatomies:", len(names))

hashes = {}
for nm in names:
    brs = load_branches(os.path.join(ROOT, nm, "Centrelines_comb"))
    for b in brs:
        h = hashlib.md5(np.asarray(b.coordinates, np.float32).tobytes()
                        + np.asarray(b.radii, np.float32).tobytes()).hexdigest()
        hashes.setdefault(short(b.name), set()).add(h)
print("distinct geometries per branch across the 216:")
print("   " + "  ".join("%s=%d" % (k, len(v))
                        for k, v in sorted(hashes.items(), key=lambda x: -len(x[1]))))
sys.stdout.flush()

rows = []
host_cache = {}
VARYING = set(["- RCCA", "- RECA", "- RVA"])
for ai, nm in enumerate(names):
    brs = load_branches(os.path.join(ROOT, nm, "Centrelines_comb"))
    dens = {}
    for b in brs:
        C = np.asarray(b.coordinates, np.float64)
        R = np.asarray(b.radii, np.float64)
        dens[short(b.name)] = densify(C, R)
    keys = list(dens.keys())
    res = {}
    if not host_cache:
        for ii, a in enumerate(keys):
            for b in keys[ii + 1:]:
                if a in VARYING or b in VARYING:
                    continue
                A, ra, sa = dens[a]
                B, rb, sb = dens[b]
                host_cache[(a, b)] = pair_analysis(A, ra, sa, B, rb, sb)
        print("  host-host pairs cached:", len(host_cache))
        sys.stdout.flush()
    res.update(host_cache)
    for ii, a in enumerate(keys):
        for b in keys[ii + 1:]:
            if not (a in VARYING or b in VARYING):
                continue
            A, ra, sa = dens[a]
            B, rb, sb = dens[b]
            res[(a, b)] = pair_analysis(A, ra, sa, B, rb, sb)

    C, rC, sC = dens["- RCCA"]
    E, rE, sE = dens["- RECA"]
    gv = np.zeros(len(E))
    for j0 in range(0, len(E), 400):
        blk = E[j0:j0 + 400]
        M = np.sqrt(((blk[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        G = M - rE[j0:j0 + 400][:, None] - rC[None, :]
        gv[j0:j0 + 400] = G.min(axis=1)
    k0 = int(np.argmin(np.linalg.norm(C - E[0], axis=1)))
    s_fork = float(sC[k0])
    d_fork = float(np.linalg.norm(C[k0] - E[0]))
    pos = np.where(gv >= 0.0)[0]
    d_sep = float(sE[pos[0]]) if pos.size else float("nan")
    if pos.size:
        tail = gv[pos[0]:]
        reapp = float(tail.min())
        reapp_s = float(sE[pos[0] + int(np.argmin(tail))])
    else:
        reapp, reapp_s = float("nan"), float("nan")

    r = {"name": nm, "s_fork": s_fork, "d_fork": d_fork, "L_eca": float(sE[-1]),
         "d_sep": d_sep, "reapp": reapp, "reapp_s": reapp_s, "L_rcca": float(sC[-1])}
    for T in TS:
        g, sa_, sb_ = res[("- RCCA", "- RECA")][T]
        r["g%g" % T] = g
        r["sR%g" % T] = sa_
        r["sE%g" % T] = sb_
    worst = []
    for (a, b), o in res.items():
        g, s1, s2 = o[8.0]
        if g == g:
            worst.append((g, a, b, s1, s2))
    worst.sort(key=lambda w: w[0])
    r["gmin_tree"] = worst[0][0]
    r["gmin_pair"] = "%s|%s" % (worst[0][1], worst[0][2])
    r["gmin_s"] = (round(worst[0][3], 2), round(worst[0][4], 2))
    r["n_neg8"] = sum(1 for w in worst if w[0] < 0)
    r["neg_pairs"] = [(round(w[0], 3), w[1], w[2], round(w[3], 1), round(w[4], 1))
                      for w in worst if w[0] < 0]
    r["second"] = "%s|%s %.3f" % (worst[1][1], worst[1][2], worst[1][0]) if len(worst) > 1 else ""
    for key in [("- RCCA", "- RVA"), ("- RCCA", "- LCCA"), ("- RECA", "- RVA"),
                ("- RECA", "- LCCA"), ("- RCCA", "(17)"), ("- RECA", "(17)")]:
        k2 = key if key in res else (key[1], key[0])
        r["ref_%s_%s" % key] = res[k2][8.0][0] if k2 in res else float("nan")
    rows.append(r)
    if (ai + 1) % 25 == 0:
        print("  ...%d/%d" % (ai + 1, len(names)))
        sys.stdout.flush()

json.dump(rows, open("/tmp/out/check8_rows.json", "w"), indent=0, default=float)


def dist(vals, label, fmt="%.3f"):
    v = np.array([x for x in vals if x == x], float)
    if not len(v):
        print("%-38s  no data" % label)
        return
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print(("%-38s n=%3d  min/p5/p25/med/p75/p95/max = " + " / ".join([fmt] * 7))
          % ((label, len(v)) + tuple(q)))


print("")
print("=" * 118)
print("A. RCCA/RECA BIFURCATION GEOMETRY (declared radii, densified 0.25 mm)")
print("=" * 118)
dist([r["s_fork"] for r in rows], "RCCA arclength of the fork (mm)", "%.2f")
dist([r["d_fork"] for r in rows], "centre gap RECA[0]-to-RCCA (mm)", "%.3f")
dist([r["L_eca"] for r in rows], "RECA length (mm)", "%.2f")
dist([r["d_sep"] for r in rows], "RECA s where envelopes separate", "%.2f")
for T in TS:
    dist([r["g%g" % T] for r in rows], "min RCCA-RECA sep, T=%g excl" % T, "%.3f")
dist([r["reapp"] for r in rows], "min sep distal to separation pt", "%.3f")

print("")
print("=" * 118)
print("B. WHOLE-TREE MINIMUM over all branch pairs, T=8 mm junction exclusion")
print("=" * 118)
dist([r["gmin_tree"] for r in rows], "tree-wide min separation (mm)", "%.3f")
print("  pair holding the tree minimum: %s" % Counter(r["gmin_pair"] for r in rows).most_common())
print("  anatomies with ANY negative pair at T=8: %d" % sum(1 for r in rows if r["n_neg8"] > 0))
negs = Counter()
for r in rows:
    for g, a, b in r["neg_pairs"]:
        negs["%s|%s" % (a, b)] += 1
print("  negative pairs by identity: %s" % negs.most_common())

print("")
print("=" * 118)
print("C. REFERENCE PAIRS (T=8 mm exclusion)")
print("=" * 118)
for k in [k for k in rows[0] if k.startswith("ref_")]:
    dist([r[k] for r in rows], k[4:], "%.3f")

print("")
print("=" * 118)
print("D. OUTLIERS")
print("=" * 118)
print(" 12 tightest RCCA-RECA at T=8:")
for r in sorted(rows, key=lambda r: r["g8"])[:12]:
    print("  %-42s g=%7.3f  RCCA s=%7.2f / RECA s=%6.2f | fork s=%6.2f d_sep=%5.2f L_eca=%5.2f g5=%6.3f g12=%6.3f"
          % (r["name"], r["g8"], r["sR8"], r["sE8"], r["s_fork"], r["d_sep"],
             r["L_eca"], r["g5"], r["g12"]))
print(" 12 tightest tree-wide at T=8:")
for r in sorted(rows, key=lambda r: r["gmin_tree"])[:12]:
    print("  %-42s gmin=%7.3f pair=%-22s neg=%s"
          % (r["name"], r["gmin_tree"], r["gmin_pair"], r["neg_pairs"][:4]))
print(" 5 widest RCCA-RECA at T=8:")
for r in sorted(rows, key=lambda r: -r["g8"])[:5]:
    print("  %-42s g=%7.3f" % (r["name"], r["g8"]))
print(" 8 largest d_sep (widest bifurcation lumen):")
for r in sorted(rows, key=lambda r: -(r["d_sep"] if r["d_sep"] == r["d_sep"] else -1))[:8]:
    print("  %-42s d_sep=%6.2f  L_eca=%6.2f" % (r["name"], r["d_sep"], r["L_eca"]))
print(" anatomies where envelopes NEVER separate along RECA:")
bad = [r["name"] for r in rows if r["d_sep"] != r["d_sep"]]
print("  %s" % (bad if bad else "none"))
