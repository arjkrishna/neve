"""CHECK 8 supplement: locate every minimum, and reproduce the audit22 protocol
(RCCA restricted to s>=135 mm) so the numbers are comparable to the 49-set's
'+7.5 mm RCCA-to-RVA' and '+0.5 to +0.9 mm on the four repaired anatomies'.
"""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from collections import Counter
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
STEP = 0.25
TOUCH = 1.0


def densify(C, R, step=STEP):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(int(np.ceil(S[-1] / step)) + 1, len(C))
    Sn = np.linspace(0.0, S[-1], n)
    return (np.stack([np.interp(Sn, S, C[:, k]) for k in range(3)], axis=1),
            np.interp(Sn, S, R), Sn)


def short(n):
    return str(n).replace("Centerline curve ", "").replace(".mrk", "")


def gmin(A, ra, sa, B, rb, sb, chunk=500):
    best = (np.inf, 0.0, 0.0, 0.0)
    for i0 in range(0, len(A), chunk):
        blk = A[i0:i0 + chunk]
        Dm = np.sqrt(((blk[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        G = Dm - ra[i0:i0 + chunk][:, None] - rb[None, :]
        i, j = np.unravel_index(np.argmin(G), G.shape)
        if G[i, j] < best[0]:
            best = (float(G[i, j]), float(sa[i0 + i]), float(sb[j]), float(Dm[i, j]))
    return best


names = sorted(os.listdir(ROOT))
names = [n for n in names if os.path.isdir(os.path.join(ROOT, n, "Centrelines_comb"))]
rows = []
for ai, nm in enumerate(names):
    brs = load_branches(os.path.join(ROOT, nm, "Centrelines_comb"))
    bd = {short(b.name): densify(np.asarray(b.coordinates, np.float64),
                                 np.asarray(b.radii, np.float64)) for b in brs}
    C, rC, sC = bd["- RCCA"]
    E, rE, sE = bd["- RECA"]

    # RECA divergence profile against the WHOLE RCCA
    gv = np.zeros(len(E)); sCat = np.zeros(len(E)); dv = np.zeros(len(E))
    for j0 in range(0, len(E), 400):
        blk = E[j0:j0 + 400]
        M = np.sqrt(((blk[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        G = M - rE[j0:j0 + 400][:, None] - rC[None, :]
        k = np.argmin(G, axis=1)
        idx = np.arange(len(blk))
        gv[j0:j0 + 400] = G[idx, k]; dv[j0:j0 + 400] = M[idx, k]; sCat[j0:j0 + 400] = sC[k]
    pos = np.where(gv >= 0.0)[0]
    d_sep = float(sE[pos[0]]) if pos.size else float("nan")
    r = {"name": nm, "d_sep": d_sep, "L_eca": float(sE[-1])}
    for lab, off in (("p3", 3.0), ("p5", 5.0), ("p8", 8.0)):
        m = sE >= (d_sep + off)
        if m.any():
            k = int(np.argmin(gv[m]))
            r["reap_" + lab] = float(gv[m][k])
            r["reap_%s_sE" % lab] = float(sE[m][k])
            r["reap_%s_sC" % lab] = float(sCat[m][k])
            r["reap_%s_d" % lab] = float(dv[m][k])
        else:
            r["reap_" + lab] = float("nan")
            r["reap_%s_sE" % lab] = float("nan")
            r["reap_%s_sC" % lab] = float("nan")
            r["reap_%s_d" % lab] = float("nan")
    # separation at the RECA tip specifically
    r["tip_g"] = float(gv[-1]); r["tip_sC"] = float(sCat[-1]); r["tip_d"] = float(dv[-1])

    # audit22 protocol: RCCA distal to s=135 vs every other branch
    sel = sC >= 135.0
    Cs, rCs, sCs = C[sel], rC[sel], sC[sel]
    aud = []
    for k2, (B, rb, sb) in bd.items():
        if k2 == "- RCCA":
            continue
        g, s1, s2, d = gmin(Cs, rCs, sCs, B, rb, sb)
        aud.append((g, k2, s1, s2, d))
    aud.sort(key=lambda x: x[0])
    r["aud_min"] = aud[0][0]; r["aud_pair"] = aud[0][1]
    r["aud_sR"] = aud[0][2]; r["aud_sB"] = aud[0][3]
    r["aud_all"] = {a[1]: round(a[0], 3) for a in aud}
    # same protocol but with RECA as the probe (whole RECA vs all non-RCCA)
    aud2 = []
    for k2, (B, rb, sb) in bd.items():
        if k2 in ("- RECA", "- RCCA"):
            continue
        g, s1, s2, d = gmin(E, rE, sE, B, rb, sb)
        aud2.append((g, k2, s1, s2))
    aud2.sort(key=lambda x: x[0])
    r["eca_min"] = aud2[0][0]; r["eca_pair"] = aud2[0][1]
    r["eca_sE"] = aud2[0][2]; r["eca_sB"] = aud2[0][3]

    # LVA/RVA vertebrobasilar confluence
    L, rL, sL = bd["- LVA"]; V, rV, sV = bd["- RVA"]
    g, s1, s2, d = gmin(L, rL, sL, V, rV, sV)
    r["lva_rva_g"] = g; r["lva_rva_sL"] = s1; r["lva_rva_sV"] = s2; r["lva_rva_d"] = d
    r["lva_rva_endgap"] = float(np.linalg.norm(L[-1] - V[-1]))
    rows.append(r)
    if (ai + 1) % 25 == 0:
        print("  ...%d/%d" % (ai + 1, len(names)))
        sys.stdout.flush()

json.dump(rows, open("/tmp/out/check8_v2.json", "w"), indent=0, default=float)


def dist(vals, label, fmt="%.3f"):
    v = np.array([x for x in vals if x == x], float)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print(("%-42s n=%3d  min/p5/p25/med/p75/p95/max = " + " / ".join([fmt] * 7))
          % ((label, len(v)) + tuple(q)))


print("")
print("=" * 120)
print("A. RCCA-RECA RE-APPROACH distal to the bifurcation (declared radii)")
print("=" * 120)
for lab in ("p3", "p5", "p8"):
    dist([r["reap_" + lab] for r in rows], "min sep, RECA s >= d_sep + %s mm" % lab[1])
dist([r["tip_g"] for r in rows], "separation at the RECA distal tip")
print(" 15 tightest re-approaches (d_sep+3 protocol):")
for r in sorted(rows, key=lambda r: r["reap_p3"])[:15]:
    print("  %-42s g=%7.3f  RECA s=%6.2f (L=%5.2f) / RCCA s=%7.2f  ctr_d=%6.3f  d_sep=%5.2f"
          % (r["name"], r["reap_p3"], r["reap_p3_sE"], r["L_eca"], r["reap_p3_sC"],
             r["reap_p3_d"], r["d_sep"]))
print(" negative at d_sep+3: %d/216 ; at d_sep+5: %d ; at d_sep+8: %d"
      % (sum(1 for r in rows if r["reap_p3"] < 0),
         sum(1 for r in rows if r["reap_p5"] < 0),
         sum(1 for r in rows if r["reap_p8"] < 0)))

print("")
print("=" * 120)
print("B. AUDIT22 PROTOCOL -- RCCA distal to s=135 mm vs every other branch")
print("   (this is the protocol behind the 49-set's +7.5 mm RCCA-to-RVA reference)")
print("=" * 120)
dist([r["aud_min"] for r in rows], "min over all neighbours")
print("  neighbour holding it: %s" % Counter(r["aud_pair"] for r in rows).most_common(6))
for br in ["- RVA", "- RECA", "- LCCA", "- LVA", "(17)", "(19)"]:
    dist([r["aud_all"].get(br, float("nan")) for r in rows], "  RCCA(s>=135) vs %s" % br)
print(" 12 tightest:")
for r in sorted(rows, key=lambda r: r["aud_min"])[:12]:
    print("  %-42s g=%7.3f vs %-8s at RCCA s=%7.2f / other s=%7.2f"
          % (r["name"], r["aud_min"], r["aud_pair"], r["aud_sR"], r["aud_sB"]))

print("")
print("=" * 120)
print("C. RECA vs every branch OTHER than RCCA")
print("=" * 120)
dist([r["eca_min"] for r in rows], "min over all non-RCCA neighbours")
print("  neighbour holding it: %s" % Counter(r["eca_pair"] for r in rows).most_common(6))
print(" 10 tightest:")
for r in sorted(rows, key=lambda r: r["eca_min"])[:10]:
    print("  %-42s g=%7.3f vs %-8s at RECA s=%6.2f / other s=%7.2f"
          % (r["name"], r["eca_min"], r["eca_pair"], r["eca_sE"], r["eca_sB"]))

print("")
print("=" * 120)
print("D. LVA/RVA vertebrobasilar confluence (the only other negative pair)")
print("=" * 120)
dist([r["lva_rva_g"] for r in rows], "LVA-RVA min separation")
dist([r["lva_rva_endgap"] for r in rows], "gap between the two distal endpoints")
print("  negative: %d/216" % sum(1 for r in rows if r["lva_rva_g"] < 0))
print("  location of the min (LVA s, RVA s): %s"
      % Counter((round(r["lva_rva_sL"]), round(r["lva_rva_sV"])) for r in rows).most_common(5))
print("  endpoint-gap classes: %s"
      % Counter(round(r["lva_rva_endgap"], 2) for r in rows).most_common(8))
