"""HANDOFF 11.3 sensitivity: how much does densification change the CHECK-8 minimum?
Native station spacing (~1 mm) vs 0.25 mm, same estimator, same pairs."""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"


def densify(C, R, step):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate(([0.0], np.cumsum(seg)))
    if step is None:
        return C, R, S
    n = max(int(np.ceil(S[-1] / step)) + 1, len(C))
    Sn = np.linspace(0.0, S[-1], n)
    return (np.stack([np.interp(Sn, S, C[:, k]) for k in range(3)], axis=1),
            np.interp(Sn, S, R), Sn)


def short(n):
    return str(n).replace("Centerline curve ", "").replace(".mrk", "")


def gmin(A, ra, B, rb):
    best = np.inf
    for i0 in range(0, len(A), 500):
        blk = A[i0:i0 + 500]
        G = (np.sqrt(((blk[:, None, :] - B[None, :, :]) ** 2).sum(-1))
             - ra[i0:i0 + 500][:, None] - rb[None, :])
        best = min(best, float(G.min()))
    return best


names = sorted(os.listdir(ROOT))
names = [n for n in names if os.path.isdir(os.path.join(ROOT, n, "Centrelines_comb"))]
d_native, d_fine = [], []
a_native, a_fine = [], []
for nm in names:
    brs = load_branches(os.path.join(ROOT, nm, "Centrelines_comb"))
    raw = {short(b.name): (np.asarray(b.coordinates, np.float64),
                           np.asarray(b.radii, np.float64)) for b in brs}
    for step, dd, aa in ((None, d_native, a_native), (0.25, d_fine, a_fine)):
        bd = {k: densify(v[0], v[1], step) for k, v in raw.items()}
        C, rC, sC = bd["- RCCA"]
        E, rE, sE = bd["- RECA"]
        # RCCA-RECA distal to the fork: RECA stations >= 12 mm (max observed d_sep 14.5,
        # so use a fixed generous 15 mm to be strictly beyond every bifurcation)
        m = sE >= 15.0
        dd.append(gmin(E[m], rE[m], C, rC) if m.any() else float("nan"))
        # audit protocol RCCA s>=135 vs RVA
        V, rV, sV = bd["- RVA"]
        sel = sC >= 135.0
        aa.append(gmin(C[sel], rC[sel], V, rV))

d_native = np.array(d_native); d_fine = np.array(d_fine)
a_native = np.array(a_native); a_fine = np.array(a_fine)


def q(v, lab):
    p = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    print(("%-40s min/p5/p25/med/p75/p95/max = " + " / ".join(["%7.3f"] * 7)) % ((lab,) + tuple(p)))


print("RCCA-RECA min separation, RECA s>=15 mm (strictly beyond every bifurcation):")
q(d_native, "  native ~1.0 mm stations")
q(d_fine, "  densified 0.25 mm")
d = d_native - d_fine
q(d, "  native minus densified (optimism)")
print("  anatomies where native is optimistic by >0.10 mm: %d/216" % int((d > 0.10).sum()))
print("  max optimism: %.3f mm on %s" % (d.max(), names[int(np.argmax(d))]))
print("")
print("RCCA(s>=135)-RVA min separation:")
q(a_native, "  native ~1.0 mm stations")
q(a_fine, "  densified 0.25 mm")
da = a_native - a_fine
q(da, "  native minus densified (optimism)")
print("  anatomies where native is optimistic by >0.10 mm: %d/216" % int((da > 0.10).sum()))
print("  max optimism: %.3f mm on %s" % (da.max(), names[int(np.argmax(da))]))
print("")
print("sign check: any negative at 0.25 mm that native missed (>=0 native, <0 fine)?")
print("  RCCA-RECA: %d ; RCCA-RVA: %d"
      % (int(((d_native >= 0) & (d_fine < 0)).sum()), int(((a_native >= 0) & (a_fine < 0)).sum())))
print("  fine-grid negatives: RCCA-RECA %d, RCCA-RVA %d"
      % (int((d_fine < 0).sum()), int((a_fine < 0).sum())))
json.dump({"names": names, "d_native": d_native.tolist(), "d_fine": d_fine.tolist(),
           "a_native": a_native.tolist(), "a_fine": a_fine.tolist()},
          open("/tmp/out/check8_dens.json", "w"))
