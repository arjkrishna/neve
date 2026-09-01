"""CHECK 8 exploration: topology of the 17-branch carotid tree, RCCA/RECA fork."""
import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/carotid/anatomies"
names = sorted(os.listdir(ROOT))
print("n anatomies:", len(names))

def arclen(C):
    return np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))))

for nm in [names[0], names[50], names[120], names[215]]:
    brs = load_branches(os.path.join(ROOT, nm, "Centrelines_comb"))
    print("=" * 100)
    print(nm, " nbranches=", len(brs))
    for b in brs:
        C = np.asarray(b.coordinates, float); R = np.asarray(b.radii, float)
        S = arclen(C)
        print("  %-28s n=%4d  L=%7.2f  spacing med=%.3f  r[min/med/max]=%.2f/%.2f/%.2f" % (
            str(b.name).replace("Centerline curve ", ""), len(C), S[-1],
            float(np.median(np.diff(S))) if len(C) > 1 else 0.0,
            R.min(), np.median(R), R.max()))
    # endpoint connectivity: for each branch endpoint, nearest other-branch point
    print("  -- endpoint proximity (start/end of each branch to nearest point of any other branch) --")
    for i, b in enumerate(brs):
        C = np.asarray(b.coordinates, float)
        for lbl, p in (("start", C[0]), ("end", C[-1])):
            best = (1e9, None, None)
            for j, o in enumerate(brs):
                if j == i: continue
                O = np.asarray(o.coordinates, float)
                d = np.linalg.norm(O - p, axis=1); k = int(np.argmin(d))
                if d[k] < best[0]:
                    best = (float(d[k]), str(o.name).replace("Centerline curve ", ""), k)
            if best[0] < 6.0:
                So = arclen(np.asarray(brs[[str(x.name) for x in brs].index([y.name for y in brs][0])].coordinates, float))
                print("     %-24s %-5s -> %-22s d=%6.3f at idx %d" % (
                    str(b.name).replace("Centerline curve ", ""), lbl, best[1], best[0], best[2]))
    # RCCA/RECA detail
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    ec = next(b for b in brs if "RECA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float); Sc = arclen(C)
    E = np.asarray(ec.coordinates, float); Se = arclen(E)
    D = np.linalg.norm(E[:, None, :] - C[None, :, :], axis=2)
    for j in range(len(E)):
        k = int(np.argmin(D[j]))
        print("     RECA idx %2d  s_eca=%6.2f  -> RCCA s=%7.2f  ctr_d=%6.3f  rE=%.2f rC=%.2f  g=%7.3f" % (
            j, Se[j], Sc[k], D[j, k], ec.radii[j], rc.radii[k],
            D[j, k] - float(ec.radii[j]) - float(rc.radii[k])))
