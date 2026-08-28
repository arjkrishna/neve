import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/results_topbrain/anatomies"
d0 = os.path.join(ROOT, "topcow_mr_023")
brs = load_branches(os.path.join(d0, "Centrelines_comb"))
for b in brs:
    C = np.asarray(b.coordinates, float)
    L = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    print("%-32s n=%4d L=%7.1f  first=%s  last=%s" % (
        str(b.name), len(C), L, np.round(C[0], 1), np.round(C[-1], 1)))

print()
print("H0 mr023 tip at step1 = (26.3? no) insertion tip3d=(33.5,40.2,395.5)")
print("H0 mr023 ep1 pid198 target=(34.5,44.7,579.7); stuck tip=(19.2,63.8,413.8)")
rc = next(b for b in brs if "RCCA" in str(b.name).upper())
C = np.asarray(rc.coordinates, float)
for q in [(33.5, 40.2, 395.5), (26.7, 55.2, 416.6), (19.2, 63.8, 413.8), (34.5, 44.7, 579.7)]:
    q = np.array(q)
    d = np.linalg.norm(C - q, axis=1)
    print("q=%s  nearest RCCA station idx=%d d=%.2f" % (q, int(d.argmin()), float(d.min())))
