"""Are the 25 RCCA courses independent anatomies, or deformations of one course?

Independent subjects would differ from the first millimetre. A procedural
generator that perturbs the distal segment leaves a shared proximal prefix.
Resamples every course to common arclength so point-count differences cannot
manufacture or hide agreement.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
HOSTCL = "/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb"
dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))


def rcca(cl_dir):
    br = [b for b in load_branches(cl_dir) if "RCCA" in str(b.name).upper()][0]
    c = np.asarray(br.coordinates, float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))
    return c, s, np.asarray(br.radii, float)


def resample(c, s, grid):
    return np.stack([np.interp(grid, s, c[:, i]) for i in range(3)], axis=1)


cur = {os.path.basename(d): rcca(os.path.join(d, "Centrelines_comb")) for d in dirs}
hc, hs, hr = rcca(HOSTCL)
L = min(min(v[1][-1] for v in cur.values()), hs[-1])
grid = np.linspace(0, L, 400)
R = {k: resample(c, s, grid) for k, (c, s, _) in cur.items()}
H = resample(hc, hs, grid)
ks = sorted(R)
A = np.stack([R[k] for k in ks])                      # 25 x 400 x 3

spread = np.linalg.norm(A - A.mean(0, keepdims=True), axis=2)   # 25 x 400
vs_host = np.linalg.norm(A - H[None], axis=2)

print("=" * 88)
print("RCCA COURSE DIVERGENCE vs ARCLENGTH  (common grid, 0 -> %.0f mm)" % L)
print("=" * 88)
print(f"  {'arclength':>10}  {'cohort spread':>14}  {'max pairwise':>13}  {'vs host patient':>16}")
print(f"  {'(mm)':>10}  {'(mm, rms)':>14}  {'(mm)':>13}  {'(mm, median)':>16}")
for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    i = min(int(f * (len(grid) - 1)), len(grid) - 1)
    pw = np.linalg.norm(A[:, i, :][:, None] - A[:, i, :][None], axis=2).max()
    print(f"  {grid[i]:10.1f}  {np.sqrt((spread[:, i]**2).mean()):14.2f}  {pw:13.2f}  "
          f"{np.median(vs_host[:, i]):16.2f}")

first = np.argmax(np.sqrt((spread**2).mean(0)) > 0.5)
print(f"\n  cohort courses coincide (rms < 0.5 mm) up to arclength {grid[first]:.1f} mm "
      f"= {100*grid[first]/L:.0f}% of the branch")
print(f"  host differs from the cohort by a median of {np.median(vs_host):.2f} mm "
      f"from arclength 0 -- i.e. everywhere")

print("\n" + "=" * 88)
print("RADIUS: is the calibre varied too?")
print("=" * 88)
rm = np.array([np.median(v[2]) for v in cur.values()])
print(f"  cohort median stated radius: {rm.min():.3f} .. {rm.max():.3f}  (host {np.median(hr):.3f})")
print(f"  ratio to host: {rm.min()/np.median(hr):.3f} .. {rm.max()/np.median(hr):.3f}")
