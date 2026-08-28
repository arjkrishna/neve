"""Whose anatomy is the TopCoW cohort's fixed template, and how does its RCCA vary?

The provenance test showed 15/16 centerlines identical across all 25 and 91% of
mesh vertices coincident -- one base with a varied RCCA. This identifies the base
by comparing the fixed centerlines against the host patient, and characterises the
RCCA variation.
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

host = {str(b.name): np.asarray(b.coordinates, float) for b in load_branches(HOSTCL)}
c001 = {str(b.name): np.asarray(b.coordinates, float) for b in load_branches(os.path.join(dirs[0], "Centrelines_comb"))}

print("=" * 96)
print("1. IS THE FIXED TEMPLATE OUR HOST PATIENT?  (host centerlines vs topcow_mr_001)")
print("=" * 96)
print(f"  host branches   : {len(host)}")
print(f"  topcow branches : {len(c001)}")
same = diff = 0
for k in sorted(set(host) | set(c001)):
    if k not in host or k not in c001:
        print(f"  {k:34s} ONLY IN {'host' if k in host else 'topcow'}")
        continue
    a, b = host[k], c001[k]
    if a.shape == b.shape:
        m = np.abs(a - b).max()
        verdict = "IDENTICAL" if m < 1e-6 else f"differs, max {m:8.3f} mm"
        same += m < 1e-6
        diff += m >= 1e-6
    else:
        verdict = f"different point count {a.shape[0]} vs {b.shape[0]}"
        diff += 1
    print(f"  {k:34s} {verdict}")
print(f"\n  -> {same} branches identical to the host, {diff} differ")

print()
print("=" * 96)
print("2. HOW DOES THE RCCA VARY ACROSS THE 25?")
print("=" * 96)
rc = {}
for d in dirs:
    br = [b for b in load_branches(os.path.join(d, "Centrelines_comb")) if "RCCA" in str(b.name).upper()][0]
    rc[os.path.basename(d)] = (np.asarray(br.coordinates, float), np.asarray(br.radii, float))
npts = {k: len(v[0]) for k, v in rc.items()}
print(f"  point counts: min {min(npts.values())}  max {max(npts.values())}  "
      f"distinct {len(set(npts.values()))}")
h = host.get("Centerline curve - RCCA.mrk")
if h is not None:
    print(f"  host RCCA points: {len(h)}")

ks = list(rc)
a = rc[ks[0]][0]
print(f"\n  displacement of each RCCA from topcow_mr_001's RCCA (same-index points where lengths match):")
for k in ks[1:9]:
    b = rc[k][0]
    n = min(len(a), len(b))
    dsp = np.linalg.norm(a[:n] - b[:n], axis=1)
    print(f"    {k:16s} n={len(b):4d}  median {np.median(dsp):6.2f}  max {dsp.max():7.2f} mm")

if h is not None:
    n = min(len(a), len(h))
    dsp = np.linalg.norm(a[:n] - h[:n], axis=1)
    print(f"\n  topcow_mr_001 RCCA vs HOST RCCA: median {np.median(dsp):.2f}  max {dsp.max():.2f} mm")

print("\n  radius profiles (median stated radius per anatomy):")
rmeds = np.array([np.median(v[1]) for v in rc.values()])
print(f"    cohort  min {rmeds.min():.2f}  median {np.median(rmeds):.2f}  max {rmeds.max():.2f}")
if h is not None:
    hb = [b for b in load_branches(HOSTCL) if "RCCA" in str(b.name).upper()][0]
    print(f"    host    {np.median(np.asarray(hb.radii, float)):.2f}")
