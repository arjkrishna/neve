"""Are the 25 TopCoW anatomies independent patients, or variations of one template?

Two signatures showed up in the quality audit that a cohort of 25 independent
segmentations cannot produce:
  * the longest mesh edge is 25.82 mm in every single one
  * clearance/stated-radius sits at 0.69-0.79 in every single one, against 1.07
    for the host patient's raw segmented surface

This tests both directly: shared vertex geometry between anatomies, whether the
non-navigated centerlines are identical, and where along the branch the clearance
deficit lives.
"""
import glob
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))


def pts(d):
    return pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean().points


print("=" * 100)
print("1. SHARED GEOMETRY — fraction of each mesh's vertices that coincide (<1e-4 mm) with anatomy 001")
print("=" * 100)
base = pts(dirs[0])
tb = cKDTree(base)
for d in dirs[:8]:
    p = pts(d)
    dist, _ = tb.query(p)
    print(f"  {os.path.basename(d):16s} {len(p):6d} verts   coincident with 001: "
          f"{100*np.mean(dist < 1e-4):5.1f}%   (median offset {np.median(dist):.3f} mm)")

print()
print("=" * 100)
print("2. CENTERLINES — which branches are identical across anatomies, which vary")
print("=" * 100)
ref = {str(b.name): np.asarray(b.coordinates, float) for b in load_branches(os.path.join(dirs[0], "Centrelines_comb"))}
agree = {k: 0 for k in ref}
nseen = {k: 0 for k in ref}
for d in dirs[1:]:
    cur = {str(b.name): np.asarray(b.coordinates, float) for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    for k, v in ref.items():
        if k not in cur:
            continue
        nseen[k] += 1
        w = cur[k]
        if w.shape == v.shape and np.abs(w - v).max() < 1e-6:
            agree[k] += 1
for k in sorted(ref, key=lambda x: -agree[x]):
    tag = "IDENTICAL in all" if agree[k] == nseen[k] and nseen[k] else ("varies" if agree[k] == 0 else "mixed")
    print(f"  {k:34s} identical in {agree[k]:2d}/{nseen[k]:2d} others   <- {tag}")

print()
print("=" * 100)
print("3. CLEARANCE DEFICIT vs STATED RADIUS along the RCCA — uniform erosion or a local pinch?")
print("=" * 100)


def profile(d, label):
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    f = m.faces.reshape(-1, 4)[:, 1:]
    v = m.points
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rg = np.random.default_rng(0)
    u = rg.random((20, len(f), 1))
    w = rg.random((20, len(f), 1))
    k = u + w > 1
    u = np.where(k, 1 - u, u)
    w = np.where(k, 1 - w, w)
    surf = np.vstack([v, (a + u * (b - a) + w * (c - a)).reshape(-1, 3)])
    br = [x for x in load_branches(os.path.join(d, "Centrelines_comb")) if "RCCA" in str(x.name).upper()][0]
    cc = np.asarray(br.coordinates, float)
    rr = np.asarray(br.radii, float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))))
    dd, _ = cKDTree(surf).query(cc)
    ratio = dd / np.maximum(rr, 1e-9)
    print(f"\n  {label}   (ratio = measured clearance / centerline's own stated radius)")
    print(f"    {'arclength band':>18}  {'stated r':>9} {'clearance':>10} {'ratio':>7}")
    edges = np.linspace(0, s[-1], 9)
    for i in range(8):
        sel = (s >= edges[i]) & (s < edges[i + 1])
        if sel.sum() == 0:
            continue
        print(f"    {edges[i]:7.0f}-{edges[i+1]:6.0f} mm  {np.median(rr[sel]):9.2f} "
              f"{np.median(dd[sel]):10.2f} {np.median(ratio[sel]):7.2f}")
    return ratio


profile(dirs[0], "topcow_mr_001")
profile([d for d in dirs if d.endswith("013")][0], "topcow_mr_013 (BLOCKED)")
