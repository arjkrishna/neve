"""Final pass: circumferential mesh resolution as a function of arclength.
If the distal tube is a 3-4 sided prism, the facet chord passes INSIDE the centerline
and clearance collapses regardless of the true lumen. Also confirms whether
topcow_mr_015's RCCA surface is truncated short of its centerline.
"""
import glob
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def load(name):
    if name == "HOST":
        vt = DualDeviceNav().vessel_tree
        mesh = pv.read(vt.mesh_path).triangulate().clean()
        brs = list(vt.branches)
    else:
        d0 = os.path.join(ROOT, name)
        mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
        brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    return mesh, np.asarray(rc.coordinates, float), np.asarray(rc.radii, float)


def ring_profile(mesh, C, R, slab=2.0):
    """At each station: how many mesh vertices form the local ring, and the max
    angular gap between them about the local tangent."""
    V = np.asarray(mesh.points)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    vt_ = cKDTree(V)
    n = np.zeros(len(C))
    gap = np.full(len(C), 360.0)
    for i in range(len(C)):
        idx = vt_.query_ball_point(C[i], 10.0)
        if not idx:
            continue
        P = V[idx] - C[i]
        t = T[i]
        P = P[np.abs(P @ t) < slab]
        n[i] = len(P)
        if len(P) < 3:
            continue
        rad = P - np.outer(P @ t, t)
        e1 = np.array([1.0, 0, 0]) - t * t[0]
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0, 1.0, 0]) - t * t[1]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        a = np.sort(np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360)
        gap[i] = np.diff(np.concatenate([a, [a[0] + 360]])).max()
    return n, gap


ALL = ["HOST"] + [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
                  if os.path.isdir(d)]

print("=" * 112)
print("R1. CIRCUMFERENTIAL RESOLUTION vs ARCLENGTH (vertices in a +-2mm slab, max angular gap)")
print("    a max gap > 180 deg means one facet chord passes on the FAR side of the centerline")
print("=" * 112)
rows = {}
for name in ALL:
    mesh, C, R = load(name)
    S = arclen(C)
    n, gap = ring_profile(mesh, C, R)
    rows[name] = (S, R, n, gap)

print(f"{'anatomy':>16} " + " ".join(f"{int(f*100):>4}%" for f in
                                     (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97)) +
      f" {'  nGap>180':>10} {'medN_prox':>10} {'medN_dist':>10}")
for tag, key in (("nVert", 2), ("maxGap", 3)):
    print(f"--- {tag} ---")
    for name in ALL:
        S, R, n, gap = rows[name]
        arr = rows[name][key]
        vals = [arr[int(f * (len(S) - 1))] for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97)]
        prox = n[S < 0.4 * S[-1]]
        dist = n[S > 0.8 * S[-1]]
        print(f"{name:>16} " + " ".join(f"{v:5.0f}" for v in vals) +
              f" {int((gap>180).sum()):10d} {np.median(prox):10.1f} {np.median(dist):10.1f}")

print("\n" + "=" * 112)
print("R2. Fraction of each RCCA whose local ring has <6 vertices / gap>120deg / gap>180deg")
print("=" * 112)
print(f"{'anatomy':>16} {'n<6':>8} {'n<4':>8} {'gap>120':>9} {'gap>180':>9} "
      f"{'first s with gap>180':>21} {'medGap prox':>12} {'medGap dist':>12}")
for name in ALL:
    S, R, n, gap = rows[name]
    f180 = np.where(gap > 180)[0]
    print(f"{name:>16} {100*(n<6).mean():7.0f}% {100*(n<4).mean():7.0f}% "
          f"{100*(gap>120).mean():8.0f}% {100*(gap>180).mean():8.0f}% "
          f"{(f'{S[f180[0]]:.0f}mm ({100*S[f180[0]]/S[-1]:.0f}%)' if len(f180) else '-'):>21} "
          f"{np.median(gap[S<0.4*S[-1]]):12.1f} {np.median(gap[S>0.8*S[-1]]):12.1f}")

print("\n" + "=" * 112)
print("R3. topcow_mr_015 / _018 / _026: is the RCCA SURFACE truncated short of its centerline?")
print("    distance from each station to the nearest surface point, and to the open rim")
print("=" * 112)
for name in ["topcow_mr_015", "topcow_mr_018", "topcow_mr_026", "topcow_mr_024", "HOST"]:
    mesh, C, R = load(name)
    S = arclen(C)
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    sd = np.array([imp.EvaluateFunction(p) for p in C])
    sgn = 1.0 if (sd > 0).mean() > 0.5 else -1.0
    be = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=False)
    rim = np.asarray(be.points) if be.n_points else np.zeros((0, 3))
    rt = cKDTree(rim) if len(rim) else None
    out = np.where(sgn * sd < 0)[0]
    print(f"  {name}: total_len {S[-1]:.1f}mm, stations outside surface: {len(out)}")
    if len(out):
        runs = np.split(out, np.where(np.diff(out) != 1)[0] + 1)
        for r in runs:
            dmax = np.abs(sd[r]).max()
            drim = rt.query(C[r])[0].min() if rt is not None else float("nan")
            print(f"     OUT run s {S[r[0]]:.1f}-{S[r[-1]]:.1f}mm ({len(r)} stations, "
                  f"{S[r[-1]]-S[r[0]]:.1f}mm), max depth outside {dmax:.2f}mm, "
                  f"nearest open rim {drim:.2f}mm, stated_r {R[r].min():.2f}-{R[r].max():.2f}")
    if rt is not None:
        dc = cKDTree(C).query(rim)[0]
        print(f"     open rim points: {len(rim)}, distance to RCCA centerline "
              f"{dc.min():.2f}-{dc.max():.2f}mm")
