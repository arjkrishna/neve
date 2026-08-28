"""Third pass. Three specific questions:
  T1  Are these meshes CLOSED?  Where are the open boundary rims, and are the blocked
      stations sitting on one?  (an open distal end explains terminal blockage in every
      anatomy including the host, and is a pure artifact)
  T2  Is the narrow stated_r at the offending stations SUBJECT-SPECIFIC (present in that
      one anatomy only => real source anatomy) or shared with the cohort (=> generator)?
  T3  What is the TRUE circumferential facet chord at those stations (count the vertices
      around the tube), hence the true inscribed-polygon sagitta?
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
    return mesh, brs, np.asarray(rc.coordinates, float), np.asarray(rc.radii, float)


def signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


ALL = ["HOST"] + [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
                  if os.path.isdir(d)]

print("=" * 116)
print("T1. MESH TOPOLOGY: open boundary rims, and their distance to the RCCA distal terminus")
print("=" * 116)
print(f"{'anatomy':>16} {'cells':>6} {'pts':>6} {'openEdge':>9} {'nonManif':>9} "
      f"{'rimPts':>7} {'dist(RCCA end -> nearest rim pt)':>32} {'rim s-range on RCCA':>22}")
topo = {}
for name in ALL:
    mesh, brs, C, R = load(name)
    S = arclen(C)
    be = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=False)
    nm = mesh.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=True)
    rim = np.asarray(be.points) if be.n_points else np.zeros((0, 3))
    if len(rim):
        ct = cKDTree(C)
        dr, ir = ct.query(rim)
        near = ir[dr < 6.0]
        srange = f"{S[near].min():.0f}-{S[near].max():.0f}mm" if len(near) else "none within 6mm"
        dend = np.linalg.norm(rim - C[-1], axis=1).min()
    else:
        srange, dend = "-", float("nan")
    print(f"{name:>16} {mesh.n_cells:6d} {mesh.n_points:6d} {be.n_cells:9d} {nm.n_cells:9d} "
          f"{len(rim):7d} {dend:32.2f} {srange:>22}")
    topo[name] = (mesh, C, R, S, rim)

print("\n" + "=" * 116)
print("T1b. For the OUT / blocked stations: how far outside, and how far from an open rim?")
print("=" * 116)
for name in ALL:
    mesh, C, R, S, rim = topo[name]
    sd = signed(mesh, C)
    ins = sd < 0
    if ins.mean() < 0.5:
        ins = ~ins
    de = np.where(ins, np.abs(sd), 0.0)
    bad = np.where((~ins) | (de < 0.35))[0]
    if not len(bad):
        continue
    rt = cKDTree(rim) if len(rim) else None
    print(f"  {name}:")
    for i in bad:
        drim = rt.query(C[i])[0] if rt is not None else float("nan")
        print(f"     s={S[i]:7.1f} ({100*S[i]/S[-1]:3.0f}%)  stated_r {R[i]:5.3f}  "
              f"signed_d {sd[i]:+7.3f}  {'OUTSIDE' if not ins[i] else 'inside '}  "
              f"dist_to_open_rim {drim:7.2f}  "
              f"{'TERMINAL(last3)' if i >= len(C)-3 else ''}")

print("\n" + "=" * 116)
print("T2. RADIUS PROVENANCE: is the narrow stated_r at 013/015/006 subject-specific?")
print("    resample every RCCA radius profile onto a common arclength grid")
print("=" * 116)
prof = {}
for name in ALL:
    mesh, C, R, S, rim = topo[name]
    prof[name] = (S, R)
L = min(v[0][-1] for v in prof.values())
g = np.arange(0, L, 2.0)
M = {k: np.interp(g, v[0], v[1]) for k, v in prof.items()}
coh = np.stack([M[k] for k in ALL if k != "HOST"])
print(f"{'s(mm)':>7} {'cohort p05':>11} {'cohort med':>11} {'cohort p95':>11} {'HOST':>7} "
      f"{'mr_013':>8} {'mr_015':>8} {'mr_006':>8} {'mr_024':>8} {'mr_027':>8} {'mr_007':>8}")
for j in range(0, len(g), 5):
    print(f"{g[j]:7.0f} {np.percentile(coh[:,j],5):11.3f} {np.median(coh[:,j]):11.3f} "
          f"{np.percentile(coh[:,j],95):11.3f} {M['HOST'][j]:7.3f} "
          f"{M['topcow_mr_013'][j]:8.3f} {M['topcow_mr_015'][j]:8.3f} {M['topcow_mr_006'][j]:8.3f} "
          f"{M['topcow_mr_024'][j]:8.3f} {M['topcow_mr_027'][j]:8.3f} {M['topcow_mr_007'][j]:8.3f}")
print("\n  z-score of each focus anatomy's radius vs the cohort, in its offending window:")
for name, lo, hi in [("topcow_mr_013", 200, 222), ("topcow_mr_015", 120, 140),
                     ("topcow_mr_006", 195, 208), ("topcow_mr_024", 185, 201),
                     ("topcow_mr_027", 205, 220), ("topcow_mr_007", 210, 230)]:
    m = (g >= lo) & (g <= hi)
    if not m.any():
        print(f"    {name}: window {lo}-{hi} beyond common grid (L={L:.0f})")
        continue
    mu = coh[:, m].mean(0)
    sd_ = coh[:, m].std(0)
    z = (M[name][m] - mu) / np.maximum(sd_, 1e-6)
    print(f"    {name:>16} s {lo}-{hi}mm  own_r {M[name][m].min():.2f}-{M[name][m].max():.2f}  "
          f"cohort mean {mu.mean():.2f}+-{sd_.mean():.2f}  z min {z.min():+.2f} med {np.median(z):+.2f}")

print("\n" + "=" * 116)
print("T3. TRUE CIRCUMFERENTIAL DISCRETISATION at the offending stations")
print("    vertices within +-2mm axially of the station, ordered by angle about the tangent")
print("=" * 116)
FOC = [("topcow_mr_013", [207.0, 208.0, 216.8, 217.8]),
       ("topcow_mr_024", [195.2, 200.1]),
       ("topcow_mr_027", [211.4, 222.3]),
       ("topcow_mr_006", [200.6, 212.6]),
       ("topcow_mr_015", [128.8, 129.8]),
       ("topcow_mr_007", [221.7]),
       ("topcow_mr_017", [223.8]),
       ("HOST", [223.9])]
print(f"{'anatomy':>16} {'s':>7} {'r':>6} {'nVert':>6} {'maxGapDeg':>10} {'medGapDeg':>10} "
      f"{'chord':>7} {'sagitta':>8} {'r-sag':>7} {'measured':>9} {'vertRadMin':>11} {'vertRadMed':>11}")
for name, slist in FOC:
    mesh, C, R, S, rim = topo[name]
    V = np.asarray(mesh.points)
    sd = signed(mesh, C)
    ins = sd < 0
    if ins.mean() < 0.5:
        ins = ~ins
    de = np.where(ins, np.abs(sd), 0.0)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    vt_ = cKDTree(V)
    for s0 in slist:
        i = int(np.argmin(np.abs(S - s0)))
        t = T[i]
        idx = vt_.query_ball_point(C[i], 8.0)
        if not idx:
            continue
        P = V[idx] - C[i]
        ax = P @ t
        sel = np.abs(ax) < 2.0
        P = P[sel]
        if len(P) < 3:
            print(f"{name:>16} {S[i]:7.1f} {R[i]:6.3f} {len(P):6d}   <3 vertices in slab")
            continue
        rad = P - np.outer(P @ t, t)
        rr = np.linalg.norm(rad, axis=1)
        e1 = np.array([1.0, 0, 0])
        e1 = e1 - t * (e1 @ t)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0, 1.0, 0]) - t * (np.array([0, 1.0, 0]) @ t)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        ang = np.sort(np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360)
        gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
        mg = gaps.max()
        rm = np.median(rr)
        chord = 2 * rm * np.sin(np.radians(mg) / 2)
        sag = rm * (1 - np.cos(np.radians(mg) / 2))
        print(f"{name:>16} {S[i]:7.1f} {R[i]:6.3f} {len(P):6d} {mg:10.1f} {np.median(gaps):10.1f} "
              f"{chord:7.2f} {sag:8.3f} {rm-sag:7.3f} {de[i]:9.3f} {rr.min():11.3f} {rm:11.3f}")
