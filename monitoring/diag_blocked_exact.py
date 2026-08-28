"""EXACT point-to-surface distance (vtkImplicitPolyDataDistance + vtkCellLocator),
no barycentric sampling. Signed => also tells us if the centerline leaves the lumen.
Tests the coarse-facet chord-error hypothesis quantitatively.
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
WIRE_R, CATH_R = 0.18, 0.35


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def exact_signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def curvature_at(c, s, win):
    R = np.full(len(c), np.inf)
    for i in range(len(c)):
        lo = np.searchsorted(s, s[i] - win)
        hi = np.searchsorted(s, s[i] + win) - 1
        if lo < 0 or hi >= len(c) or hi - lo < 2:
            continue
        A, B, Cc = c[lo], c[i], c[hi]
        a = np.linalg.norm(B - Cc)
        b = np.linalg.norm(A - Cc)
        e = np.linalg.norm(A - B)
        ar = 0.5 * np.linalg.norm(np.cross(B - A, Cc - A))
        R[i] = np.inf if ar < 1e-12 else (a * b * e) / (4 * ar)
    return R


def load(name):
    if name == "HOST":
        vt = DualDeviceNav().vessel_tree
        mesh = pv.read(vt.mesh_path).triangulate().clean()
        brs = list(vt.branches)
    else:
        d0 = os.path.join(ROOT, name)
        mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
        brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    return mesh, brs


def edge_stats(mesh):
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    e = np.stack([np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1),
                  np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1),
                  np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1)], 1)
    return f, v, e


ALL = ["HOST"] + [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
                  if os.path.isdir(d)]
FOCUS = ["topcow_mr_013", "topcow_mr_024", "topcow_mr_027", "topcow_mr_006",
         "topcow_mr_015", "topcow_mr_007", "topcow_mr_017", "HOST"]

print("=" * 122)
print("A. EXACT clearance (vtkImplicitPolyDataDistance), all anatomies, branch RCCA")
print("   d_eff = |d| inside the lumen, 0 where the centerline is OUTSIDE the closed surface")
print("=" * 122)
print(f"{'anatomy':>16} {'n_st':>5} {'len':>6} {'min|d|':>7} {'p05':>6} {'med':>6} "
      f"{'nBLK':>5} {'nCATH':>6} {'nOUT':>5} {'s@min':>7} {'pctbr':>6} {'r@min':>6} "
      f"{'d_last':>7} {'d_last1':>8} {'r_med':>6}")
store = {}
for name in ALL:
    mesh, brs = load(name)
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    R = np.asarray(rc.radii, float)
    S = arclen(C)
    sd = exact_signed(mesh, C)
    d = np.abs(sd)
    inside = sd < 0
    if inside.mean() < 0.5:
        inside = ~inside
    nout = int((~inside).sum())
    d_eff = np.where(inside, d, 0.0)
    blk = d_eff < WIRE_R
    cat = d_eff < CATH_R
    mi = int(np.argmin(d_eff))
    print(f"{name:>16} {len(C):5d} {S[-1]:6.1f} {d.min():7.3f} {np.percentile(d,5):6.3f} "
          f"{np.median(d):6.3f} {blk.sum():5d} {cat.sum():6d} {nout:5d} {S[mi]:7.1f} "
          f"{100*S[mi]/S[-1]:6.0f} {R[mi]:6.3f} {d_eff[-1]:7.3f} {d_eff[-2]:8.3f} "
          f"{np.median(R):6.3f}")
    store[name] = (mesh, brs, C, R, S, d, d_eff, inside, blk, cat)

print("\n" + "=" * 122)
print("B. TERMINAL-STATION (end-cap) EFFECT: clearance at the last 3 stations vs branch median")
print("=" * 122)
print(f"{'anatomy':>16} {'d[-3]':>7} {'d[-2]':>7} {'d[-1]':>7} {'med':>7} {'last/med':>9} "
      f"  blocked stations EXCLUDING last 2")
for name in ALL:
    _, _, C, R, S, d, de, ins, blk, cat = store[name]
    bi = np.where(blk)[0]
    bi2 = bi[bi < len(C) - 2]
    print(f"{name:>16} {de[-3]:7.3f} {de[-2]:7.3f} {de[-1]:7.3f} {np.median(de):7.3f} "
          f"{de[-1]/np.median(de):9.2f}   {[f'{S[i]:.0f}mm/{de[i]:.2f}' for i in bi2]}")

print("\n" + "=" * 122)
print("C. CHORD-ERROR PREDICTION at each focus anatomy's worst NON-TERMINAL station")
print("   longitudinal sagitta   = L^2 / (8*Rc)   (a flat facet chords a bend of radius Rc)")
print("   circumferential sagitta= L^2 / (8*r)    (the polygon inscribes a circle of radius r)")
print("   predicted clearance    = stated_r - longsag - circsag")
print("=" * 122)
print(f"{'anatomy':>16} {'s':>7} {'pctbr':>6} {'statedr':>8} {'Rc4':>7} {'RcPCT':>6} "
      f"{'L_loc':>6} {'longsag':>8} {'circsag':>8} {'pred_cl':>8} {'meas_cl':>8} {'meas-pred':>9}")
for name in FOCUS:
    mesh, brs, C, R, S, d, de, ins, blk, cat = store[name]
    f, v, e = edge_stats(mesh)
    cent = v[f].mean(1)
    ctree = cKDTree(cent)
    kR = curvature_at(C, S, 4.0)
    kfin = kR[np.isfinite(kR)]
    core = np.arange(len(C) - 2)
    mi = int(core[np.argmin(de[core])])
    _, ii = ctree.query(C[mi], k=30)
    Lloc = float(np.mean(e[ii]))
    r = R[mi]
    Rc = kR[mi]
    lsag = Lloc ** 2 / (8 * Rc) if np.isfinite(Rc) else np.nan
    csag = Lloc ** 2 / (8 * r)
    pred = r - lsag - csag
    print(f"{name:>16} {S[mi]:7.1f} {100*S[mi]/S[-1]:6.0f} {r:8.3f} {Rc:7.1f} "
          f"{100*(kfin<Rc).mean():6.0f} {Lloc:6.2f} {lsag:8.3f} {csag:8.3f} {pred:8.3f} "
          f"{de[mi]:8.3f} {de[mi]-pred:9.3f}")

print("\n" + "=" * 122)
print("D. COHORT-WIDE: is the clearance deficit driven by stated_r and by curvature?")
print("=" * 122)
rows = []
for name in ALL:
    mesh, brs, C, R, S, d, de, ins, blk, cat = store[name]
    kR = curvature_at(C, S, 4.0)
    ok = np.arange(len(C) - 2)
    rows.append(np.stack([R[ok], kR[ok], de[ok], de[ok] / R[ok],
                          np.full(len(ok), 0.0 if name == "HOST" else 1.0)], 1))
A = np.vstack(rows)
A = A[np.isfinite(A).all(1)]
hostm = A[:, 4] == 0
coh = A[:, 4] == 1
print(f"  stations: host {int(hostm.sum())}, cohort {int(coh.sum())}")
print(f"  {'group':>8} {'r bin (mm)':>12} {'Rc bin (mm)':>12} {'n':>6} {'med c/r':>9} "
      f"{'med clear':>10} {'p05 clear':>10} {'min clear':>10}")
for tag, m0 in (("cohort", coh), ("host", hostm)):
    for rlo, rhi in [(0, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 2.5), (2.5, 99)]:
        for klo, khi in [(0, 6), (6, 12), (12, 1e9)]:
            m = m0 & (A[:, 0] >= rlo) & (A[:, 0] < rhi) & (A[:, 1] >= klo) & (A[:, 1] < khi)
            if m.sum() < 4:
                continue
            kh = khi if khi < 1e8 else 999
            print(f"  {tag:>8} {f'{rlo}-{rhi}':>12} {f'{klo}-{kh}':>12} {int(m.sum()):6d} "
                  f"{np.median(A[m,3]):9.2f} {np.median(A[m,2]):10.3f} "
                  f"{np.percentile(A[m,2],5):10.3f} {A[m,2].min():10.3f}")

print("\n  narrow (stated_r<1.6) AND tight (Rc4<6) station counts per anatomy:")
for name in ALL:
    mesh, brs, C, R, S, d, de, ins, blk, cat = store[name]
    kR = curvature_at(C, S, 4.0)
    m = (R < 1.6) & (kR < 6)
    print(f"    {name:>16}  narrow&tight {int(m.sum()):3d}/{len(C)}   min stated_r {R.min():.3f}"
          f"   n(r<1.3) {int((R<1.3).sum()):3d}   n(r<1.6) {int((R<1.6).sum()):3d}"
          f"   n(Rc4<6) {int((kR<6).sum()):3d}")
