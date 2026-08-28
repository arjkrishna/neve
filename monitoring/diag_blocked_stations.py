"""Root-cause the 3 BLOCKED + 2 marginal cohort anatomies.

Discriminates: genuine narrow source anatomy  vs  coarse-facet chord artifact
vs  a FOREIGN structure (graft crossing / self-intersection) intruding.
"""
import glob, os, sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
WIRE_R = 0.18
CATH_R = 0.35
CASES = ["topcow_mr_013", "topcow_mr_024", "topcow_mr_027",
         "topcow_mr_006", "topcow_mr_015",
         "topcow_mr_007", "topcow_mr_017"]


def dense_points(mesh, per_tri=40, seed=0):
    """Sample per_tri barycentric points per triangle. Returns pts, owning face idx."""
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rng = np.random.default_rng(seed)
    u = rng.random((per_tri, len(f), 1)); w = rng.random((per_tri, len(f), 1))
    m = (u + w) > 1
    u = np.where(m, 1 - u, u); w = np.where(m, 1 - w, w)
    pts = (a + u * (b - a) + w * (c - a)).reshape(-1, 3)
    fid = np.tile(np.arange(len(f)), per_tri)
    # include vertices too, attributed to their first face
    vid = np.full(len(v), -1)
    for j in range(3):
        vid[f[:, j]] = np.arange(len(f))
    return np.vstack([v, pts]), np.concatenate([vid, fid])


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def resample_uniform(c, step=0.25):
    s = arclen(c)
    g = np.arange(0, s[-1], step)
    return np.stack([np.interp(g, s, c[:, i]) for i in range(3)], 1), g


def curvature_at(c, s, win_mm):
    """Radius of curvature via circumcircle of points +-win_mm along arclength."""
    R = np.full(len(c), np.inf)
    for i in range(len(c)):
        lo = np.searchsorted(s, s[i] - win_mm); hi = np.searchsorted(s, s[i] + win_mm) - 1
        if lo < 0 or hi >= len(c) or hi - lo < 2:
            continue
        A, B, C = c[lo], c[i], c[hi]
        a = np.linalg.norm(B - C); b = np.linalg.norm(A - C); cc = np.linalg.norm(A - B)
        ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
        R[i] = np.inf if ar < 1e-12 else (a * b * cc) / (4 * ar)
    return R


def tangents(c):
    t = np.gradient(c, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True); n[n == 0] = 1
    return t / n


def analyse(name):
    d0 = os.path.join(ROOT, name)
    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float); R = np.asarray(rc.radii, float)
    S = arclen(C)
    surf, fid = dense_points(mesh)
    tree = cKDTree(surf)
    d, idx = tree.query(C)
    P = surf[idx]                      # nearest surface point per station
    T = tangents(C)
    axial = np.abs(np.einsum('ij,ij->i', (P - C) / np.maximum(d, 1e-9)[:, None], T))
    kR = curvature_at(C, S, 4.0)       # radius of curvature, +-4 mm window
    kR8 = curvature_at(C, S, 8.0)

    # foreign-structure test: nearest OTHER branch centerline to each nearest surface pt
    others = {}
    for b in brs:
        if "RCCA" in str(b.name).upper():
            continue
        cc, _ = resample_uniform(np.asarray(b.coordinates, float), 0.25)
        others[str(b.name)] = cKDTree(cc)
    rc_dense, _ = resample_uniform(C, 0.25)
    rc_tree = cKDTree(rc_dense)

    # is each station inside the closed surface?
    ptsC = pv.PolyData(C)
    try:
        enc = ptsC.select_enclosed_points(mesh, tolerance=1e-6, check_surface=False)
        inside = np.asarray(enc["SelectedPoints"], bool)
    except Exception as e:
        inside = np.full(len(C), -1); print("  (enclosed-points failed:", e, ")")

    blocked = d < WIRE_R
    marg = d < CATH_R
    print("\n" + "=" * 108)
    print(f"### {name}   cells={mesh.n_cells}  stations={len(C)}  total_arclen={S[-1]:.1f} mm")
    print(f"    clearance min {d.min():.3f}  p05 {np.percentile(d,5):.3f}  med {np.median(d):.3f}")
    print(f"    stated_r  min {R.min():.3f}  med {np.median(R):.3f}  max {R.max():.3f}")
    print(f"    blocked(<{WIRE_R}) stations {blocked.sum()}   cath-marginal(<{CATH_R}) {marg.sum()}")
    print(f"    stations OUTSIDE the closed surface: {int((inside==0).sum()) if inside.dtype==bool else 'n/a'}")
    kfin = kR[np.isfinite(kR)]
    print(f"    branch radius-of-curvature (+-4mm): p05 {np.percentile(kfin,5):.1f}  "
          f"med {np.median(kfin):.1f}  p95 {np.percentile(kfin,95):.1f} mm")

    # pick focus stations
    if blocked.any():
        foci = [int(np.argmin(d))]
    else:
        foci = [int(np.argmin(d))]
    for i0 in foci:
        s0 = S[i0]
        sel = np.where(np.abs(S - s0) <= 20.0)[0]
        print(f"\n  --- window +-20 mm around arclength {s0:.1f} mm (min-clearance station) ---")
        print(f"  {'s(mm)':>7} {'stated_r':>9} {'clear':>7} {'c/r':>6} {'Rcurv4':>8} {'Rcurv8':>8} "
              f"{'axial':>6} {'in?':>4} {'nearest-surf-pt: closest branch':>34} {'dist':>7} {'dRCCA':>7}")
        for i in sel:
            dr, _ = rc_tree.query(P[i])
            bn, bd = "-", np.inf
            for k, t in others.items():
                dd, _ = t.query(P[i])
                if dd < bd:
                    bd, bn = dd, k
            flag = "***" if d[i] < WIRE_R else (" ** " if d[i] < CATH_R else "")
            ins = ("in" if inside[i] else "OUT") if inside.dtype == bool else "?"
            print(f"  {S[i]:7.1f} {R[i]:9.3f} {d[i]:7.3f} {d[i]/R[i]:6.2f} "
                  f"{kR[i]:8.1f} {kR8[i]:8.1f} {axial[i]:6.2f} {ins:>4} "
                  f"{bn[:34]:>34} {bd:7.2f} {dr:7.2f} {flag}")

    if blocked.any():
        bi = np.where(blocked)[0]
        runs = np.split(bi, np.where(np.diff(bi) != 1)[0] + 1)
        print(f"\n  blocked runs: {[(f'{S[r[0]]:.1f}-{S[r[-1]]:.1f}mm', len(r)) for r in runs]}")
        for r in runs:
            print(f"    run @{S[r[0]]:.1f}-{S[r[-1]]:.1f} mm span {S[r[-1]]-S[r[0]]:.2f} mm, "
                  f"{len(r)} stations, stated_r {R[r].min():.2f}-{R[r].max():.2f}, "
                  f"clear {d[r].min():.3f}-{d[r].max():.3f}, Rcurv4 {np.nanmin(kR[r]):.1f}, "
                  f"pct-of-branch {100*S[r[0]]/S[-1]:.0f}%")
    mi = int(np.argmin(d))
    print(f"  MIN station: s={S[mi]:.1f} ({100*S[mi]/S[-1]:.0f}% of branch, "
          f"{'GRAFT >137mm' if S[mi]>137 else 'SHARED CARRIER <137mm'}), "
          f"stated_r {R[mi]:.3f}, clear {d[mi]:.3f}, ratio {d[mi]/R[mi]:.2f}, "
          f"Rcurv4 {kR[mi]:.1f} (branch pctile "
          f"{100*(kfin<kR[mi]).mean():.0f}), axial {axial[mi]:.2f}")
    return dict(name=name, S=S, R=R, d=d, kR=kR, axial=axial, P=P, C=C, mesh=mesh,
                inside=inside, others=others, rc_tree=rc_tree)


res = {}
for n in CASES:
    try:
        res[n] = analyse(n)
    except Exception as e:
        import traceback; traceback.print_exc()
