"""Calibrate the vtkImplicitPolyDataDistance sign, then audit whether every
pooled TARGET of a TopBrain anatomy lies inside the collision surface.

Read-only. The cohort meshes are non-watertight (2-5 open boundary edges), so
the sign convention is established empirically from points known to be outside
(far field) rather than assumed.
"""
import os
import sys

import numpy as np
import pyvista as pv
import vtk

sys.path.insert(0, "/opt/eve_training/training_scripts")

from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain  # noqa: E402

RCCA = "Centerline curve - RCCA.mrk"
DIR = "/opt/eve_training/results_topbrain/anatomies"
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["topcow_mr_020"]
trims = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]

for name in names:
    iv = DualDeviceNavTopBrain(anatomy_dir=DIR, seed=900000,
                               episodes_between_change=1, only=[name])
    vt = iv.vessel_tree
    mesh = pv.read(vt.mesh_path).triangulate().clean()
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)

    ctr = np.asarray(mesh.center, float)
    far = ctr + np.array([1e4, 1e4, 1e4])
    d_far = imp.EvaluateFunction(far)
    sign = 1.0 if d_far > 0 else -1.0     # make POSITIVE mean OUTSIDE
    print(f"\n=== {name} ===")
    print(f"mesh={vt.mesh_path} cells={mesh.n_cells} "
          f"open_edges={mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False).n_cells}")
    print(f"sign calibration: far-field point -> {d_far:.1f}; "
          f"treating {'positive' if sign > 0 else 'negative'} as OUTSIDE")

    rcca = next(b for b in vt.branches if str(b.name) == RCCA)
    c = np.asarray(rcca.coordinates, float)
    d_cl = sign * np.array([imp.EvaluateFunction(p) for p in c])
    print(f"RCCA centerline: {len(c)} stations, "
          f"inside={(d_cl < 0).sum()}/{len(c)}  "
          f"clearance(inside) min={-d_cl.max():.3f} "
          f"median={-np.median(d_cl):.3f} max={-d_cl.min():.3f} mm")
    out_idx = np.where(d_cl >= 0)[0]
    if len(out_idx):
        print(f"  stations OUTSIDE the surface: {out_idx.tolist()}")

    for n_trim in trims:
        tgt = iv.target
        tgt._trim_last_stations = n_trim
        tgt._branches_initialized = None
        if n_trim:
            # replicate the eval script's class-level patch
            from eve.intervention.target.centerlinerandom import CenterlineRandom
            if not getattr(CenterlineRandom, "_p", False):
                _o = CenterlineRandom._arclength_from_start_mask

                def _m(self, points, _o=_o):
                    m = _o(self, points)
                    n = int(getattr(self, "_trim_last_stations", 0) or 0)
                    if n <= 0 or len(m) == 0:
                        return m
                    m = np.asarray(m, bool).copy()
                    m[-min(n, len(m)):] = False
                    return m
                CenterlineRandom._arclength_from_start_mask = _m
                CenterlineRandom._p = True
        tgt.reset(0, 0)
        pool = np.asarray(tgt._branch_targets[RCCA], float)
        d = sign * np.array([imp.EvaluateFunction(p) for p in pool])
        n_out = int((d >= 0).sum())
        wire_r = 0.18
        n_tight = int((-d < wire_r).sum())
        print(f"trim={n_trim}: pool={len(pool)}/{len(c)} stations  "
              f"inside={len(pool)-n_out}/{len(pool)}  "
              f"clearance min={-d.max():.3f} p05={-np.percentile(d,95):.3f} "
              f"median={-np.median(d):.3f} max={-d.min():.3f} mm  "
              f"below wire radius {wire_r}: {n_tight}")
        if n_out:
            print(f"  *** targets OUTSIDE the surface: "
                  f"{np.where(d >= 0)[0].tolist()[:20]} ***")
