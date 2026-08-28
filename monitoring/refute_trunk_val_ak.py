"""Validate the rebake instrument: re-run generate_mesh on a cohort anatomy's OWN branches
and compare to its shipped vessel_architecture_collision.obj. If they reproduce, the
HOST_REBAKE control is a faithful application of the same pipeline.
"""
import os
import sys

import numpy as np
import pyvista as pv
import vtk

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve.intervention.vesseltree.util.meshing import generate_mesh  # noqa: E402
from eve_bench.dualdevicenav import load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
os.makedirs("/tmp/out", exist_ok=True)


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


for name in ["topcow_mr_001", "topcow_mr_016"]:
    d0 = os.path.join(ROOT, name)
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    shipped = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    out = "/tmp/out/%s_rebake.obj" % name
    generate_mesh(brs, out, 0.99)
    mine = pv.read(out).triangulate().clean()
    b = next(x for x in brs if "RCCA" in str(x.name).upper())
    C = np.asarray(b.coordinates, float)
    S = arclen(C)
    m = S < 130
    a = signed(shipped, C[m])
    c = signed(mine, C[m])
    if (a < 0).mean() < 0.5:
        a = -a
    if (c < 0).mean() < 0.5:
        c = -c
    print("%-15s shipped pts=%d cells=%d | rebake pts=%d cells=%d | trunk clearance "
          "med shipped=%.4f rebake=%.4f  max|diff|=%.4f  med|diff|=%.4f"
          % (name, shipped.n_points, shipped.n_cells, mine.n_points, mine.n_cells,
             np.median(np.abs(a)), np.median(np.abs(c)),
             np.abs(np.abs(a) - np.abs(c)).max(), np.median(np.abs(np.abs(a) - np.abs(c)))),
          flush=True)
