"""Locate the deep mid-vessel RCCA blockages relative to the RCCA/RECA bifurcation."""
import os, sys
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches
CASES = ["case_w_047_left__topcow_mr_018_L","case_w_052_left__topcow_mr_027_L",
         "case_m_022_left__topcow_mr_020","case_w_007_right__topcow_mr_027_L",
         "case_w_047_left__topcow_mr_023","case_w_046_left__topcow_mr_020_L",
         "case_k_011_left__topcow_mr_023_L","case_w_003_left__topcow_mr_007_L",
         "case_w_047_left__topcow_mr_027","case_m_024_left__topcow_mr_023_L"]
for nm in CASES:
    d = "/opt/eve_training/carotid/anatomies/" + nm
    brs = {str(b.name): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    kr = [k for k in brs if "RCCA" in k.upper()][0]
    ke = [k for k in brs if "RECA" in k.upper()][0]
    c = np.asarray(brs[kr].coordinates, float); e = np.asarray(brs[ke].coordinates, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    dd = np.linalg.norm(c - e[0], axis=1); i = int(dd.argmin())
    # departure point: last index where RECA still within 1mm of RCCA
    print("%-46s L=%.1f  RECA root nearest RCCA s=%.2f (gap %.2f mm)"
          % (nm, s[-1], s[i], dd[i]))
