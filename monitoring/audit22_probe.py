import sys, os, glob, importlib
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
for m in ["numpy","vtk","pyvista","trimesh","scipy","networkx","rtree"]:
    try:
        mm=importlib.import_module(m); print(f"OK  {m} {getattr(mm,'__version__','?')}")
    except Exception as e: print(f"NO  {m}: {e}")
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
import numpy as np
d="/opt/eve_training/results_topbrain/anatomies/topcow_mr_001/Centrelines_comb"
brs=load_branches(d)
print("\ncohort 001 branches:")
for b in brs:
    c=np.asarray(b.coordinates,float)
    s=float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())
    print(f"  {str(b.name):28s} n={len(c):4d} len={s:7.1f} r_med={np.median(np.asarray(b.radii,float)):6.3f}")
vt=DualDeviceNav().vessel_tree
print("\nHOST branches:", "mesh:",vt.mesh_path)
for b in vt.branches:
    c=np.asarray(b.coordinates,float)
    s=float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())
    print(f"  {str(b.name):28s} n={len(c):4d} len={s:7.1f} r_med={np.median(np.asarray(b.radii,float)):6.3f}")
