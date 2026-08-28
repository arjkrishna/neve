"""M2. TRUE min distance from each interior shell to the RCCA centerline and to ALL
centerlines, and what lumen the shells sit in. Host has none (single watertight body)."""
import sys, os
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
from eve_bench.dualdevicenav import load_branches
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
print(f"{'anatomy':>15} {'#shells':>7} {'shellCells':>10} {'minD_to_RCCA':>12} {'s@':>6} {'minD_anyCL':>10} {'branch@min':>26} {'shell z-range':>18}")
for n in RETAIN:
    nm=f"topcow_mr_{n}"; d0=os.path.join(ROOT,nm)
    m=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper()); C=np.asarray(rc.coordinates,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    conn=m.connectivity(); cid=conn.cell_data["RegionId"]
    idx=np.where(cid!=0)[0]
    sh=conn.extract_cells(idx).extract_surface()
    P=np.asarray(sh.points,float)
    dR,iR=cKDTree(C).query(P); k=int(np.argmin(dR))
    best=(1e9,None)
    for b in brs:
        B=np.asarray(b.coordinates,float)
        dd=cKDTree(B).query(P)[0].min()
        if dd<best[0]: best=(float(dd),str(b.name))
    print(f"{nm:>15} {int(cid.max()):7d} {len(idx):10d} {dR.min():12.2f} {S[iR[k]]:6.1f} {best[0]:10.2f} "
          f"{best[1].replace('Centerline curve ',''):>26} {str(np.round([P[:,2].min(),P[:,2].max()],1)):>18}")
