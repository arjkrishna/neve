"""M. The 2-3 small extra connected components in every cohort mesh: where are they, and
does SOFA see them as an obstacle inside or near the RCCA lumen? (MeshObjLoader loads the
whole file into the TriangleCollisionModel, so a stray shell is a phantom wall.)"""
import sys, os
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
from eve_bench.dualdevicenav import load_branches
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
print("="*150)
print("M. EXTRA CONNECTED COMPONENTS (component 0 = main body containing the RCCA)")
print("="*150)
print(f"{'anatomy':>15} {'comp':>4} {'ncells':>6} {'npts':>5} {'bboxdiag':>8} {'centroid':>28} {'min dist to RCCA centerline':>28} {'inside main body?':>17}")
for n in RETAIN:
    nm=f"topcow_mr_{n}"; d0=os.path.join(ROOT,nm)
    m=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper()); C=np.asarray(rc.coordinates,float)
    allC=np.vstack([np.asarray(b.coordinates,float) for b in brs])
    conn=m.connectivity(); cid=conn.cell_data["RegionId"]
    main=conn.extract_cells(np.where(cid==0)[0]).extract_surface().triangulate()
    for c in range(int(cid.max())+1):
        sub=conn.extract_cells(np.where(cid==c)[0]).extract_surface()
        P=np.asarray(sub.points,float); ctr=P.mean(0)
        bb=np.asarray(sub.bounds); diag=float(np.linalg.norm(bb[1::2]-bb[0::2]))
        dr=float(cKDTree(C).query(ctr)[0])
        ip=pv.PolyData(P).compute_implicit_distance(main)["implicit_distance"]
        frac_in=float((np.asarray(ip)<0).mean())
        tag = "-" if c==0 else (f"{100*frac_in:.0f}% pts inside")
        print(f"{nm if c==0 else '':>15} {c:4d} {sub.n_cells:6d} {sub.n_points:5d} {diag:8.1f} "
              f"{str(np.round(ctr,1)):>28} {dr:28.1f} {tag:>17}")
