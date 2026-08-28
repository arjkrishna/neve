"""E. FUSION TEST: between RCCA and its nearest neighbour branch, is there still WALL?
   Sample the straight segment joining the two centerlines at their closest approach.
   Properly separated -> a contiguous OUTSIDE stretch in the middle.
   Fused             -> every sample INSIDE (one merged cavity).
F. tail profiles for HOST / 024 / 027. G. self-intersection test non-vacuity."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN]
def load(name):
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0=os.path.join(ROOT,name)
    return (pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0,"Centrelines_comb")))
def enclosed(mesh,pts):
    ps=pv.PolyData(np.asarray(pts,float))
    sel=vtk.vtkSelectEnclosedPoints(); sel.SetInputData(ps); sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6); sel.CheckSurfaceOff(); sel.Update()
    o=sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))])>0.5
print("="*150)
print("E. WALL-PRESENT TEST at the tightest RCCA-vs-branch approaches (all pairs with centreline gap < 3.5 mm, RCCA s>=135)")
print("   pathfrac_out = fraction of the 200-sample connecting segment lying OUTSIDE the surface (0.00 => LUMENS FUSED)")
print("="*150)
print(f"{'anatomy':>15} {'neighbour':>22} {'gap':>7} {'s@':>6} {'ctr_dist':>8} {'rA':>5} {'rB':>5} {'frac_out':>8} {'out_span_mm':>11} {'verdict':>10}")
bad=[]
for name in NAMES:
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); Rr=np.asarray(rc.radii,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    sel=S>=135.0; Cs,Rs,Ss=C[sel],Rr[sel],S[sel]
    for b in brs:
        nm=str(b.name)
        if "RCCA" in nm.upper(): continue
        B=np.asarray(b.coordinates,float); Rb=np.asarray(b.radii,float)
        D=np.linalg.norm(Cs[:,None,:]-B[None,:,:],axis=2); G=D-Rs[:,None]-Rb[None,:]
        i,j=np.unravel_index(np.argmin(G),G.shape); g=float(G.min())
        if g>=3.5: continue
        P,Q=Cs[i],B[j]; t=np.linspace(0,1,200); seg=P[None]+t[:,None]*(Q-P)[None]
        e=enclosed(m,seg); fo=float((~e).mean()); span=fo*float(np.linalg.norm(Q-P))
        v="FUSED" if fo<0.01 else ("thin" if span<0.30 else "ok")
        if v!="ok": bad.append((name,nm,g,fo,span))
        print(f"{name:>15} {nm.replace('Centerline curve ',''):>22} {g:7.3f} {Ss[i]:6.1f} "
              f"{float(D[i,j]):8.3f} {Rs[i]:5.2f} {Rb[j]:5.2f} {fo:8.3f} {span:11.3f} {v:>10}")
print("\n  flagged:",bad if bad else "none")

print("\n"+"="*150); print("F. DISTAL PROFILE, last 20 stations (d_eff; * = blocked <0.18, o = OUTSIDE)"); print("="*150)
for name in ["HOST","topcow_mr_024","topcow_mr_027","topcow_mr_004","topcow_mr_008","topcow_mr_025","topcow_mr_023","topcow_mr_006"]:
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(m)
    sd=np.array([imp.EvaluateFunction(p) for p in C]); ins=enclosed(m,C)
    de=np.where(ins,np.abs(sd),0.0)
    print(f"{name:>15} (n={len(C)}, L={S[-1]:.1f})")
    print("   " + " ".join(f"{S[k]:.0f}:{de[k]:.2f}{'o' if not ins[k] else ('*' if de[k]<0.18 else '')}" for k in range(len(C)-20,len(C))))

print("\n"+"="*150); print("G. self-intersection candidate-pair counts (non-vacuity check)"); print("="*150)
from scipy.spatial import cKDTree
for name in ["HOST","topcow_mr_001","topcow_mr_024","topcow_mr_026"]:
    m,_=load(name); v=m.points; f=m.faces.reshape(-1,4)[:,1:]
    cen=v[f].mean(1); rad=np.linalg.norm(v[f]-cen[:,None],axis=2).max(1)
    pr=cKDTree(cen).query_pairs(2*float(rad.max()),output_type='ndarray')
    ok=(np.linalg.norm(cen[pr[:,0]]-cen[pr[:,1]],axis=1)<=rad[pr[:,0]]+rad[pr[:,1]])
    pr=pr[ok]; share=np.array([len(set(f[a]).intersection(f[b]))>0 for a,b in pr])
    print(f"  {name:>15}  ncells={m.n_cells}  bbox-overlap pairs={len(pr)}  non-adjacent tested={int((~share).sum())}  max_tri_circumradius={rad.max():.2f}mm")
