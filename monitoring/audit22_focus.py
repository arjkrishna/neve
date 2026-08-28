"""K. Section-by-section diagnostic at the tight spots: in-plane inside-mask, region area,
   whether the flood region touches the window boundary (=> bore is truncated/contaminated)."""
import sys, os
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv
from scipy import ndimage
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
H,STEP=3.0,0.05
g=np.arange(-H,H+1e-9,STEP); NG=len(g); A,B=np.meshgrid(g,g,indexing="ij"); c0=NG//2
def load(name):
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0=os.path.join(ROOT,name)
    return (pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0,"Centrelines_comb")))
def sdist(m,P): return np.asarray(pv.PolyData(np.asarray(P,float)).compute_implicit_distance(m)["implicit_distance"])
def frame(T):
    a=np.array([0.,0.,1.]) if abs(T[2])<0.9 else np.array([1.,0.,0.])
    u=np.cross(T,a); u/=np.linalg.norm(u); v=np.cross(T,u); return u,v
JOBS=[("topcow_mr_024",188,203),("HOST",216,235),("topcow_mr_027",216,225),
      ("topcow_mr_025",148,162),("topcow_mr_011",143,152)]
for name,i0,i1 in JOBS:
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); R=np.asarray(rc.radii,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    T=np.gradient(C,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
    print("\n"+"="*126); print(f"{name}  n={len(C)} L={S[-1]:.1f}   stations {i0}..{min(i1,len(C))-1}"); print("="*126)
    print(f"{'i':>4} {'s_mm':>7} {'ctrClr':>7} {'stated_r':>8} {'bore':>6} {'areaMM2':>8} {'equivR':>7} {'touchWall':>9} {'nregions':>8} {'off':>6}")
    for i in range(i0,min(i1,len(C))):
        u,v=frame(T[i])
        P=C[i]+A.ravel()[:,None]*u+B.ravel()[:,None]*v
        d=sdist(m,P).reshape(NG,NG)
        if np.median(d)>0: d=-d
        ins=d<0; lab,nl=ndimage.label(ins)
        cl=lab[c0,c0]
        if cl==0:
            rr=np.where(ins,np.hypot(A,B),1e9); k=np.unravel_index(np.argmin(rr),rr.shape); cl=lab[k]
        reg=(lab==cl); area=reg.sum()*STEP*STEP
        touch=bool(reg[0,:].any() or reg[-1,:].any() or reg[:,0].any() or reg[:,-1].any())
        dm=np.where(reg,d,0.0); k=np.unravel_index(np.argmin(dm),dm.shape)
        print(f"{i:4d} {S[i]:7.1f} {abs(d[c0,c0]):7.3f} {R[i]:8.3f} {-d[k]:6.3f} {area:8.2f} "
              f"{np.sqrt(area/np.pi):7.3f} {str(touch):>9} {nl:8d} {np.hypot(A[k],B[k]):6.2f}")
