"""I. PASSABLE BORE. At each RCCA station, cut a plane through the station normal to the
tangent and find the in-plane point maximising the EXACT signed distance to the surface.
That is the largest sphere that fits at that cross-section = the largest device that can
be threaded through it, regardless of whether the centerline itself is centred.
Two-stage grid + refine, all exact (pyvista compute_implicit_distance -> vtkImplicitPolyDataDistance)."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
EXCL=["013","014","015"]
NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN+EXCL]
WIRE,CATH=0.18,0.35
def load(name):
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0=os.path.join(ROOT,name)
    return (pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0,"Centrelines_comb")))
def sdist(mesh,P):
    return np.asarray(pv.PolyData(np.asarray(P,float)).compute_implicit_distance(mesh)["implicit_distance"])
def frame(T):
    a=np.array([0.,0.,1.]) if abs(T[2])<0.9 else np.array([1.,0.,0.])
    u=np.cross(T,a); u/=np.linalg.norm(u); v=np.cross(T,u); return u,v
print("="*160)
print("I. MAX PASSABLE SPHERE RADIUS per RCCA cross-section (exact; in-plane maximiser of signed distance)")
print("   bore < 0.18 => guidewire cannot pass at all;  < 0.35 => catheter cannot pass")
print("="*160)
print(f"{'anatomy':>15} {'grp':>4} {'bore_min':>8} {'s@min':>6} {'bore_p05':>8} {'bore_med':>8} "
      f"{'n<0.18':>6} {'n<0.35':>6} {'n<0.60':>6} {'min excl last2':>14} {'s@':>6} {'offset@min':>10} {'ctr_d@min':>9}")
res={}
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    T=np.gradient(C,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
    bore=np.zeros(len(C)); off=np.zeros(len(C)); ctr=np.zeros(len(C))
    for i in range(len(C)):
        u,v=frame(T[i])
        # stage 1: +-3.0 mm @ 0.15
        g=np.arange(-3.0,3.001,0.15); A,B=np.meshgrid(g,g,indexing="ij")
        P=C[i]+A.ravel()[:,None]*u+B.ravel()[:,None]*v
        d=sdist(m,P)
        if np.median(d)>0: d=-d
        k=int(np.argmin(d)); a0,b0=A.ravel()[k],B.ravel()[k]
        # stage 2: refine +-0.15 @ 0.01
        g2=np.arange(-0.16,0.161,0.01); A2,B2=np.meshgrid(g2,g2,indexing="ij")
        P2=C[i]+(a0+A2.ravel())[:,None]*u+(b0+B2.ravel())[:,None]*v
        d2=sdist(m,P2)
        if np.median(d2)>0: d2=-d2
        k2=int(np.argmin(d2))
        bore[i]=max(0.0,-float(d2[k2]))
        off[i]=float(np.hypot(a0+A2.ravel()[k2],b0+B2.ravel()[k2]))
        dc=sdist(m,C[i][None]); ctr[i]=abs(float(dc[0]))
    n=len(C); mi=int(np.argmin(bore)); nt=np.arange(n)<n-2
    mit=int(np.argmin(np.where(nt,bore,1e9)))
    print(f"{name:>15} {grp:>4} {bore.min():8.3f} {S[mi]:6.1f} {np.percentile(bore,5):8.3f} {np.median(bore):8.3f} "
          f"{int((bore<WIRE).sum()):6d} {int((bore<CATH).sum()):6d} {int((bore<0.60).sum()):6d} "
          f"{bore[mit]:14.3f} {S[mit]:6.1f} {off[mi]:10.3f} {ctr[mi]:9.3f}")
    res[name]=dict(bore_min=float(bore.min()),s_min=float(S[mi]),med=float(np.median(bore)),
                   p05=float(np.percentile(bore,5)),
                   n_wire=int((bore<WIRE).sum()),n_cath=int((bore<CATH).sum()),
                   bore_min_trim=float(bore[mit]),s_min_trim=float(S[mit]),
                   bore=[round(float(x),3) for x in bore],s=[round(float(x),1) for x in S])
    sys.stdout.flush()
json.dump(res,open("/opt/eve_training/results_topbrain/_audit_bore.json","w"))
print("\nwrote _audit_bore.json")
