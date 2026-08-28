"""I(fixed). MAX PASSABLE SPHERE per RCCA cross-section, constrained to the in-plane lumen
region CONNECTED to the centerline point (4-connected flood fill on the inside mask), so the
maximiser cannot jump into a neighbouring vessel. Exact signed distance throughout."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv
from scipy import ndimage
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
EXCL=["013","014","015"]
NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN+EXCL]
WIRE,CATH=0.18,0.35
H,STEP=3.0,0.05
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
g=np.arange(-H,H+1e-9,STEP); NG=len(g); A,B=np.meshgrid(g,g,indexing="ij")
c0=NG//2   # index of offset 0
print("="*168)
print("I. MAX PASSABLE SPHERE RADIUS per RCCA cross-section (in-plane, flood-filled from the centerline)")
print("   grid +-3.0 mm @ 0.05 mm; bore<0.18 guidewire cannot pass, <0.35 catheter cannot pass")
print("="*168)
print(f"{'anatomy':>15} {'grp':>4} {'bore_min':>8} {'s@min':>6} {'p05':>6} {'med':>6} {'n<0.18':>6} "
      f"{'n<0.35':>6} {'n<0.60':>6} {'n<1.00':>6} | {'trim2 min':>9} {'s@':>6} | {'off@min':>7} {'ctrClr@min':>10} {'bore/ctr med':>12}")
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
        P=C[i]+A.ravel()[:,None]*u+B.ravel()[:,None]*v
        d=sdist(m,P).reshape(NG,NG)
        if np.median(d)>0: d=-d          # inside negative
        ins=d<0
        lab,_=ndimage.label(ins)
        cl=lab[c0,c0]
        if cl==0:                        # centerline itself outside -> take nearest inside blob
            if lab.max()==0: bore[i]=0.0; ctr[i]=abs(d[c0,c0]); off[i]=0.0; continue
            rr=np.hypot(A,B); rr=np.where(ins,rr,1e9); k=np.unravel_index(np.argmin(rr),rr.shape); cl=lab[k]
        reg=(lab==cl)
        dm=np.where(reg,d,0.0); k=np.unravel_index(np.argmin(dm),dm.shape)
        # local refine, still inside reg by construction (interior maximum)
        gg=np.arange(-0.05,0.0501,0.005); A2,B2=np.meshgrid(gg,gg,indexing="ij")
        P2=C[i]+(A[k]+A2.ravel())[:,None]*u+(B[k]+B2.ravel())[:,None]*v
        d2=sdist(m,P2);  d2=-d2 if np.median(d2)>0 else d2
        k2=int(np.argmin(d2))
        bore[i]=max(0.0,-float(d2[k2])); off[i]=float(np.hypot(A[k],B[k])); ctr[i]=abs(float(d[c0,c0]))
    n=len(C); mi=int(np.argmin(bore)); nt=np.arange(n)<n-2
    bt=np.where(nt,bore,1e9); mit=int(np.argmin(bt))
    print(f"{name:>15} {grp:>4} {bore.min():8.3f} {S[mi]:6.1f} {np.percentile(bore,5):6.3f} {np.median(bore):6.3f} "
          f"{int((bore<WIRE).sum()):6d} {int((bore<CATH).sum()):6d} {int((bore<0.6).sum()):6d} {int((bore<1.0).sum()):6d} | "
          f"{bore[mit]:9.3f} {S[mit]:6.1f} | {off[mi]:7.3f} {ctr[mi]:10.3f} {np.median(bore/np.maximum(ctr,1e-6)):12.2f}")
    res[name]=dict(bore_min=float(bore.min()),s_min=float(S[mi]),med=float(np.median(bore)),
                   n_wire=int((bore<WIRE).sum()),n_cath=int((bore<CATH).sum()),
                   n06=int((bore<0.6).sum()),n10=int((bore<1.0).sum()),
                   trim_min=float(bore[mit]),s_trim=float(S[mit]),
                   bore=[round(float(x),3) for x in bore],s=[round(float(x),1) for x in S],
                   ctr=[round(float(x),3) for x in ctr])
    sys.stdout.flush()
json.dump(res,open("/opt/eve_training/results_topbrain/_audit_bore2.json","w"))
print("\n"+"="*168); print("J. TIGHTEST 6 SECTIONS PER ANATOMY (s_mm : bore / centerline-clearance)"); print("="*168)
for k,v in res.items():
    b=np.array(v["bore"]); s=np.array(v["s"]); c=np.array(v["ctr"])
    o=np.argsort(b)[:6]
    print(f"{k:>15}  " + "  ".join(f"{s[j]:.0f}:{b[j]:.2f}/{c[j]:.2f}" for j in sorted(o,key=lambda j:s[j])))
