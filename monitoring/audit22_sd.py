"""STAGE 1+2: exact signed distance on RCCA, raw stations + 0.25mm densified,
full set and terminal-trimmed. Cohort 25 (incl the 3 excluded, as reference) + HOST."""
import sys, os, glob, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

ROOT="/opt/eve_training/results_topbrain/anatomies"
WIRE,CATH=0.18,0.35
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
EXCL=["013","014","015"]

def arclen(c): return np.concatenate(([0.0],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))
def signed(mesh,pts):
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])
def enclosed(mesh,pts):
    ps=pv.PolyData(np.asarray(pts,float))
    sel=vtk.vtkSelectEnclosedPoints(); sel.SetInputData(ps); sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6); sel.CheckSurfaceOff(); sel.Update()
    o=sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))])>0.5
def load(name):
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    d0=os.path.join(ROOT,name)
    return (pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean(),
            load_branches(os.path.join(d0,"Centrelines_comb")))
def densify(C,step=0.25):
    S=arclen(C); g=np.arange(0.0,S[-1],step)
    g=np.append(g,S[-1])
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1)
    return P,g

NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN]+[f"topcow_mr_{n}" for n in EXCL]
out={}
hdr=f"{'anatomy':>15} {'grp':>4} {'nst':>4} {'len':>6} {'signOK%':>7} {'nOUT':>5} {'maxOut':>7} {'min_d':>7} {'nBLK':>5} {'nCAT':>5} | {'nOUT_t':>6} {'nBLK_t':>6} {'nCAT_t':>6} | {'BLKidx(non-term) s_mm/d':>28}"
print("="*190); print("A. RAW CENTERLINE STATIONS  (d_eff = |d| if inside else 0)"); print("="*190); print(hdr)
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    mesh,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); R=np.asarray(rc.radii,float); S=arclen(C)
    sd=signed(mesh,C); ins_sd=sd<0
    if ins_sd.mean()<0.5: ins_sd=~ins_sd; sd=-sd
    enc=enclosed(mesh,C)
    agree=100.0*(ins_sd==enc).mean()
    ins=enc              # authoritative
    d=np.abs(sd); d_eff=np.where(ins,d,0.0)
    outd=d[~ins]; maxout=float(outd.max()) if len(outd) else 0.0
    blk=d_eff<WIRE; cat=d_eff<CATH
    n=len(C); term=np.arange(n)>=n-2
    blk_t=blk&~term; cat_t=cat&~term; out_t=(~ins)&~term
    bi=np.where(blk_t)[0]
    print(f"{name:>15} {grp:>4} {n:4d} {S[-1]:6.1f} {agree:7.1f} {int((~ins).sum()):5d} {maxout:7.3f} "
          f"{d_eff.min():7.3f} {int(blk.sum()):5d} {int(cat.sum()):5d} | {int(out_t.sum()):6d} {int(blk_t.sum()):6d} "
          f"{int(cat_t.sum()):6d} | {str([f'{S[i]:.0f}/{d_eff[i]:.2f}' for i in bi])[:70]}")
    out[name]=dict(n=n,L=float(S[-1]),agree=agree,nout=int((~ins).sum()),maxout=maxout,
                   nblk=int(blk.sum()),ncat=int(cat.sum()),nout_t=int(out_t.sum()),
                   nblk_t=int(blk_t.sum()),ncat_t=int(cat_t.sum()),
                   blk_s=[float(S[i]) for i in np.where(blk)[0]],
                   out_s=[float(S[i]) for i in np.where(~ins)[0]],
                   cat_s=[float(S[i]) for i in np.where(cat)[0]],
                   mind=float(d_eff.min()), med=float(np.median(d_eff)))

print("\n"+"="*190); print("B. DENSIFIED 0.25 mm ALONG THE POLYLINE (catches mid-segment wall crossings station sampling misses)"); print("="*190)
print(f"{'anatomy':>15} {'grp':>4} {'npts':>5} {'nOUT':>5} {'maxOut':>7} {'outRUNS (s_mm span, max depth)':>44} {'nBLK':>5} {'BLK runs (s span / min d)':>46} {'min_d':>7} {'med':>6}")
dens={}
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    mesh,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float)
    P,g=densify(C)
    sd=signed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    ins=enclosed(mesh,P)
    d=np.abs(sd); d_eff=np.where(ins,d,0.0)
    blk=d_eff<WIRE
    def runs(mask):
        r=[];i=0
        while i<len(mask):
            if mask[i]:
                j=i
                while j+1<len(mask) and mask[j+1]: j+=1
                r.append((g[i],g[j],float(d[i:j+1].max()),float(d_eff[i:j+1].min()))); i=j+1
            else: i+=1
        return r
    ro=runs(~ins); rb=runs(blk)
    print(f"{name:>15} {grp:>4} {len(P):5d} {int((~ins).sum()):5d} {(d[~ins].max() if (~ins).any() else 0.0):7.3f} "
          f"{str([f'{a:.0f}-{b:.0f}:{c:.2f}' for a,b,c,_ in ro])[:42]:>44} {int(blk.sum()):5d} "
          f"{str([f'{a:.0f}-{b:.0f}:{dd:.3f}' for a,b,_,dd in rb])[:44]:>46} {d_eff.min():7.3f} {np.median(d_eff):6.3f}")
    dens[name]=dict(npts=len(P),nout=int((~ins).sum()),nblk=int(blk.sum()),
                    out_runs=[[float(a),float(b),float(c)] for a,b,c,_ in ro],
                    blk_runs=[[float(a),float(b),float(dd)] for a,b,_,dd in rb],
                    L=float(g[-1]))
json.dump({"stations":out,"dense":dens},open("/opt/eve_training/results_topbrain/_audit_sd.json","w"),indent=1)
print("\nwrote _audit_sd.json")
