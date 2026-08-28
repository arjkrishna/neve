"""TASK1: exact signed-clearance reachability map on RCCA for the 22-anatomy cohort,
then re-score the 220-episode eval. READ-ONLY. Prints everything to stdout."""
import sys, os, json, math
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT="/opt/eve_training/results_topbrain/anatomies"
OFF=33.31                     # path_len = s_RCCA + OFF
THR={"wire_0.18":0.18,"sofa_0.30":0.30,"cath_0.35":0.35,"sofa+wire_0.48":0.48,"sofa+cath_0.65":0.65,"strict_0.90":0.90}
TERM_MM=8.0                   # a blockage whose run lies within the last 8 mm is "terminal"
MIN_ARC=40.0                  # sampler min_arclength_from_start
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016",
        "017","018","020","021","022","023","024","025","026","027"]

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
def densify(C,step=0.25):
    S=arclen(C); g=np.arange(0.0,S[-1],step); g=np.append(g,S[-1])
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1)
    return P,g
def runs(mask,g,d_eff):
    r=[];i=0
    while i<len(mask):
        if mask[i]:
            j=i
            while j+1<len(mask) and mask[j+1]: j+=1
            r.append(dict(s0=float(g[i]),s1=float(g[j]),
                          length=float(g[j]-g[i]),
                          min_d=float(d_eff[i:j+1].min()))); i=j+1
        else: i+=1
    return r

GEO={}
for n in RETAIN:
    name=f"topcow_mr_{n}"; d0=os.path.join(ROOT,name)
    mesh=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); S=arclen(C); L=float(S[-1])
    P,g=densify(C)
    sd=signed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    ins_sd=sd<0
    enc=enclosed(mesh,P)
    agree=100.0*(ins_sd==enc).mean()
    ins=enc
    d=np.abs(sd); d_eff=np.where(ins,d,0.0)
    rec=dict(name=name,short=f"mr_{n}",L=L,nstat=len(C),npts=len(P),agree=agree,
             min_d=float(d_eff.min()),med_d=float(np.median(d_eff)),
             nout=int((~ins).sum()))
    for k,t in THR.items():
        rr=runs(d_eff<t,g,d_eff)
        for r in rr: r["terminal"]= (r["s1"] >= L-TERM_MM)
        mid=[r for r in rr if not r["terminal"]]
        rec[k]=dict(runs=rr,n_runs=len(rr),n_mid=len(mid),
                    s_block=(min(r["s0"] for r in mid) if mid else None),
                    s_block_any=(min(r["s0"] for r in rr) if rr else None))
    # admissible target pool = raw RCCA stations, arclength>=40mm, minus points inside other branches
    others=[b for b in brs if b is not rc]
    pts=C[S>=MIN_ARC]; sp=S[S>=MIN_ARC]
    inex=np.zeros(len(pts),bool)
    for b in others:
        try:
            if hasattr(b,"in_branch") and getattr(b,"radii",None) is not None:
                inex |= np.asarray(b.in_branch(pts),bool)
        except Exception: pass
    pool_s=sp[~inex]
    rec["pool_n"]=int(len(pool_s)); rec["pool_s_min"]=float(pool_s.min()); rec["pool_s_max"]=float(pool_s.max())
    rec["pool_s"]=[float(x) for x in pool_s]
    GEO[name]=rec
    print(f"[geom] {name} L={L:.1f} nstat={rec['nstat']} pool={rec['pool_n']} "
          f"agree={agree:.1f}% min_d={rec['min_d']:.3f} nout={rec['nout']}", flush=True)

print("\n@@@JSON@@@")
print(json.dumps(GEO))
