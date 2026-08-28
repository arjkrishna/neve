"""TASK2 geometry: dense exact signed-clearance profile per anatomy on the RCCA branch.
READ-ONLY. Dumps JSON to stdout after @@@JSON@@@."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT="/opt/eve_training/results_topbrain/anatomies"
THR={"w018":0.18,"s030":0.30,"c035":0.35}
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
            r.append(dict(s0=float(g[i]),s1=float(g[j]),length=float(g[j]-g[i]),
                          min_d=float(d_eff[i:j+1].min()),
                          s_at_min=float(g[i+int(np.argmin(d_eff[i:j+1]))]))); i=j+1
        else: i+=1
    return r

OUT={}
for n in RETAIN:
    name=f"topcow_mr_{n}"; d0=os.path.join(ROOT,name)
    mesh=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); Rr=np.asarray(rc.radii,float)
    S=arclen(C); L=float(S[-1])
    P,g=densify(C)
    sd=signed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    enc=enclosed(mesh,P)
    d=np.abs(sd); d_eff=np.where(enc,d,0.0)
    rec=dict(short=f"topcowmr{n}",L=L,nstat=int(len(C)),
             g=[round(float(x),3) for x in g], d=[round(float(x),4) for x in d_eff],
             r_decl=[round(float(x),3) for x in np.interp(g,S,Rr)],
             nout=int((~enc).sum()))
    for k,t in THR.items():
        rr=runs(d_eff<t,g,d_eff)
        rec[k]=rr
    OUT[f"topcowmr{n}"]=rec
    blk=[r for r in rec["s030"] if r["s1"]<L-8.0]
    print(f"[geom] {name} L={L:.1f} nout={rec['nout']} min_d={min(rec['d']):.3f} "
          f"n_s030_runs={len(rec['s030'])} n_mid={len(blk)} "
          f"mid_s0={[round(r['s0'],1) for r in blk]}", flush=True)
print("@@@JSON@@@")
print(json.dumps(OUT))
