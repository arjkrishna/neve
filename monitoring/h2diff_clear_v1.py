"""Exact signed clearance along the GRAFT region (s_RCCA>=133.6) for 22 + HOST."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
SEAM=133.6; WIRE=0.18; CATH=0.35; SOFA=0.30
RET=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
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
out={}
print("%-16s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s"%("anat","L","gL","clr_min","clr_p05","clr_p25","clr_med","n<.35","n<.30","n<.18","nOUT"))
for name in ["HOST"]+["topcow_mr_"+n for n in RET]:
    mesh,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); R=np.asarray(rc.radii,float); S=arclen(C); L=float(S[-1])
    g=np.arange(SEAM,L,0.25); g=np.append(g,L)
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1)
    Rg=np.interp(g,S,R)
    sd=signed(mesh,P)
    ins=enclosed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    d=np.abs(sd); deff=np.where(ins,d,0.0)
    o=dict(L=L,gL=L-SEAM,nst=len(g),
        clr_min=float(deff.min()),clr_p05=float(np.percentile(deff,5)),
        clr_p25=float(np.percentile(deff,25)),clr_med=float(np.median(deff)),
        clr_term=float(deff[-1]),
        n_lt_cath=int((deff<CATH).sum()),n_lt_sofa=int((deff<SOFA).sum()),
        n_lt_wire=int((deff<WIRE).sum()),n_out=int((~ins).sum()),
        f_lt_cath=float((deff<CATH).mean()),
        rdec_min=float(Rg.min()),rdec_med=float(np.median(Rg)),
        clr_minus_r_med=float(np.median(deff-Rg)))
    # deepest non-terminal blockage (exclude last 2mm)
    nt=g<L-2.0
    o["clr_min_nonterm"]=float(deff[nt].min()) if nt.sum() else float("nan")
    o["s_at_min"]=float(g[int(np.argmin(deff))])
    out[name]=o
    print("%-16s %7.1f %7.1f %7.3f %7.3f %7.3f %7.3f %6d %6d %6d %6d"%(name,L,o["gL"],o["clr_min"],o["clr_p05"],o["clr_p25"],o["clr_med"],o["n_lt_cath"],o["n_lt_sofa"],o["n_lt_wire"],o["n_out"]))
print("JSONSTART"); print(json.dumps(out)); print("JSONEND")
