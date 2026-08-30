"""Dump full 0.25mm graft-region clearance + declared-radius profiles, 22 + HOST."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"; SEAM=133.6
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
out={}
for name in ["HOST"]+["topcow_mr_"+n for n in RET]:
    if name=="HOST":
        vt=DualDeviceNav().vessel_tree
        mesh=pv.read(vt.mesh_path).triangulate().clean(); brs=list(vt.branches)
    else:
        d0=os.path.join(ROOT,name)
        mesh=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
        brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); R=np.asarray(rc.radii,float); S=arclen(C); L=float(S[-1])
    g=np.arange(SEAM,L,0.25); g=np.append(g,L)
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1); Rg=np.interp(g,S,R)
    sd=signed(mesh,P); ins=enclosed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    deff=np.where(ins,np.abs(sd),0.0)
    out[name]={"L":L,"s":[round(float(x),3) for x in g],
               "clr":[round(float(x),4) for x in deff],"rdec":[round(float(x),4) for x in Rg],
               "xyz":[[round(float(c),3) for c in p] for p in P]}
print("JSONSTART"); print(json.dumps(out)); print("JSONEND")
