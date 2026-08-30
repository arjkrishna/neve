"""TASK3 geometry: per-anatomy dense RCCA profile (arclen, declared radius,
exact signed clearance, Menger curvature at 2 scales, cumulative turning).
READ-ONLY on data. Writes /opt/mon/arj_t3_geom.json"""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT="/opt/eve_training/results_topbrain/anatomies"
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
def resamp(C,S,step):
    g=np.arange(0.0,S[-1],step); g=np.append(g,S[-1])
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1)
    return P,g
def menger(p,M):
    a=p[:-2*M]; b=p[M:len(p)-M]; c=p[2*M:]
    ab=np.linalg.norm(b-a,axis=1); bc=np.linalg.norm(c-b,axis=1); ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1); den=ab*bc*ca
    k=np.where(den>1e-12,4.0*ar/np.maximum(den,1e-12),0.0)
    return np.concatenate([np.full(M,k[0]),k,np.full(M,k[-1])])
def turning(p):
    d=np.diff(p,axis=0); n=np.linalg.norm(d,axis=1,keepdims=True); d=d/np.maximum(n,1e-9)
    ang=np.degrees(np.arccos(np.clip((d[:-1]*d[1:]).sum(1),-1,1)))
    return np.concatenate([[0.0],ang,[0.0]])

STEP=0.25
OUT={}
for n in RETAIN:
    name="topcow_mr_%s"%n; d0=os.path.join(ROOT,name)
    mesh=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float); Rr=np.asarray(rc.radii,float); S=arclen(C); L=float(S[-1])
    P,g=resamp(C,S,STEP)
    sd=signed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    enc=enclosed(mesh,P); d_eff=np.where(enc,np.abs(sd),0.0)
    r_decl=np.interp(g,S,Rr)
    k1=menger(P,4)    # +/-1.0 mm arm
    k3=menger(P,12)   # +/-3.0 mm arm
    # turning on 1.0 mm resample, mapped back to g
    P1,g1=resamp(C,S,1.0); t1=turning(P1); cum1=np.cumsum(t1)
    cumturn=np.interp(g,g1,cum1)
    OUT["topcowmr%s"%n]=dict(L=L,step=STEP,
        g=[round(float(x),3) for x in g],
        d=[round(float(x),4) for x in d_eff],
        r=[round(float(x),4) for x in r_decl],
        k1=[round(float(x),6) for x in k1],
        k3=[round(float(x),6) for x in k3],
        cum=[round(float(x),3) for x in cumturn],
        nout=int((~enc).sum()))
    print("[g] %s L=%.2f mind=%.3f maxk1=%.4f maxk3=%.4f turn=%.0f"%(
        name,L,d_eff.min(),k1.max(),k3.max(),cumturn[-1]),flush=True)
json.dump(OUT,open("/opt/mon/arj_t3_geom.json","w"))
print("DONE")
