"""L. Re-score exact clearance at the SOFA gate. sofabeamadapter.py:356-357 sets
LocalMinDistance(contactDistance=0.3, alarmDistance=0.5); the beam collision models are
Line/Point with proximity=0.0, so the guidewire BEAM AXIS must hold >=0.30 mm from the wall
(0.18 mm geometric radius never enters the collision test). Densified 0.25 mm."""
import sys, os, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches, DualDeviceNav
ROOT="/opt/eve_training/results_topbrain/anatomies"
RETAIN=["001","002","003","004","005","006","007","008","010","011","012","016","017","018","020","021","022","023","024","025","026","027"]
EXCL=["013","014","015"]
NAMES=["HOST"]+[f"topcow_mr_{n}" for n in RETAIN+EXCL]
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
print("="*176)
print("L. CLEARANCE AT THE SOFA GATE (densified 0.25 mm along the RCCA polyline; d = exact signed distance, 0 if outside)")
print("   CONTACT 0.30 mm = LocalMinDistance.contactDistance : below this the wire axis is inside the contact constraint")
print("   ALARM   0.50 mm = LocalMinDistance.alarmDistance   : below this the narrow phase is generating contacts every step")
print("="*176)
print(f"{'anatomy':>15} {'grp':>4} {'L':>6} {'min_d':>6} {'med':>5} | {'mm<0.18':>7} {'mm<0.30':>7} {'mm<0.50':>7} "
      f"{'%<0.50':>6} | {'longest<0.30 run':>16} {'@s':>6} | {'mm<0.30 excl last 3mm':>21} | {'sub-0.30 runs (s span)':>44}")
res={}
for name in NAMES:
    grp="HOST" if name=="HOST" else ("keep" if name[-3:] in RETAIN else "EXCL")
    m,brs=load(name)
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    C=np.asarray(rc.coordinates,float)
    S=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(C,axis=0),axis=1))))
    g=np.append(np.arange(0.,S[-1],0.25),S[-1])
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1)
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(m)
    sd=np.array([imp.EvaluateFunction(p) for p in P])
    ins=enclosed(m,P); d=np.where(ins,np.abs(sd),0.0)
    def runs(mask):
        r=[];i=0
        while i<len(mask):
            if mask[i]:
                j=i
                while j+1<len(mask) and mask[j+1]: j+=1
                r.append((g[i],g[j],float(d[i:j+1].min()))); i=j+1
            else: i+=1
        return r
    m18=0.25*int((d<0.18).sum()); m30=0.25*int((d<0.30).sum()); m50=0.25*int((d<0.50).sum())
    r30=runs(d<0.30)
    ln=max([b-a for a,b,_ in r30],default=0.0); at=([a for a,b,_ in r30 if b-a==ln] or [float('nan')])[0]
    keep=g<g[-1]-3.0
    m30t=0.25*int(((d<0.30)&keep).sum())
    print(f"{name:>15} {grp:>4} {g[-1]:6.1f} {d.min():6.3f} {np.median(d):5.2f} | {m18:7.2f} {m30:7.2f} {m50:7.2f} "
          f"{100*m50/g[-1]:6.1f} | {ln:16.2f} {at:6.1f} | {m30t:21.2f} | "
          f"{str([f'{a:.0f}-{b:.0f}({c:.2f})' for a,b,c in r30])[:42]:>44}")
    res[name]=dict(mm18=m18,mm30=m30,mm50=m50,mm30_trim=m30t,longest=ln,at=float(at) if at==at else None,
                   runs30=[[round(a,1),round(b,1),round(c,3)] for a,b,c in r30],L=float(g[-1]),
                   mind=float(d.min()),med=float(np.median(d)))
    sys.stdout.flush()
json.dump(res,open("/opt/eve_training/results_topbrain/_audit_gate.json","w"),indent=1)
print("\n  ranking of the 22 by millimetres of RCCA below the 0.30 mm contact distance (excluding the last 3 mm):")
k=[(v["mm30_trim"],n) for n,v in res.items() if n!="HOST" and n[-3:] in RETAIN]
for val,n in sorted(k,reverse=True): print(f"    {n:>15} {val:8.2f} mm   (HOST control: {res['HOST']['mm30_trim']:.2f} mm)")
