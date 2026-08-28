import json, sys
import numpy as np
RCCA = "Centerline curve - RCCA.mrk"
EXCL = ["topcow_mr_013","topcow_mr_014","topcow_mr_015"]
def cum(c): return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))

def pt_to_polyline(pts, poly):
    a=poly[:-1]; b=poly[1:]; ab=b-a; L2=(ab*ab).sum(1)
    out=np.empty(len(pts))
    for i,p in enumerate(pts):
        t=np.clip(((p-a)*ab).sum(1)/np.maximum(L2,1e-12),0,1)
        proj=a+t[:,None]*ab
        out[i]=np.min(np.linalg.norm(proj-p,axis=1))
    return out

from eve_bench.dualdevicenav import DualDeviceNav
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
import vtk
host=DualDeviceNav(); hvt=host.vessel_tree
hrc=np.asarray(hvt[RCCA].coordinates,dtype=np.float64); hs=cum(hrc)
iv=DualDeviceNavTopBrain(anatomy_dir="/opt/eve_training/results_topbrain/anatomies",
                         seed=42,episodes_between_change=1,exclude=EXCL)
vt=iv.vessel_tree; names=list(vt.anatomy_names)

GRID=np.array([40,60,80,100,110,120,125,130,135,140,150,170,200.])
def clr(mesh,pts):
    r=vtk.vtkOBJReader(); r.SetFileName(mesh); r.Update()
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(r.GetOutput())
    return np.array([imp.EvaluateFunction(*map(float,p)) for p in pts])

dev={}; clg={}; rad={}
for i,nm in enumerate(names):
    vt._select(i)
    rc=np.asarray(vt[RCCA].coordinates,dtype=np.float64); s=cum(rc)
    idx=[int(np.searchsorted(s,g)) for g in GRID if g<=s[-1]]
    d=pt_to_polyline(rc,hrc)                      # geometric deviation vs host curve
    dev[nm]=(s,d)
    gi=np.clip(np.searchsorted(s,GRID),0,len(s)-1)
    clg[nm]=clr(vt.mesh_path, rc[gi])
    rad[nm]=np.asarray(vt[RCCA].radii,dtype=float)[gi]
    j5=int(np.argmax(d>0.5)) if np.any(d>0.5) else len(s)-1
    j1=int(np.argmax(d>1.0)) if np.any(d>1.0) else len(s)-1
    j2=int(np.argmax(d>2.0)) if np.any(d>2.0) else len(s)-1
    print("%-14s dev>0.5 at s=%.1f  >1.0 at s=%.1f  >2.0 at s=%.1f | dev@40=%.2f @100=%.2f "
          "@125=%.2f @130=%.2f | max dev in s<130 = %.2f" % (
        nm, s[j5], s[j1], s[j2], d[int(np.searchsorted(s,40))],
        d[int(np.searchsorted(s,100))], d[int(np.searchsorted(s,125))],
        d[int(np.searchsorted(s,130))], d[s<130].max()))
    sys.stdout.flush()

hg=np.clip(np.searchsorted(hs,GRID),0,len(hs)-1)
hclr=clr(hvt.mesh_path, hrc[hg]); hrad=np.asarray(hvt[RCCA].radii,dtype=float)[hg]
print("\n s(mm) | host_clr host_r | cohort clr mean/std/min/max | cohort r mean/std/min/max")
for k,g in enumerate(GRID):
    cc=np.array([clg[n][k] for n in names]); rr=np.array([rad[n][k] for n in names])
    print("%6.0f | %8.2f %6.2f | %6.2f %5.2f %6.2f %6.2f | %6.2f %5.2f %6.2f %6.2f" % (
        g,hclr[k],hrad[k],cc.mean(),cc.std(),cc.min(),cc.max(),
        rr.mean(),rr.std(),rr.min(),rr.max()))

# mutual spread among cohort on the shared 1mm parameterisation
allrc={}
for i,nm in enumerate(names):
    vt._select(i); allrc[nm]=np.asarray(vt[RCCA].coordinates,dtype=np.float64)
m=min(len(v) for v in allrc.values())
st=np.stack([allrc[n][:m] for n in names])
spread=np.linalg.norm(st-st.mean(0),axis=2).max(0)
s0=cum(allrc[names[0]])[:m]
print("\ncohort mutual max-deviation-from-mean by arclength:")
for g in GRID:
    if g<s0[-1]:
        j=int(np.searchsorted(s0,g)); print("  s=%5.0f  spread=%.3f mm"%(g,spread[j]))
json.dump({n:[dev[n][0].tolist(),dev[n][1].tolist()] for n in names}, open("/scratch/dev.json","w"))
print("done")
