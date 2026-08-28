import os, json, sys
import numpy as np
RCCA = "Centerline curve - RCCA.mrk"
EXCL = ["topcow_mr_013","topcow_mr_014","topcow_mr_015"]

def cum(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))

from eve_bench.dualdevicenav import DualDeviceNav
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
import vtk

host = DualDeviceNav(); hvt = host.vessel_tree
hrc = np.asarray(hvt[RCCA].coordinates, dtype=np.float64)
hs  = cum(hrc)
print("host RCCA dtype", np.asarray(hvt[RCCA].coordinates).dtype, "n", len(hrc))

iv = DualDeviceNavTopBrain(anatomy_dir="/opt/eve_training/results_topbrain/anatomies",
                           seed=42, episodes_between_change=1, exclude=EXCL)
vt = iv.vessel_tree; names=list(vt.anatomy_names)

# ---- sign convention probe: a point deep inside the aortic trunk ----
def probe_sign(mesh_path, branches):
    r = vtk.vtkOBJReader(); r.SetFileName(mesh_path); r.Update()
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(r.GetOutput())
    # widest branch point = largest declared radius
    best=None; bestr=-1
    for b in branches:
        rr = np.asarray(getattr(b,'radii',[]),dtype=float)
        if rr.size:
            k=int(np.argmax(rr))
            if rr[k]>bestr: bestr=float(rr[k]); best=np.asarray(b.coordinates[k],dtype=float)
    d = imp.EvaluateFunction(*map(float,best))
    return d, bestr, best

print("HOST sign probe (deep interior): d=%.3f at declared r=%.2f" % probe_sign(hvt.mesh_path, hvt.branches)[:2])
vt._select(0)
print("MR001 sign probe: d=%.3f at declared r=%.2f" % probe_sign(vt.mesh_path, vt.branches)[:2])

# ---- divergence of each cohort RCCA from the host RCCA ----
out={}
allrc={}
for i,nm in enumerate(names):
    vt._select(i)
    rc = np.asarray(vt[RCCA].coordinates, dtype=np.float64); allrc[nm]=rc
    s = cum(rc); n=min(len(rc),len(hrc))
    dif = np.linalg.norm(rc[:n]-hrc[:n],axis=1)
    row={}
    for th in (1e-9,1e-6,1e-3,1e-2,0.1,1.0):
        j = int(np.argmax(dif>th)) if np.any(dif>th) else n
        row["s_div_%g"%th]=float(s[min(j,len(s)-1)])
    row["max_dif_first100"]=float(dif[:100].max())
    row["dif_at_100mm"]=float(dif[int(np.searchsorted(s,100.0))])
    row["dif_at_130mm"]=float(dif[int(np.searchsorted(s,130.0))])
    row["n"]=len(rc); row["s_total"]=float(s[-1])
    # radii divergence
    ra=np.asarray(vt[RCCA].radii,dtype=float); hra=np.asarray(hvt[RCCA].radii,dtype=float)
    rd=np.abs(ra[:n]-hra[:n])
    jr=int(np.argmax(rd>1e-6)) if np.any(rd>1e-6) else n
    row["s_raddiv"]=float(s[min(jr,len(s)-1)])
    out[nm]=row
    print("%-14s n=%3d s_tot=%.1f  s_div(1e-9)=%.2f (1e-3)=%.2f (0.1)=%.2f (1.0)=%.2f "
          "| dif@100=%.4f dif@130=%.4f | s_raddiv=%.1f" % (
        nm,row["n"],row["s_total"],row["s_div_1e-09"],row["s_div_0.001"],
        row["s_div_0.1"],row["s_div_1"],row["dif_at_100mm"],row["dif_at_130mm"],row["s_raddiv"]))
    sys.stdout.flush()

# ---- pairwise: where do the 22 first differ from EACH OTHER (>0.01 mm) ----
ns=[len(allrc[n]) for n in names]; m=min(ns)
stack=np.stack([allrc[n][:m] for n in names])
spread=np.linalg.norm(stack-stack.mean(0),axis=2).max(0)
s0=cum(allrc[names[0]])
for th in (1e-9,1e-3,1e-2,0.1,1.0):
    j=int(np.argmax(spread>th)) if np.any(spread>th) else m
    print("cohort mutual spread > %g first at idx %d  s=%.2f mm" % (th,j,s0[min(j,len(s0)-1)]))
json.dump(out, open("/scratch/div.json","w"))
print("done")
