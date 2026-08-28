import os,sys,json
sys.path.insert(0,'/opt/eve_training/eve'); sys.path.insert(0,'/opt/eve_training/eve_bench')
import numpy as np
from eve_bench.dualdevicenavtopbrain import find_anatomies,_CENTERLINE_SUBDIR
from eve_bench.dualdevicenav import DualDeviceNav, load_branches
def arclen(c): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))])
def pick(bs,k):
    for b in bs:
        if k in str(b.name).upper(): return b
h=DualDeviceNav(); hb=list(h.vessel_tree.branches); hr=pick(hb,"RCCA")
HC=np.asarray(hr.coordinates,float); HR=np.asarray(hr.radii,float); HS=arclen(HC)
roots,names=find_anatomies("/opt/eve_training/results_topbrain/anatomies",exclude=["topcow_mr_013","topcow_mr_014","topcow_mr_015"])
out={}
for root,nm in zip(roots,names):
    bl=load_branches(os.path.join(root,_CENTERLINE_SUBDIR)); rb=pick(bl,"RCCA")
    C=np.asarray(rb.coordinates,float); R=np.asarray(rb.radii,float); S=arclen(C)
    # arclength-matched deviation: for each host station s, nearest cohort point
    dev=[]
    for s,p in zip(HS,HC):
        j=int(np.argmin(np.abs(S-s)))
        dev.append(np.linalg.norm(C[j]-p))
    dev=np.array(dev)
    # radius deviation at matched arclength
    rdev=np.array([abs(R[int(np.argmin(np.abs(S-s)))]-rr) for s,rr in zip(HS,HR)])
    def first(th,a): 
        i=np.argmax(a>th); return float(HS[i]) if np.any(a>th) else None
    out[nm]=dict(dev_first_0p5=first(0.5,dev),dev_first_2=first(2.0,dev),dev_first_5=first(5.0,dev),
                 rdev_first_0p05=first(0.05,rdev),rdev_first_0p2=first(0.2,rdev),
                 dev_max=float(dev.max()),rcca_len=float(S[-1]),host_len=float(HS[-1]),
                 dev_at=[float(dev[np.argmin(np.abs(HS-x))]) for x in (50,100,120,130,140,160)],
                 rdev_at=[float(rdev[np.argmin(np.abs(HS-x))]) for x in (50,100,120,130,140,160)])
print(json.dumps(out,indent=1))
