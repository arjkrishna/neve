import os,sys,json
sys.path.insert(0,'/opt/eve_training/eve'); sys.path.insert(0,'/opt/eve_training/eve_bench')
import numpy as np
from eve_bench.dualdevicenavtopbrain import find_anatomies,_CENTERLINE_SUBDIR
from eve_bench.dualdevicenav import DualDeviceNav, load_branches
FLOOR,CEIL,KR,MAXR=2.,12.,1.5,12.
SOFF=33.4685613383592
def arclen(c): return np.concatenate([[0.],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))])
def pick(bs,k):
    for b in bs:
        if k in str(b.name).upper(): return b
h=DualDeviceNav(); hb=list(h.vessel_tree.branches); hr=pick(hb,"RCCA")
HC=np.asarray(hr.coordinates,float);HR=np.asarray(hr.radii,float);HS=arclen(HC)
Htol=np.maximum(2.,KR*np.clip(HR,FLOOR,CEIL))
hel=HS>=40; hroute=SOFF+HS[hel]; HMAX=hroute.max(); HMIN=hroute.min()
hz3=HS>=136.
roots,names=find_anatomies("/opt/eve_training/results_topbrain/anatomies",exclude=["topcow_mr_013","topcow_mr_014","topcow_mr_015"])
R=[];T=[];L={}
for root,nm in zip(roots,names):
    bl=load_branches(os.path.join(root,_CENTERLINE_SUBDIR)); rb=pick(bl,"RCCA")
    C=np.asarray(rb.coordinates,float);Rr=np.asarray(rb.radii,float);S=arclen(C)
    e=S>=40; R+= (SOFF+S[e]).tolist(); L[nm]=float(S[-1])
    tol=np.maximum(2.,KR*np.clip(Rr,FLOOR,CEIL)); T+=tol[S>=136.].tolist()
R=np.array(R);T=np.array(T)
htz=Htol[hz3]
print(json.dumps(dict(
 host_route_min=float(HMIN),host_route_max=float(HMAX),
 cohort_route_frac_above_host_max=float((R>HMAX).mean()),
 cohort_route_frac_below_host_min=float((R<HMIN).mean()),
 cohort_route_max=float(R.max()),cohort_route_min=float(R.min()),
 host_z3_tol_max=float(htz.max()),host_z3_tol_med=float(np.median(htz)),
 cohort_z3_tol_frac_above_host_max=float((T>htz.max()).mean()),
 cohort_z3_tol_med=float(np.median(T)),cohort_z3_tol_max=float(T.max()),
 obs49_gain_host_z3=1.0/(2*np.median(htz)),obs49_gain_cohort_z3=1.0/(2*np.median(T)),
 obs49_gain_ratio=float(np.median(htz)/np.median(T)),
 rcca_len_host=float(HS[-1]),
 n_cohort_longer_than_host=int(sum(1 for v in L.values() if v>HS[-1])),
 rcca_lens={k:round(v,2) for k,v in sorted(L.items())}),indent=1))
