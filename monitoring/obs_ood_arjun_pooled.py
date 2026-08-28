import os, json, sys
sys.path.insert(0,'/opt/eve_training/eve'); sys.path.insert(0,'/opt/eve_training/eve_bench')
import numpy as np
from eve_bench.dualdevicenavtopbrain import find_anatomies, _CENTERLINE_SUBDIR
from eve_bench.dualdevicenav import DualDeviceNav, load_branches

FLOOR,CEIL,KR,MINTOL,MAXR,LOOK = 2.0,12.0,1.5,2.0,12.0,20.0
ADIR="/opt/eve_training/results_topbrain/anatomies"; EXCL=["topcow_mr_013","topcow_mr_014","topcow_mr_015"]
def arclen(c):
    return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))])
def pick(bs,k):
    for b in bs:
        if k in str(b.name).upper(): return b
def bridge(bs):
    for b in bs:
        if "(11)" in str(b.name): return b
def r_at(s,r,q):
    q=np.clip(q,0,s[-1]); return r[np.argmin(np.abs(s[:,None]-q[None,:]),axis=0)]

h=DualDeviceNav(); hb=list(h.vessel_tree.branches)
hr=pick(hb,"RCCA"); HC=np.asarray(hr.coordinates,float); HR=np.asarray(hr.radii,float); HS=arclen(HC)
br=bridge(hb); ost=HC[0]; bc=np.asarray(br.coordinates,float)
if np.linalg.norm(bc[0]-ost)<np.linalg.norm(bc[-1]-ost): bc=bc[::-1]
bs_=arclen(bc); SOFF=float(bs_[-1]-bs_[2])

def feats(C,R):
    S=arclen(C); rc=np.clip(R,FLOOR,CEIL); tol=np.maximum(MINTOL,KR*rc)
    o47=rc/MAXR; o48=np.clip(r_at(S,R,S+LOOK),FLOOR,CEIL)/MAXR
    return S,rc,tol,o47,o48

res={"SOFF":SOFF}
HSs,Hrc,Htol,Ho47,Ho48=feats(HC,HR)
roots,names=find_anatomies(ADIR,exclude=EXCL)
div={}; pool={k:[] for k in ["r","o47","o48","tol"]}; per={}
routes={}
for root,nm in zip(roots,names):
    bl=load_branches(os.path.join(root,_CENTERLINE_SUBDIR)); rb=pick(bl,"RCCA")
    C=np.asarray(rb.coordinates,float); R=np.asarray(rb.radii,float)
    S,rc,tol,o47,o48=feats(C,R)
    n=min(len(C),len(HC))
    d=np.linalg.norm(C[:n]-HC[:n],axis=1)
    k=int(np.argmax(d>1e-6)) if np.any(d>1e-6) else n-1
    sdiv=float(S[k]); div[nm]=sdiv
    m=S>sdiv
    per[nm]=dict(sdiv=sdiv,ndist=int(m.sum()),n=len(S),rcca_len=float(S[-1]),
        r=[float(R[m].min()),float(np.median(R[m])),float(R[m].max()),float(R[m].std())],
        o47=[float(o47[m].min()),float(np.median(o47[m])),float(o47[m].max()),float(o47[m].std())],
        o48=[float(o48[m].min()),float(np.median(o48[m])),float(o48[m].max()),float(o48[m].std())],
        tol=[float(tol[m].min()),float(np.median(tol[m])),float(tol[m].max())],
        clamp=float((R[m]<FLOOR).mean()), pinned47=float((o47[m]<=1/6+1e-9).mean()),
        uniq47=int(len(np.unique(np.round(o47[m],6)))))
    pool["r"]+=R[m].tolist(); pool["o47"]+=o47[m].tolist(); pool["o48"]+=o48[m].tolist(); pool["tol"]+=tol[m].tolist()
    el=S>=40.0
    routes[nm]=[float(SOFF+S[el].min()),float(SOFF+np.median(S[el])),float(SOFF+S[el].max()),float(SOFF+S[-1])]

SD=float(np.median(list(div.values())))
hm=HSs>SD
host=dict(sdiv_med=SD,ndist=int(hm.sum()),n=len(HSs),rcca_len=float(HSs[-1]),
  r=[float(HR[hm].min()),float(np.median(HR[hm])),float(HR[hm].max()),float(HR[hm].std())],
  o47=[float(Ho47[hm].min()),float(np.median(Ho47[hm])),float(Ho47[hm].max()),float(Ho47[hm].std())],
  o48=[float(Ho48[hm].min()),float(np.median(Ho48[hm])),float(Ho48[hm].max()),float(Ho48[hm].std())],
  tol=[float(Htol[hm].min()),float(np.median(Htol[hm])),float(Htol[hm].max())],
  clamp=float((HR[hm]<FLOOR).mean()),pinned47=float((Ho47[hm]<=1/6+1e-9).mean()),
  uniq47=int(len(np.unique(np.round(Ho47[hm],6)))),
  r_iqr=float(np.percentile(HR[hm],75)-np.percentile(HR[hm],25)),
  o47_iqr=float(np.percentile(Ho47[hm],75)-np.percentile(Ho47[hm],25)),
  tol_iqr=float(np.percentile(Htol[hm],75)-np.percentile(Htol[hm],25)))
el=HSs>=40.0
host["routes"]=[float(SOFF+HSs[el].min()),float(SOFF+np.median(HSs[el])),float(SOFF+HSs[el].max()),float(SOFF+HSs[-1])]
# whole-branch (trunk+graft) pooled too
host["all"]=dict(o47=[float(Ho47.min()),float(np.median(Ho47)),float(Ho47.max())],
                 clamp=float((HR<FLOOR).mean()),pinned47=float((Ho47<=1/6+1e-9).mean()))

ov={}
for k in ["r","o47","o48","tol"]:
    P=np.array(pool[k]); Hv={"r":HR[hm],"o47":Ho47[hm],"o48":Ho48[hm],"tol":Htol[hm]}[k]
    lo,hi=Hv.min(),Hv.max()
    ov[k]=dict(host=[float(lo),float(np.median(Hv)),float(hi)],
               cohort=[float(P.min()),float(np.median(P)),float(P.max())],
               frac_cohort_in_host_range=float(((P>=lo)&(P<=hi)).mean()),
               frac_cohort_above_host_max=float((P>hi).mean()),
               frac_cohort_below_host_min=float((P<lo).mean()),
               shift_med_over_host_iqr=float((np.median(P)-np.median(Hv))/max(np.percentile(Hv,75)-np.percentile(Hv,25),1e-9)),
               shift_med_over_host_std=float((np.median(P)-np.median(Hv))/max(Hv.std(),1e-9)),
               host_std=float(Hv.std()),cohort_std=float(P.std()),
               host_uniq=int(len(np.unique(np.round(Hv,6)))),cohort_uniq=int(len(np.unique(np.round(P,6)))))
print(json.dumps(dict(res=res,host=host,per=per,div=div,routes=routes,overlap=ov),indent=1))
