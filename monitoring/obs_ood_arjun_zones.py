import os,sys,json
sys.path.insert(0,'/opt/eve_training/eve'); sys.path.insert(0,'/opt/eve_training/eve_bench')
import numpy as np
from eve_bench.dualdevicenavtopbrain import find_anatomies,_CENTERLINE_SUBDIR
from eve_bench.dualdevicenav import DualDeviceNav, load_branches
FLOOR,CEIL,KR,MINTOL,MAXR,LOOK=2.,12.,1.5,2.,12.,20.
def arclen(c): return np.concatenate([[0.],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))])
def pick(bs,k):
    for b in bs:
        if k in str(b.name).upper(): return b
def r_at(s,r,q):
    q=np.clip(q,0,s[-1]); return r[np.argmin(np.abs(s[:,None]-q[None,:]),axis=0)]
def F(C,R):
    S=arclen(C); rc=np.clip(R,FLOOR,CEIL); return dict(S=S,r=R,rc=rc,tol=np.maximum(MINTOL,KR*rc),
        o47=rc/MAXR,o48=np.clip(r_at(S,R,S+LOOK),FLOOR,CEIL)/MAXR)
h=DualDeviceNav(); hb=list(h.vessel_tree.branches); hr=pick(hb,"RCCA")
H=F(np.asarray(hr.coordinates,float),np.asarray(hr.radii,float))
ZON={"Z1_shared_0_103":(0,103.4),"Z2_ramp_103_136":(103.4,136.0),"Z3_graft_136_end":(136.0,1e9),
     "TGT_eligible_40_end":(40.0,1e9)}
roots,names=find_anatomies("/opt/eve_training/results_topbrain/anatomies",exclude=["topcow_mr_013","topcow_mr_014","topcow_mr_015"])
CO=[]
for root,nm in zip(roots,names):
    bl=load_branches(os.path.join(root,_CENTERLINE_SUBDIR)); rb=pick(bl,"RCCA")
    CO.append((nm,F(np.asarray(rb.coordinates,float),np.asarray(rb.radii,float))))
def stats(v):
    v=np.asarray(v,float)
    return dict(n=len(v),min=float(v.min()),p05=float(np.percentile(v,5)),med=float(np.median(v)),
        p95=float(np.percentile(v,95)),max=float(v.max()),std=float(v.std()),
        iqr=float(np.percentile(v,75)-np.percentile(v,25)),uniq=int(len(np.unique(np.round(v,6)))))
out={}
for zn,(a,b) in ZON.items():
    hm=(H["S"]>=a)&(H["S"]<b)
    z={"host":{},"cohort":{},"per_anat":{}}
    for k in ["r","o47","o48","tol"]:
        z["host"][k]=stats(H[k][hm])
    z["host"]["clamped_frac"]=float((H["r"][hm]<FLOOR).mean())
    z["host"]["o47_pinned_frac"]=float((H["o47"][hm]<=1/6+1e-9).mean())
    z["host"]["o48_pinned_frac"]=float((H["o48"][hm]<=1/6+1e-9).mean())
    pool={k:[] for k in ["r","o47","o48","tol"]}; cl=[];pin=[];pin48=[];uq=[]
    for nm,P in CO:
        m=(P["S"]>=a)&(P["S"]<b)
        for k in pool: pool[k]+=P[k][m].tolist()
        cl.append(float((P["r"][m]<FLOOR).mean())); pin.append(float((P["o47"][m]<=1/6+1e-9).mean()))
        pin48.append(float((P["o48"][m]<=1/6+1e-9).mean()))
        uq.append(int(len(np.unique(np.round(P["o47"][m],6)))))
        z["per_anat"][nm]=dict(n=int(m.sum()),clamp=round(float((P["r"][m]<FLOOR).mean()),4),
            r_med=round(float(np.median(P["r"][m])),4),o47_med=round(float(np.median(P["o47"][m])),4),
            o47_max=round(float(P["o47"][m].max()),4),tol_med=round(float(np.median(P["tol"][m])),4))
    for k in pool: z["cohort"][k]=stats(pool[k])
    z["cohort"]["clamped_frac_per_anat"]=[float(np.min(cl)),float(np.median(cl)),float(np.max(cl))]
    z["cohort"]["o47_pinned_frac_per_anat"]=[float(np.min(pin)),float(np.median(pin)),float(np.max(pin))]
    z["cohort"]["o48_pinned_frac_per_anat"]=[float(np.min(pin48)),float(np.median(pin48)),float(np.max(pin48))]
    z["cohort"]["o47_uniq_per_anat"]=[int(np.min(uq)),int(np.median(uq)),int(np.max(uq))]
    z["compare"]={}
    for k in ["r","o47","o48","tol"]:
        Hv=H[k][hm]; P=np.array(pool[k]); lo,hi=Hv.min(),Hv.max()
        z["compare"][k]=dict(frac_in=float(((P>=lo)&(P<=hi)).mean()),
            frac_above=float((P>hi).mean()),frac_below=float((P<lo).mean()),
            shift_iqr=float((np.median(P)-np.median(Hv))/max(np.percentile(Hv,75)-np.percentile(Hv,25),1e-9)),
            shift_std=float((np.median(P)-np.median(Hv))/max(Hv.std(),1e-9)))
    out[zn]=z
# route / target
SOFF=33.4685613383592
rt={"host_rcca_len":float(H["S"][-1]),"host_route_total":float(H["S"][-1]+SOFF)}
el=H["S"]>=40
rt["host_target_route"]=[float(SOFF+H["S"][el].min()),float(SOFF+np.median(H["S"][el])),float(SOFF+H["S"][el].max())]
L=[float(P["S"][-1]) for _,P in CO]
rt["cohort_rcca_len"]=[min(L),float(np.median(L)),max(L)]
tm=[];tx=[]
for _,P in CO:
    e=P["S"]>=40; tm.append(float(SOFF+np.median(P["S"][e]))); tx.append(float(SOFF+P["S"][e].max()))
rt["cohort_target_route_med"]=[min(tm),float(np.median(tm)),max(tm)]
rt["cohort_target_route_max"]=[min(tx),float(np.median(tx)),max(tx)]
rt["host_d_rem_norm_start"]=rt["host_route_total"]/400.
rt["cohort_d_rem_norm_start"]=[ (l+SOFF)/400. for l in (min(L),max(L))]
out["route"]=rt
print("JSONSTART"); print(json.dumps(out,indent=1))
