import json, math
import numpy as np
from scipy import stats
R=json.load(open("/opt/mon/arj_t3_rows.json"))
E=json.load(open("/opt/mon/arj_t3_merged.json"))
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float); n=len(R)
V={k:np.array([r[k] for r in R],float) for k in ("dg","s","pl","frac","s_end","L","cum_t","k3_sp","d_sp","r_sp")}
print("[WHICH AXIS LOCATES THE TEACHER CLIFF BEST?] best 2-group split, max Fisher -log10 p, min 20% per side")
for k in ("dg","s","pl","frac","s_end","cum_t","k3_sp","d_sp","r_sp","L"):
    x=V[k]; best=(0,None)
    for q in np.quantile(x,np.linspace(0.15,0.85,60)):
        a=yT[x<q]; b=yT[x>=q]
        if len(a)<15 or len(b)<15: continue
        p=stats.fisher_exact([[int(a.sum()),len(a)-int(a.sum())],[int(b.sum()),len(b)-int(b.sum())]])[1]
        if -math.log10(p)>best[0]: best=(-math.log10(p),q,a.mean(),b.mean(),len(a),len(b))
    ah=yH[x<best[1]].mean(); bh=yH[x>=best[1]].mean()
    print("  %-7s cut=%8.2f  T: %.3f(n=%d) -> %.3f(n=%d)  -log10p=%5.2f | H: %.3f -> %.3f"%(
        k,best[1],best[2],best[4],best[3],best[5],best[0],ah,bh))
# monotone-vs-quadratic AIC on each axis for teacher
def logfit(X,y):
    b=np.zeros(X.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1]); st=np.linalg.solve(H,X.T@(y-p)); b=b+st
        if np.max(np.abs(st))<1e-11: break
    p=1/(1+np.exp(-(X@b)))
    return float((y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum())
print("\n[QUADRATIC FIT logLik by axis, TEACHER / HEURISTIC]  (null T=-85.158 H=-84.639)")
for k in ("dg","s","pl","frac","s_end","cum_t","k3_sp","d_sp","r_sp"):
    z=(V[k]-V[k].mean())/V[k].std(); X=np.column_stack([np.ones(n),z,z*z])
    print("  %-7s T=%8.3f  H=%8.3f"%(k,logfit(X,yT),logfit(X,yH)))
# expected phi under conditional independence given A2b-ish quadratic depth model
zd=(V["dg"]-V["dg"].mean())/V["dg"].std(); X=np.column_stack([np.ones(n),zd,zd*zd])
def fitp(y):
    b=np.zeros(3)
    for _ in range(400):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+1e-6*np.eye(3); st=np.linalg.solve(H,X.T@(y-p)); b=b+st
        if np.max(np.abs(st))<1e-11: break
    return 1/(1+np.exp(-(X@b)))
pT=fitp(yT); pH=fitp(yH)
cov=np.mean(pT*pH)-np.mean(pT)*np.mean(pH)
sT=math.sqrt(np.mean(pT*(1-pT))+np.var(pT)); sH=math.sqrt(np.mean(pH*(1-pH))+np.var(pH))
print("\n[phi] observed=%.4f ; expected under conditional independence given quadratic depth=%.4f"%(
    np.corrcoef(yT,yH)[0,1],cov/(sT*sH)))
print("  both-succeed obs=%d  exp_cond_indep=%.1f  both-fail obs=%d exp=%.1f"%(
    int(((yT==1)&(yH==1)).sum()),float((pT*pH).sum()),int(((yT==0)&(yH==0)).sum()),float(((1-pT)*(1-pH)).sum())))
# teacher-minus-heuristic by depth band
print("\n[PAIRED DIFFERENCE by depth band]  d = T - H, exact McNemar within band")
bands=[(0,30),(30,60),(60,80),(80,120)]
dgv=V["dg"]
for lo,hi in bands:
    m=(dgv>=lo)&(dgv<hi)
    to=int(((yT==1)&(yH==0)&m).sum()); ho=int(((yT==0)&(yH==1)&m).sum())
    bb=int(((yT==1)&(yH==1)&m).sum()); nn=int(((yT==0)&(yH==0)&m).sum())
    p=stats.binomtest(to,to+ho,0.5).pvalue if to+ho>0 else 1.0
    print("  dg[%3d,%3d) n=%2d  BB=%2d Tonly=%2d Honly=%2d NN=%2d   T=%.3f H=%.3f  d=%+.3f McNemar p=%.4f"%(
        lo,hi,int(m.sum()),bb,to,ho,nn,yT[m].mean(),yH[m].mean(),yT[m].mean()-yH[m].mean(),p))
# non-grafted band too
ng=[e for e in E if e["pl"]<=166.91]
print("  s<=133.6 (pre-seam) n=%d  BB=%d Tonly=%d Honly=%d NN=%d  T=1.000 H=%.3f McNemar p=%.3g"%(
    len(ng),sum(1 for e in ng if e["T"] and e["H"]),sum(1 for e in ng if e["T"] and not e["H"]),
    sum(1 for e in ng if not e["T"] and e["H"]),sum(1 for e in ng if not e["T"] and not e["H"]),
    np.mean([e["H"] for e in ng]),stats.binomtest(29,29,0.5).pvalue))
