"""TASK3 v3: CV validation of A2, GLMM profile likelihood, paired teacher-vs-heuristic."""
import json, math, collections
import numpy as np
from scipy import stats, optimize
R=json.load(open("/opt/mon/arj_t3_rows.json"))
ANAT=sorted({r["anat"] for r in R}); n=len(R); K=len(ANAT)
aid=np.array([ANAT.index(r["anat"]) for r in R])
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
F={k:np.array([r[k] for r in R],float) for k in ("dg","frac","s_end","L","r_t","d_t","k3_t","cum_t","d_sp","k3_sp","r_sp")}
F["logd_t"]=np.log(np.maximum(F["d_t"],0.02)); F["logd_sp"]=np.log(np.maximum(F["d_sp"],0.02))
mu={k:F[k].mean() for k in F}; sd={k:F[k].std(ddof=0) for k in F}
Z={k:(F[k]-mu[k])/sd[k] for k in F}
dgz=Z["dg"]; Xd=np.column_stack([np.ones(n),dgz])
RES={}
for k in ("k3_sp","logd_sp","r_sp","cum_t","k3_t","logd_t","r_t"):
    c=np.linalg.lstsq(Xd,Z[k],rcond=None)[0]; rr=Z[k]-Xd@c; RES[k+"_r"]=rr/rr.std(ddof=0)
COL=dict(Z); COL.update(RES)
def build(keys,idx=None):
    cols=[np.ones(n)]+[COL[k] for k in keys]; X=np.column_stack(cols)
    return X if idx is None else X[idx]
def logfit(X,y,ridge=1e-6,maxit=400):
    b=np.zeros(X.shape[1])
    for _ in range(maxit):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        s=np.linalg.solve(H,g); b=b+s
        if np.max(np.abs(s))<1e-11: break
    p=1/(1+np.exp(-(X@b)))
    ll=float((y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum())
    return b,ll
SETS={"A0 null":[], "A1 depth":["dg"], "B1 turn":["cum_t"],
      "A2 depth+geo_r":["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"],
      "A2b depth+turn_r+k3_r":["dg","cum_t_r","k3_sp_r"]}
def auc(y,p):
    o=np.argsort(p); yy=y[o]; r=np.empty(n); r[o]=np.arange(1,n+1)
    pos=y.sum(); neg=n-pos
    return (r[y==1].sum()-pos*(pos+1)/2)/(pos*neg)
print("[CV] 10-fold stratified + leave-one-anatomy-out, mean out-of-sample log-loss (lower=better) and AUC")
rng=np.random.default_rng(3)
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in SETS.items():
        X=build(keys)
        # 10-fold repeated
        lls=[]; 
        for rep in range(20):
            perm=rng.permutation(n); folds=np.array_split(perm,10); tot=0.0
            for f in folds:
                tr=np.setdiff1d(np.arange(n),f)
                b,_=logfit(X[tr],y[tr],ridge=1e-3)
                p=1/(1+np.exp(-(X[f]@b)))
                tot+=-(y[f]*np.log(np.clip(p,1e-9,1))+(1-y[f])*np.log(np.clip(1-p,1e-9,1))).sum()
            lls.append(tot/n)
        # LOAO
        pv=np.zeros(n)
        for a in range(K):
            te=np.where(aid==a)[0]; tr=np.where(aid!=a)[0]
            b,_=logfit(X[tr],y[tr],ridge=1e-3); pv[te]=1/(1+np.exp(-(X[te]@b)))
        loao=-(y*np.log(np.clip(pv,1e-9,1))+(1-y)*np.log(np.clip(1-pv,1e-9,1))).mean()
        print("  %-9s %-22s 10fold=%.4f (sd %.4f)  LOAO=%.4f  LOAO_AUC=%.3f"%(lab,nm,np.mean(lls),np.std(lls),loao,auc(y,pv)))

# ---- GLMM profile over sigma ----
gx,gw=np.polynomial.hermite_e.hermegauss(80)
def marg_ll(b,sig,X,y):
    e0=X@b; t=0.0
    for a in range(K):
        m=aid==a
        if not m.any(): continue
        e=e0[m][:,None]+sig*gx[None,:]; p=1/(1+np.exp(-e))
        lp=(y[m][:,None]*np.log(np.clip(p,1e-12,1))+(1-y[m][:,None])*np.log(np.clip(1-p,1e-12,1))).sum(0)
        mx=lp.max(); t+=np.log((gw*np.exp(lp-mx)).sum())+mx-0.5*math.log(2*math.pi)
    return t
print("\n[GLMM profile logLik over sigma_u]")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in (("uncond",[]),("|depth",["dg"]),("|A2",["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"])):
        X=build(keys); b0,ll0=logfit(X,y); prof=[]
        for sig in (0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.4,2.0,3.0):
            if sig==0: prof.append((sig,ll0)); continue
            f=lambda b: -marg_ll(b,sig,X,y)
            r=optimize.minimize(f,b0,method="Nelder-Mead",options=dict(maxiter=50000,maxfev=50000,xatol=1e-8,fatol=1e-10))
            prof.append((sig,-r.fun))
        best=max(prof,key=lambda t:t[1]); lr=2*(best[1]-ll0)
        print("  %-9s %-8s sigma_hat=%.2f  LR=%.3f p=%.4f | "%(lab,nm,best[0],lr,0.5*stats.chi2.sf(max(lr,0),1))
              +" ".join("%.1f:%.2f"%(s,l) for s,l in prof))

# ---- TASK 4 PAIRED ----
print("\n[PAIRED 2x2 per anatomy]  BB=both succeed, TT=teacher-only, HH=heuristic-only, NN=both fail")
tot=[0,0,0,0]
print("%-12s %3s %4s %4s %4s %4s   T%%    H%%   d(T-H)"%("anatomy","n","BB","Tonly","Honly","NN"))
per={}
for a in ANAT:
    idx=[i for i in range(n) if aid[i]==ANAT.index(a)]
    bb=sum(1 for i in idx if yT[i]==1 and yH[i]==1)
    to=sum(1 for i in idx if yT[i]==1 and yH[i]==0)
    ho=sum(1 for i in idx if yT[i]==0 and yH[i]==1)
    nn=sum(1 for i in idx if yT[i]==0 and yH[i]==0)
    per[a]=(len(idx),bb,to,ho,nn)
    for j,v in enumerate((bb,to,ho,nn)): tot[j]+=v
    print("%-12s %3d %4d %4d %4d %4d  %5.1f %5.1f  %+5.1f"%(a,len(idx),bb,to,ho,nn,
        100*(bb+to)/len(idx),100*(bb+ho)/len(idx),100*(to-ho)/len(idx)))
print("%-12s %3d %4d %4d %4d %4d"%("TOTAL",n,*tot))
bb,to,ho,nn=tot
print("McNemar exact (b=%d Tonly, c=%d Honly): p=%.4f  (binom two-sided)"%(to,ho,stats.binomtest(to,to+ho,0.5).pvalue if hasattr(stats,'binomtest') else stats.binom_test(to,to+ho,0.5)))
print("phi(T,H)=%.3f  kappa=%.3f"%(np.corrcoef(yT,yH)[0,1],
      ( (bb+nn)/n - (((bb+to)*(bb+ho)+(ho+nn)*(to+nn))/n**2) )/(1-(((bb+to)*(bb+ho)+(ho+nn)*(to+nn))/n**2))))
# discordant-only: does anatomy predict WHICH way a discordant pair goes?
disc=[i for i in range(n) if yT[i]!=yH[i]]
print("\n[DISCORDANT] n=%d (Tonly=%d, Honly=%d)"%(len(disc),to,ho))
dy=np.array([1.0 if yT[i]==1 else 0.0 for i in disc])
da=aid[np.array(disc)]
by=collections.defaultdict(lambda:[0,0])
for i,a in zip(disc,da): by[ANAT[a]][0]+=1; by[ANAT[a]][1]+= (1 if yT[i]==1 else 0)
ks=sorted(by); m=len(disc); p=dy.mean(); G2=0.0;X2=0.0
for a in ks:
    ni,yi=by[a]; ei=ni*p
    for obs,exp in ((yi,ei),(ni-yi,ni-ei)):
        if obs>0: G2+=2*obs*math.log(obs/exp)
        X2+=(obs-exp)**2/exp
df=len(ks)-1
rng2=np.random.default_rng(5); ns=np.array([by[a][0] for a in ks]); B=200000
sim=rng2.binomial(ns[None,:],p,size=(B,len(ks))).astype(float); exp=ns[None,:]*p
with np.errstate(divide='ignore',invalid='ignore'):
    t1=np.where(sim>0,2*sim*np.log(np.where(sim>0,sim/exp,1)),0.0)
    f2=ns[None,:]-sim; ef=ns[None,:]-exp
    t2=np.where(f2>0,2*f2*np.log(np.where(f2>0,f2/ef,1)),0.0)
Gs=(t1+t2).sum(1); pb=float(((Gs>=G2).sum()+1)/(B+1))
print("  heterogeneity of P(teacher wins | discordant) across anatomies: G2=%.3f X2=%.3f df=%d p_asym=%.4f p_boot=%.4f"%(G2,X2,df,stats.chi2.sf(G2,df),pb))
for a in ks: print("    %-12s disc=%d teacher-wins=%d"%(a,by[a][0],by[a][1]))
# depth effect on discordance direction
dd=np.array([Z["dg"][i] for i in disc]); Xdd=np.column_stack([np.ones(len(disc)),dd])
b=np.zeros(2)
for _ in range(200):
    pp=1/(1+np.exp(-(Xdd@b))); W=np.maximum(pp*(1-pp),1e-9)
    H=Xdd.T@(Xdd*W[:,None])+1e-6*np.eye(2); g=Xdd.T@(dy-pp); b=b+np.linalg.solve(H,g)
se=np.sqrt(np.diag(np.linalg.pinv(Xdd.T@(Xdd*np.maximum(pp*(1-pp),1e-9))[:,None]*Xdd if False else Xdd.T@(Xdd*np.maximum(pp*(1-pp),1e-9)[:,None]))))
print("  logistic P(teacher wins|discordant) ~ depth: b_dg=%.3f se=%.3f p=%.4f"%(b[1],se[1],2*stats.norm.sf(abs(b[1]/se[1]))))
