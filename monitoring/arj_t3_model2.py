"""TASK3 v2: de-collinearized models, bootstrapped anatomy LRT, paired contrast."""
import json, math, collections
import numpy as np
from scipy import stats, optimize
R=json.load(open("/opt/mon/arj_t3_rows.json"))
ANAT=sorted({r["anat"] for r in R}); n=len(R); K=len(ANAT)
aid=np.array([ANAT.index(r["anat"]) for r in R])
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
F={k:np.array([r[k] for r in R],float) for k in
   ("dg","frac","s_end","L","r_t","d_t","k3_t","cum_t","d_w","k3_w","r_w","d_sp","k3_sp","r_sp")}
F["logd_t"]=np.log(np.maximum(F["d_t"],0.02)); F["logd_sp"]=np.log(np.maximum(F["d_sp"],0.02))
def zs(v): v=np.asarray(v,float); return (v-v.mean())/v.std(ddof=0)
def logfit(X,y,ridge=1e-6,maxit=300):
    b=np.zeros(X.shape[1])
    for _ in range(maxit):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        s=np.linalg.solve(H,g); b=b+s
        if np.max(np.abs(s))<1e-11: break
    p=1/(1+np.exp(-(X@b)))
    ll=float((y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum())
    W=np.maximum(p*(1-p),1e-9); H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1])
    return b,np.sqrt(np.diag(np.linalg.pinv(H))),ll,p
# residualize geometry on depth (dg)
dgz=zs(F["dg"]); Xd=np.column_stack([np.ones(n),dgz])
RES={}
for k in ("k3_sp","logd_sp","r_sp","cum_t","k3_t","logd_t","r_t","k3_w","d_w","L"):
    v=zs(F[k]); c=np.linalg.lstsq(Xd,v,rcond=None)[0]; rr=v-Xd@c
    RES[k+"_r"]=rr/rr.std(ddof=0)
    print("[resid] %-9s R2_on_depth=%.3f"%(k,1-rr.var()/v.var()))
def build(keys):
    cols=[np.ones(n)]
    for k in keys: cols.append(RES[k] if k.endswith("_r") else zs(F[k]))
    return np.column_stack(cols)
def show(lab,nm,keys,y):
    X=build(keys); b,se,ll,_=logfit(X,y)
    print("\n[%s|%s] logLik=%.3f k=%d"%(lab,nm,ll,X.shape[1]))
    for j,name in enumerate(["intercept"]+list(keys)):
        z=b[j]/se[j]; print("   %-11s b=%7.3f se=%6.3f z=%6.2f p=%.4f"%(name,b[j],se[j],z,2*stats.norm.sf(abs(z))))
    return ll,X.shape[1]
SETS=[("A0 null",[]),("A1 depth",["dg"]),
      ("A2 depth+geo_resid",["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"]),
      ("A3 depth+localtgt_resid",["dg","logd_t_r","r_t_r","k3_t_r"]),
      ("B1 turning only",["cum_t"])]
LL={}
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    LL[lab]={}
    for nm,keys in SETS: LL[lab][nm]=show(lab,nm,keys,y)
print("\n[LRT]")
for lab in LL:
    for a,b_ in (("A0 null","A1 depth"),("A1 depth","A2 depth+geo_resid"),("A1 depth","A3 depth+localtgt_resid"),("A0 null","B1 turning only")):
        d=2*(LL[lab][b_][0]-LL[lab][a][0]); df=LL[lab][b_][1]-LL[lab][a][1]
        print("  %-9s %-22s -> %-24s LR=%6.3f df=%d p=%.4f"%(lab,a,b_,d,df,stats.chi2.sf(max(d,0),df)))

# ---- anatomy LRT conditional on depth, parametric bootstrap null ----
D=np.zeros((n,K-1))
for i,a in enumerate(aid):
    if a>0: D[i,a-1]=1
def anat_lr(y,keys,ridge=1e-4):
    X0=build(keys); _,_,ll0,p0=logfit(X0,y)
    X1=np.column_stack([X0,D]); _,_,ll1,_=logfit(X1,y,ridge=ridge)
    return 2*(ll1-ll0),p0
print("\n[ANATOMY DUMMIES LRT, parametric-bootstrap null, ridge=1e-4]")
rng=np.random.default_rng(11); B=2000
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in (("uncond",[]),("|depth",["dg"]),("|depth+geo_resid",["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"])):
        obs,pfit=anat_lr(y,keys)
        cnt=0
        for _ in range(B):
            ys=(rng.random(n)<pfit).astype(float)
            try: s,_=anat_lr(ys,keys)
            except Exception: continue
            if s>=obs: cnt+=1
        print("  %-9s %-18s LR=%7.3f df=%d  p_asym=%.4f  p_boot=%.4f"%(
            lab,nm,obs,K-1,stats.chi2.sf(max(obs,0),K-1),(cnt+1)/(B+1)))

# ---- GLMM sigma_u on de-collinearized models ----
gx,gw=np.polynomial.hermite_e.hermegauss(60)
def glmm(keys,y):
    X=build(keys); b0,_,ll0,_=logfit(X,y)
    def nll(par):
        b=par[:X.shape[1]]; sig=math.exp(par[-1]); e0=X@b; t=0.0
        for a in range(K):
            m=aid==a
            if not m.any(): continue
            e=e0[m][:,None]+sig*gx[None,:]; p=1/(1+np.exp(-e))
            lp=(y[m][:,None]*np.log(np.clip(p,1e-12,1))+(1-y[m][:,None])*np.log(np.clip(1-p,1e-12,1))).sum(0)
            mx=lp.max(); t+=np.log((gw*np.exp(lp-mx)).sum())+mx-0.5*math.log(2*math.pi)
        return -t
    best=None
    for s0 in (-3.,-1.5,-0.5,0.3,0.8):
        r=optimize.minimize(nll,np.concatenate([b0,[s0]]),method="Nelder-Mead",
                            options=dict(maxiter=60000,maxfev=60000,xatol=1e-8,fatol=1e-10))
        if best is None or r.fun<best.fun: best=r
    sig=math.exp(best.x[-1]); ll=-best.fun; lr=2*(ll-ll0)
    return sig,lr,0.5*stats.chi2.sf(max(lr,0),1)
print("\n[GLMM sigma_u]")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in (("uncond",[]),("|depth",["dg"]),("|depth+geo_resid",["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"])):
        sig,lr,p=glmm(keys,y)
        print("  %-9s %-18s sigma_u=%.4f ICC=%.4f LR=%.3f p=%.4f"%(lab,nm,sig,sig*sig/(sig*sig+math.pi**2/3),lr,p))
