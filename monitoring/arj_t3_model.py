"""TASK3 models: logistic + anatomy random effect + paired teacher/heuristic contrast."""
import json, math, collections
import numpy as np
from scipy import stats, optimize
R=json.load(open("/opt/mon/arj_t3_rows.json"))
ANAT=sorted({r["anat"] for r in R})
n=len(R); print("n=",n,"k=",len(ANAT))

def logistic_fit(X,y,ridge=1e-6,maxit=200):
    b=np.zeros(X.shape[1])
    for _ in range(maxit):
        eta=X@b; p=1/(1+np.exp(-eta)); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(H,g,rcond=None)[0]
        b=b+step
        if np.max(np.abs(step))<1e-10: break
    eta=X@b; p=1/(1+np.exp(-eta))
    ll=float(np.sum(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))))
    W=np.maximum(p*(1-p),1e-9); H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1])
    se=np.sqrt(np.diag(np.linalg.pinv(H)))
    return b,se,ll

def zs(v):
    v=np.asarray(v,float); return (v-v.mean())/v.std(ddof=0), float(v.mean()), float(v.std(ddof=0))

FE={}
for k in ("dg","s","frac","s_end","L","r_t","d_t","k3_t","k1_t","cum_t","d_w","k3_w","r_w","d_sp","k3_sp","r_sp"):
    FE[k]=np.array([r[k] for r in R],float)
FE["logd_t"]=np.log(np.maximum(FE["d_t"],0.02))
FE["logd_sp"]=np.log(np.maximum(FE["d_sp"],0.02))
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
aid=np.array([ANAT.index(r["anat"]) for r in R])

print("\n[UNIVARIATE] episode-level logistic, one predictor at a time (standardized)")
print("%-10s %8s %8s %8s %8s | %8s %8s %8s"%("feat","bT","seT","zT","pT","bH","seH","pH"))
uni={}
for k in ("dg","s","frac","s_end","L","r_t","logd_t","k3_t","k1_t","cum_t","d_w","k3_w","r_w","logd_sp","k3_sp","r_sp"):
    x,_,_=zs(FE[k]); X=np.column_stack([np.ones(n),x])
    bT,seT,_=logistic_fit(X,yT); bH,seH,_=logistic_fit(X,yH)
    zT=bT[1]/seT[1]; zH=bH[1]/seH[1]
    uni[k]=(bT[1],seT[1],2*stats.norm.sf(abs(zT)))
    print("%-10s %8.3f %8.3f %8.2f %8.4f | %8.3f %8.3f %8.4f"%(k,bT[1],seT[1],zT,2*stats.norm.sf(abs(zT)),bH[1],seH[1],2*stats.norm.sf(abs(zH))))

# correlation among candidate predictors
print("\n[COLLINEARITY] pearson r among selected")
sel=["dg","frac","s_end","L","r_sp","logd_sp","k3_sp","cum_t","r_t","logd_t","k3_t"]
M=np.array([zs(FE[k])[0] for k in sel])
print("      "+" ".join("%7s"%k for k in sel))
for i,k in enumerate(sel):
    print("%-6s"%k+" ".join("%7.2f"%np.corrcoef(M[i],M[j])[0,1] for j in range(len(sel))))

# ---------- nested models ----------
def build(keys): 
    return np.column_stack([np.ones(n)]+[zs(FE[k])[0] for k in keys])
def anat_dummies():
    D=np.zeros((n,len(ANAT)-1))
    for i,a in enumerate(aid):
        if a>0: D[i,a-1]=1
    return D
D=anat_dummies()

def report(name,keys,y,label):
    X=build(keys); b,se,ll=logistic_fit(X,y)
    print("\n[%s | %s] keys=%s  logLik=%.3f  df=%d"%(label,name,keys,ll,X.shape[1]))
    nm=["intercept"]+list(keys)
    for j in range(X.shape[1]):
        z=b[j]/se[j]
        print("   %-10s b=%7.3f se=%6.3f z=%6.2f p=%.4f  OR/sd=%.3f"%(nm[j],b[j],se[j],z,2*stats.norm.sf(abs(z)),math.exp(b[j])))
    return ll,X

MODELS=[("M0 null",[]),("M1 depth",["dg"]),("M1b depth+frac",["dg","frac"]),
        ("M2 depth+localgeo",["dg","k3_sp","logd_sp","r_sp","cum_t"]),
        ("M3 full",["dg","frac","L","k3_sp","logd_sp","r_sp","cum_t","k3_t","logd_t","r_t"])]
LLT={}; LLH={}
for lab,y,store in (("TEACHER",yT,LLT),("HEURISTIC",yH,LLH)):
    for nm,keys in MODELS:
        ll,X=report(nm,keys,y,lab); store[nm]=(ll,len(keys)+1)

print("\n[LRT nested]")
for lab,store in (("TEACHER",LLT),("HEURISTIC",LLH)):
    for a,b_ in (("M0 null","M1 depth"),("M1 depth","M2 depth+localgeo"),("M2 depth+localgeo","M3 full")):
        d=2*(store[b_][0]-store[a][0]); df=store[b_][1]-store[a][1]
        print("  %-9s %-18s -> %-18s LR=%7.3f df=%d p=%.4f"%(lab,a,b_,d,df,stats.chi2.sf(max(d,0),df)))

# ---------- anatomy fixed effects, conditional on covariates ----------
print("\n[ANATOMY FIXED EFFECTS LRT] does adding 21 anatomy dummies improve fit?")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in [("uncond",[]),("|depth",["dg"]),("|depth+geo",["dg","k3_sp","logd_sp","r_sp","cum_t"])]:
        X0=build(keys); _,_,ll0=logistic_fit(X0,y)
        X1=np.column_stack([X0,D]); _,_,ll1=logistic_fit(X1,y,ridge=1e-4)
        d=2*(ll1-ll0); df=D.shape[1]
        print("  %-9s %-12s LR=%7.3f df=%d p=%.4f"%(lab,nm,d,df,stats.chi2.sf(max(d,0),df)))

# ---------- GLMM random intercept, adaptive-free 40-node GH ----------
gh_x,gh_w=np.polynomial.hermite_e.hermegauss(60)
def glmm_nll(par,X,y,aid,k):
    b=par[:X.shape[1]]; ls=par[-1]; sig=math.exp(ls)
    eta0=X@b; tot=0.0
    for a in range(k):
        m=aid==a
        if not m.any(): continue
        e=eta0[m][:,None]+sig*gh_x[None,:]
        p=1/(1+np.exp(-e))
        lp=(y[m][:,None]*np.log(np.clip(p,1e-12,1))+(1-y[m][:,None])*np.log(np.clip(1-p,1e-12,1))).sum(0)
        mx=lp.max(); val=np.log(np.sum(gh_w*np.exp(lp-mx)))+mx-0.5*math.log(2*math.pi)
        tot+=val
    return -tot
def fit_glmm(keys,y):
    X=build(keys); k=len(ANAT)
    b0,_,ll0=logistic_fit(X,y)
    best=None
    for s0 in (-2.5,-1.0,-0.3,0.4):
        p0=np.concatenate([b0,[s0]])
        res=optimize.minimize(glmm_nll,p0,args=(X,y,aid,k),method="Nelder-Mead",
                              options=dict(maxiter=40000,maxfev=40000,xatol=1e-7,fatol=1e-9))
        if best is None or res.fun<best.fun: best=res
    sig=math.exp(best.x[-1]); ll=-best.fun
    lr=2*(ll-ll0); p=0.5*stats.chi2.sf(max(lr,0),1)
    return sig,ll,ll0,lr,p,best.x[:X.shape[1]]
print("\n[GLMM random anatomy intercept]  (LRT sigma=0 uses 0.5*chi2_1 mixture)")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in [("uncond",[]),("|depth",["dg"]),("|depth+geo",["dg","k3_sp","logd_sp","r_sp","cum_t"])]:
        sig,ll,ll0,lr,p,bb=fit_glmm(keys,y)
        icc=sig*sig/(sig*sig+math.pi*math.pi/3)
        print("  %-9s %-12s sigma_u=%.4f  ICC=%.4f  logLik=%.3f (fixed %.3f) LR=%.3f p=%.4f"%(lab,nm,sig,icc,ll,ll0,lr,p))
json.dump({"ok":1},open("/opt/mon/arj_t3_model_done.json","w"))
