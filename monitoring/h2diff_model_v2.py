import json, math, collections
import numpy as np
rng=np.random.default_rng(3)
M="monitoring/"; ROWS=json.load(open(M+"h2diff_eprows.json")); D=json.load(open(M+"h2diff_diff.json"))
def gammp(a,x):
    if x<=0: return 0.0
    if x<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(500):
            ap+=1; d*=x/ap; s+=d
            if abs(d)<abs(s)*1e-14: break
        return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    b=x+1-a; c=1e300; d=1.0/b; h=d
    for i in range(1,500):
        an=-i*(i-a); b+=2
        d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
        c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1.0/d; de=d*c; h*=de
        if abs(de-1)<1e-14: break
    return 1.0-math.exp(-x+a*math.log(x)-math.lgamma(a))*h
def chi2sf(x,df): return 1.0-gammp(df/2.0,x/2.0) if x>0 else 1.0
def normp(z): return math.erfc(abs(z)/math.sqrt(2))
def logit(X,y,ridge=1e-6,it=300):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p)+1e-9
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        d=np.linalg.solve(H,g); b+=d
        if np.max(np.abs(d))<1e-11: break
    p=np.clip(1/(1+np.exp(-X@b)),1e-12,1-1e-12)
    return b,float((y*np.log(p)+(1-y)*np.log(1-p)).sum())
def cse(X,y,b,groups):
    p=1/(1+np.exp(-X@b)); W=p*(1-p)+1e-9; A=np.linalg.pinv(X.T@(X*W[:,None])); B=np.zeros((X.shape[1],)*2)
    for g in set(groups):
        m=np.array([gg==g for gg in groups]); u=(X[m]*((y[m]-p[m])[:,None])).sum(0); B+=np.outer(u,u)
    return np.sqrt(np.diag(A@B@A))
anat=[r["anat"] for r in ROWS]; y=np.array([r["succ"] for r in ROWS],float); yh=np.array([r["hsucc"] for r in ROWS],float)
def z(k): v=np.array([r[k] for r in ROWS],float); return (v-v.mean())/v.std()
S=z("s_tgt"); TU=z("turn_cum"); TPM=z("turn_per_mm"); TO=z("tort"); RC=z("Rc_min"); CL=z("clr_min"); DGz=z("dgraft")
raw={k:np.array([r[k] for r in ROWS],float) for k in ["s_tgt","turn_cum","dgraft","turn_per_mm","tort","Rc_min","clr_min","r_min"]}
print("collinearity among episode-level route covariates (Pearson r):")
ks=list(raw)
for i in range(len(ks)):
    print("  "+ks[i]+": "+" ".join("%s=%+.2f"%(ks[j],np.corrcoef(raw[ks[i]],raw[ks[j]])[0,1]) for j in range(len(ks)) if j!=i))
one=np.ones(len(y))
A_=sorted(set(anat)); FE=np.stack([[1.0 if a==aa else 0.0 for aa in A_[1:]] for a in anat])
print("\n=== WITHIN-ANATOMY (anatomy fixed effects) episode-level effect of route geometry ===")
for yv,lab in [(y,"POLICY"),(yh,"HEUR")]:
    base=np.concatenate([np.stack([one],1),FE],1); _,ll0=logit(base,yv)
    for nm,v in [("s_tgt",S),("turn_cum",TU),("turn_per_mm",TPM),("tort",TO),("Rc_min",RC),("clr_min",CL)]:
        X=np.concatenate([np.stack([one,v],1),FE],1); b,ll=logit(X,yv)
        print("  %-5s %-12s b=%+.3f  LRT G2=%.2f df=1 p=%.4f"%(lab,nm,b[1],2*(ll-ll0),chi2sf(2*(ll-ll0),1)))
print("\n=== BETWEEN vs WITHIN decomposition for route turning (policy) ===")
gm={a:np.mean([raw["turn_cum"][i] for i in range(len(anat)) if anat[i]==a]) for a in A_}
bet=np.array([gm[a] for a in anat]); wit=raw["turn_cum"]-bet
bet=(bet-bet.mean())/bet.std(); wit=(wit-wit.mean())/wit.std()
for yv,lab in [(y,"POLICY"),(yh,"HEUR")]:
    X=np.stack([one,bet,wit],1); b,_=logit(X,yv); se=cse(X,yv,b,anat)
    print("  %-6s between-anatomy b=%+.3f (z=%+.2f p=%.3f) ; within-anatomy b=%+.3f (z=%+.2f p=%.3f)"
          %(lab,b[1],b[1]/se[1],normp(b[1]/se[1]),b[2],b[2]/se[2],normp(b[2]/se[2])))
print("\n=== route-clearance screen (per grafted episode) ===")
for thr,nm in [(0.18,"wire"),(0.30,"SOFA"),(0.35,"cath")]:
    bad=[(r["anat"],round(r["s_tgt"],1),r["succ"]) for r in ROWS if r["clr_min"]<thr]
    print("  episodes with route min clearance < %.2f (%s): %d  %s"%(thr,nm,len(bad),str(sorted(bad))[:200]))
print("\n=== exclude mr_025 : re-fit ===")
keep=np.array([a!="topcowmr025" for a in anat])
for yv,lab in [(y,"POLICY"),(yh,"HEUR")]:
    for nm,v in [("turn_cum",TU),("s_tgt",S),("DIFF",np.array([D["DIFF"][a] for a in anat]))]:
        X=np.stack([one[keep],v[keep]],1); b,_=logit(X,yv[keep]); se=cse(X,yv[keep],b,[a for a in anat if a!="topcowmr025"])
        print("  %-6s %-9s b=%+.3f z=%+.2f p=%.4f (n=%d)"%(lab,nm,b[1],b[1]/se[1],normp(b[1]/se[1]),keep.sum()))
