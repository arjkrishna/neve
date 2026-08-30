"""TASK3 v4: leak-free LOAO, residual diagnostics, bootstrap sensitivity, policy x anatomy."""
import json, math, collections
import numpy as np
from scipy import stats, optimize
R=json.load(open("/opt/mon/arj_t3_rows.json"))
ANAT=sorted({r["anat"] for r in R}); n=len(R); K=len(ANAT)
aid=np.array([ANAT.index(r["anat"]) for r in R])
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
RAW={k:np.array([r[k] for r in R],float) for k in ("dg","frac","L","r_t","d_t","k3_t","cum_t","d_sp","k3_sp","r_sp")}
RAW["logd_t"]=np.log(np.maximum(RAW["d_t"],0.02)); RAW["logd_sp"]=np.log(np.maximum(RAW["d_sp"],0.02))
def logfit(X,y,ridge=1e-6,maxit=400):
    b=np.zeros(X.shape[1])
    for _ in range(maxit):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        s=np.linalg.solve(H,g); b=b+s
        if np.max(np.abs(s))<1e-11: break
    p=1/(1+np.exp(-(X@b)))
    ll=float((y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum())
    W=np.maximum(p*(1-p),1e-9); H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1])
    return b,ll,np.sqrt(np.diag(np.linalg.pinv(H))),p
def design(keys,tr_idx,all_idx):
    """standardize + residualize-on-depth using ONLY tr_idx statistics"""
    mu={k:RAW[k][tr_idx].mean() for k in RAW}; sd={k:RAW[k][tr_idx].std(ddof=0) for k in RAW}
    Zf={k:(RAW[k]-mu[k])/max(sd[k],1e-12) for k in RAW}
    Xd_tr=np.column_stack([np.ones(len(tr_idx)),Zf["dg"][tr_idx]])
    cols=[np.ones(len(all_idx))]
    for k in keys:
        base=k[:-2] if k.endswith("_r") else k
        v=Zf[base]
        if k.endswith("_r"):
            c=np.linalg.lstsq(Xd_tr,v[tr_idx],rcond=None)[0]
            Xd_all=np.column_stack([np.ones(len(all_idx)),Zf["dg"][all_idx]])
            v=(v[all_idx]-Xd_all@c); s=(Zf[base][tr_idx]-Xd_tr@c).std(ddof=0); v=v/max(s,1e-12)
        else: v=v[all_idx]
        cols.append(v)
    return np.column_stack(cols)
SETS={"A0 null":[], "A1 depth":["dg"], "B1 turn":["cum_t"],
      "A2b depth+turn_r+k3sp_r":["dg","cum_t_r","k3_sp_r"],
      "A2 depth+4geo_r":["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"],
      "C1 turn+clr_sp":["cum_t","logd_sp"]}
allidx=np.arange(n)
def auc(y,p):
    pos=y==1; neg=~pos; 
    if p.std()<1e-12: return 0.5
    r=stats.rankdata(p); return (r[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())
print("[LOAO leak-free] out-of-anatomy log-loss / AUC / Brier")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in SETS.items():
        pv=np.zeros(n)
        for a in range(K):
            te=np.where(aid==a)[0]; tr=np.where(aid!=a)[0]
            Xtr=design(keys,tr,tr); Xte=design(keys,tr,te)
            b,_,_,_=logfit(Xtr,y[tr],ridge=1e-3); pv[te]=1/(1+np.exp(-(Xte@b)))
        ll=-(y*np.log(np.clip(pv,1e-9,1))+(1-y)*np.log(np.clip(1-pv,1e-9,1))).mean()
        print("  %-9s %-24s LOAO_ll=%.4f AUC=%.3f Brier=%.4f"%(lab,nm,ll,auc(y,pv),((y-pv)**2).mean()))

print("\n[FINAL COEFS on full data, leak-free-equivalent]")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in (("A1 depth",["dg"]),("A2b",["dg","cum_t_r","k3_sp_r"]),("A2",["dg","k3_sp_r","logd_sp_r","r_sp_r","cum_t_r"])):
        X=design(keys,allidx,allidx); b,ll,se,_=logfit(X,y)
        print("  [%s|%s] logLik=%.3f"%(lab,nm,ll))
        for j,nmj in enumerate(["intercept"]+keys):
            print("     %-11s b=%7.3f se=%6.3f z=%6.2f p=%.4f"%(nmj,b[j],se[j],b[j]/se[j],2*stats.norm.sf(abs(b[j]/se[j]))))

# per-anatomy residuals under A1 and A2b
print("\n[PER-ANATOMY RESIDUAL] obs successes vs model-expected (A1 depth-only and A2b), z=(O-E)/sqrt(sum p(1-p))")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    ps={}
    for nm,keys in (("A1",["dg"]),("A2b",["dg","cum_t_r","k3_sp_r"])):
        X=design(keys,allidx,allidx); b,_,_,p=logfit(X,y); ps[nm]=p
    print(" %s: anatomy   n  O   E_A1  z_A1   E_A2b  z_A2b"%lab)
    for a in ANAT:
        m=aid==ANAT.index(a); O=y[m].sum(); ln=m.sum()
        out=[]
        for nm in ("A1","A2b"):
            E=ps[nm][m].sum(); V=(ps[nm][m]*(1-ps[nm][m])).sum(); out.append((E,(O-E)/math.sqrt(max(V,1e-9))))
        print("   %-12s %2d %2d  %5.2f %6.2f  %5.2f %6.2f"%(a,ln,O,out[0][0],out[0][1],out[1][0],out[1][1]))

# bootstrap sensitivity: null p from LOAO (not in-sample) for the anatomy-dummy LRT
D=np.zeros((n,K-1))
for i,a in enumerate(aid):
    if a>0: D[i,a-1]=1
def anat_lr(y,keys):
    X0=design(keys,allidx,allidx); _,ll0,_,p0=logfit(X0,y)
    X1=np.column_stack([X0,D]); _,ll1,_,_=logfit(X1,y,ridge=1e-4)
    return 2*(ll1-ll0)
def loao_p(y,keys):
    pv=np.zeros(n)
    for a in range(K):
        te=np.where(aid==a)[0]; tr=np.where(aid!=a)[0]
        b,_,_,_=logfit(design(keys,tr,tr),y[tr],ridge=1e-3); pv[te]=1/(1+np.exp(-(design(keys,tr,te)@b)))
    return pv
print("\n[ANATOMY-DUMMY LRT: bootstrap null from IN-SAMPLE p vs LOAO p]")
rng=np.random.default_rng(21); B=1500
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    for nm,keys in (("uncond",[]),("|depth",["dg"]),("|A2b",["dg","cum_t_r","k3_sp_r"])):
        obs=anat_lr(y,keys)
        _,_,_,pin=logfit(design(keys,allidx,allidx),y)
        pcv=loao_p(y,keys)
        res={}
        for tag,pp in (("insample",pin),("loao",pcv)):
            c=0
            for _ in range(B):
                ys=(rng.random(n)<pp).astype(float)
                if anat_lr(ys,keys)>=obs: c+=1
            res[tag]=(c+1)/(B+1)
        print("  %-9s %-9s LR=%7.3f p_asym=%.4f p_boot_insample=%.4f p_boot_LOAO=%.4f"%(
            lab,nm,obs,stats.chi2.sf(max(obs,0),K-1),res["insample"],res["loao"]))

# policy x anatomy interaction and T-H correlation given depth
print("\n[T vs H dependence]")
print("  raw phi=%.4f"%np.corrcoef(yT,yH)[0,1])
Xd=design(["dg"],allidx,allidx)
_,_,_,pT=logfit(Xd,yT); _,_,_,pH=logfit(Xd,yH)
rT=yT-pT; rH=yH-pH
print("  partial corr given depth (Pearson resid) = %.4f"%np.corrcoef(rT,rH)[0,1])
Xg=design(["dg","cum_t_r","k3_sp_r"],allidx,allidx)
_,_,_,qT=logfit(Xg,yT); _,_,_,qH=logfit(Xg,yH)
print("  partial corr given depth+geo           = %.4f"%np.corrcoef(yT-qT,yH-qH)[0,1])
print("  corr of fitted probs pT,pH (A2b)       = %.4f"%np.corrcoef(qT,qH)[0,1])
# expected agreement if independent
pTm=yT.mean(); pHm=yH.mean()
print("  both-succeed obs=%d exp_indep=%.1f | both-fail obs=%d exp_indep=%.1f"%(
    int(((yT==1)&(yH==1)).sum()),n*pTm*pHm,int(((yT==0)&(yH==0)).sum()),n*(1-pTm)*(1-pHm)))
