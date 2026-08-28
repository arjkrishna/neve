import json,itertools,math
import numpy as np
rng=np.random.default_rng(0)
G=json.load(open("monitoring/attack2_geom_m4.json"))
S=json.load(open("monitoring/attack2_siphonband.json"))
HOLD=["topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"]
rows={x["a"]:x for x in G["rows"]}
names=sorted(rows)
print("=== 1. SELECTION-BIAS TEST: exact permutation over all C(22,4)=%d subsets ==="%math.comb(22,4))
def test(vals,name,easier="high"):
    v={n:vals[n] for n in names}
    obs=np.mean([v[n] for n in HOLD])
    allsub=[np.mean([v[n] for n in c]) for c in itertools.combinations(names,4)]
    allsub=np.array(allsub)
    p = (allsub>=obs).mean() if easier=="high" else (allsub<=obs).mean()
    rank=sorted(names,key=lambda n:-v[n] if easier=="high" else v[n])
    pos=[rank.index(n)+1 for n in HOLD]
    print("  %-28s holdout mean=%7.3f cohort mean=%7.3f  p(4 random >= this benign)=%.5f  ranks(1=most benign)=%s"%(
        name,obs,np.mean(list(v.values())),p,sorted(pos)))
    return p
test({n:rows[n]["Rcmin"] for n in names},"graft Rc_min (mm)")
test({n:-rows[n]["kp95"] for n in names},"graft -kappa_p95")
test({n:-rows[n]["bendmax"] for n in names},"graft -max bend (8mm)")
test({n:S[n]["Rcmin"] for n in names},"SIPHON-BAND Rc_min (mm)")
test({n:-S[n]["bendmax"] for n in names},"SIPHON-BAND -max bend")
test({n:-rows[n]["D"] for n in names},"composite -D (difficulty)")
print()
print("=== 2. DOES GEOMETRY PREDICT THE OBSERVED 4-ANATOMY RANKING? ===")
obs={"topcow_mr_008":(26,26),"topcow_mr_017":(28,29),"topcow_mr_004":(24,27),"topcow_mr_023":(12,16)}
D=np.array([rows[n]["D"] for n in HOLD]); k=np.array([obs[n][0] for n in HOLD],float); N=np.array([obs[n][1] for n in HOLD],float)
r=k/N
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return np.corrcoef(ra,rb)[0,1]
rho=spearman(-D,r)   # -D = benign-ness; positive rho = geometry predicts
print("  Spearman(benign-ness, success rate) over the 4 = %+.3f  (n=4; |rho|=1 needs p=0.042, rho=0.4 -> p=0.75 two-sided)"%rho)
print("  observed order hardest->easiest: 023(.750) 004(.889) 017(.966) 008(1.000)")
print("  geometric order hardest->easiest: 023(D=%.2f) 008(%.2f) 004(%.2f) 017(%.2f)"%tuple(D[[3,1,0,2]]))
# homogeneity of the 4 rates
p0=k.sum()/N.sum()
G2=2*sum(ki*np.log(ki/(Ni*p0)) if ki>0 else 0 for ki,Ni in zip(k,N)) + \
   2*sum((Ni-ki)*np.log((Ni-ki)/(Ni*(1-p0))) if Ni-ki>0 else 0 for ki,Ni in zip(k,N))
from math import erfc
print("  G-test of homogeneity across the 4 anatomies: G=%.2f df=3 -> p~%.3f (rates are NOT homogeneous)"%(G2,
   1-__import__('scipy.stats',fromlist=['chi2']).chi2.cdf(G2,3) if False else 0.0))
print()
print("=== 3. SIPHON SECTION alone ===")
sip={"topcow_mr_008":(5,5),"topcow_mr_017":(11,11),"topcow_mr_004":(6,6),"topcow_mr_023":(0,4)}
print("  m008 5/5  m017 11/11  m004 6/6  m023 0/4 -> pooled 22/26=84.6%")
print("  Fisher exact, m023 vs the other three (0/4 vs 22/22): p = 1/C(26,4) = %.2e"%(1/math.comb(26,4)))
print()
print("=== 4. ANATOMY-LEVEL EXTRAPOLATION ===")
Dall=np.array([rows[n]["D"] for n in names]); Dh=np.array([rows[n]["D"] for n in HOLD])
Do=np.array([rows[n]["D"] for n in names if n not in HOLD])
print("  composite D: holdout mean=%.2f (sd %.2f)  other18 mean=%.2f (sd %.2f)  gap=%.2f z-units"%(
  Dh.mean(),Dh.std(ddof=1),Do.mean(),Do.std(ddof=1),Do.mean()-Dh.mean()))
def fitlogit(D,k,N,ridge=1e-3):
    X=np.column_stack([np.ones(len(D)),D]); b=np.zeros(2)
    for _ in range(300):
        p=1/(1+np.exp(-X@b)); W=N*np.maximum(p*(1-p),1e-9)
        g=X.T@(k-N*p)-ridge*b; H=X.T@(X*W[:,None])+ridge*np.eye(2)
        b=b+np.linalg.solve(H,g)
    return b
b=fitlogit(Dh,k,N)
print("  logistic fit on the 4: logit(p) = %.3f + %.3f*D   (slope<0 => harder D lowers success)"%(b[0],b[1]))
pred18=1/(1+np.exp(-(b[0]+b[1]*Do)))
w18=np.ones(18); w4=np.array([N[i] for i in range(4)])
print("  point prediction other18 = %.3f ; all22 = %.3f"%(pred18.mean(),(pred18.sum()+r.sum())/22))
# bootstrap: resample anatomies + binomial noise
bs=[]
for _ in range(20000):
    idx=rng.integers(0,4,4)
    kk=rng.binomial(N[idx],np.clip(k[idx]/N[idx],1e-6,1-1e-6)).astype(float)
    try: bb=fitlogit(Dh[idx],kk,N[idx],ridge=0.5)
    except Exception: continue
    pp=1/(1+np.exp(-(bb[0]+bb[1]*Do)))
    bs.append((pp.mean()*18+kk.sum()/N[idx].sum()*4)/22)
bs=np.array(bs)
print("  bootstrap all-22 (anatomy+binomial, ridge-stabilised): median %.3f  80%%CI [%.3f,%.3f]  95%%CI [%.3f,%.3f]"%(
  np.median(bs),np.percentile(bs,10),np.percentile(bs,90),np.percentile(bs,2.5),np.percentile(bs,97.5)))
