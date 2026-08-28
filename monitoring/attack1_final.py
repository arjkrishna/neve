"""ATTACK 1 final: stratified / standardised comparison v1bp vs H0, and OOD depth profile."""
import json, math, os
from collections import Counter
import numpy as np
ROOT=r"d:/Arjun/workspace/neve"
V=json.load(open(os.path.join(ROOT,"monitoring","_attack1_rows_20260828_045651.json")))
H=json.load(open(os.path.join(ROOT,"monitoring","_attack1_h0rows.json")))
SC=133.6; OFF=33.314

def wil(k,n,z=1.959964):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def rate(ks):
    n=len(ks); k=sum(r["succ"] for r in ks); lo,hi=wil(k,n)
    return k,n,(100*k/n if n else float('nan')),lo,hi
def P(tag,ks):
    k,n,p,lo,hi=rate(ks); print("%-56s %3d/%-3d = %5.1f%%  [%4.1f,%5.1f]"%(tag,k,n,p,lo,hi))

def fisher(a,b,c,d):
    from math import comb
    n=a+b+c+d
    def pr(a_):
        b_=a+b-a_; c_=a+c-a_; d_=d-(a_-a)
        if min(b_,c_,d_)<0: return 0.0
        return comb(a+b,a_)*comb(c+d,c_)/comb(n,a+c)
    p0=pr(a); lo=max(0,a+c-(c+d)); hi=min(a+b,a+c)
    return sum(pr(x) for x in range(lo,hi+1) if pr(x)<=p0+1e-12)

print("="*96); print("A. DEPTH-MIX OF THE TWO RUNS (they are NOT the same task draw)")
print("="*96)
for nm,R in (("v1bp",V),("H0  ",H)):
    ts=np.array([r["tgt_s"] for r in R])
    print("%s n=%d  anatomy mix %s"%(nm,len(R),dict(sorted(Counter(r["fp"] for r in R).items()))))
    print("      target s_RCCA: median %.1f  frac proximal to seam %.3f  frac in siphon(s>178) %.3f"
          %(np.median(ts),(ts<=SC).mean(),(ts>178).mean()))

print(); print("="*96); print("B. STRATIFIED BY SEAM")
print("="*96)
for nm,R in (("v1bp",V),("H0",H)):
    P("%s  ALL"%nm,R)
    P("%s  target <=133.6  (host-identical course)"%nm,[r for r in R if r["tgt_s"]<=SC])
    P("%s  target > 133.6  (grafted / OOD)"%nm,[r for r in R if r["tgt_s"]>SC])
a=sum(r["succ"] for r in V if r["tgt_s"]>SC); b=sum(1 for r in V if r["tgt_s"]>SC)-a
c=sum(r["succ"] for r in H if r["tgt_s"]>SC); d=sum(1 for r in H if r["tgt_s"]>SC)-c
print("  Fisher exact, OOD-only v1bp(%d/%d) vs H0(%d/%d): p=%.4f"%(a,a+b,c,c+d,fisher(a,b,c,d)))
a2=sum(r["succ"] for r in V); b2=len(V)-a2; c2=sum(r["succ"] for r in H); d2=len(H)-c2
print("  Fisher exact, ALL      v1bp(%d/%d) vs H0(%d/%d): p=%.4f"%(a2,a2+b2,c2,c2+d2,fisher(a2,b2,c2,d2)))

print(); print("="*96); print("C. DIRECT STANDARDISATION (remove the anatomy-mix confound)")
print("="*96)
anas=sorted({r["fp"] for r in V}|{r["fp"] for r in H})
wV={f:sum(1 for r in V if r["fp"]==f)/len(V) for f in anas}
wH={f:sum(1 for r in H if r["fp"]==f)/len(H) for f in anas}
def std(R,w):
    tot=0.0
    for f in anas:
        kk=[r for r in R if r["fp"]==f]
        if not kk: return float('nan')
        tot+=w[f]*sum(r["succ"] for r in kk)/len(kk)
    return 100*tot
print("  per-anatomy rates:")
for f in anas:
    kv=[r for r in V if r["fp"]==f]; kh=[r for r in H if r["fp"]==f]
    print("    %-12s v1bp %2d/%-2d=%5.1f%%   H0 %2d/%-2d=%5.1f%%   (weights v1bp %.3f / H0 %.3f)"
          %(f,sum(r["succ"] for r in kv),len(kv),100*np.mean([r["succ"] for r in kv]),
            sum(r["succ"] for r in kh),len(kh),100*np.mean([r["succ"] for r in kh]),wV[f],wH[f]))
print("  crude:                v1bp %.1f%%   H0 %.1f%%   (gap %.1f pp)"
      %(100*np.mean([r["succ"] for r in V]),100*np.mean([r["succ"] for r in H]),
        100*(np.mean([r["succ"] for r in V])-np.mean([r["succ"] for r in H]))))
print("  standardised to v1bp anatomy mix: v1bp %.1f%%  H0 %.1f%%  (gap %.1f pp)"
      %(std(V,wV),std(H,wV),std(V,wV)-std(H,wV)))
print("  standardised to H0   anatomy mix: v1bp %.1f%%  H0 %.1f%%  (gap %.1f pp)"
      %(std(V,wH),std(H,wH),std(V,wH)-std(H,wH)))
# also standardise on (anatomy x seam) strata
def std2(R,W):
    tot=0.0; miss=0
    for key,w in W.items():
        kk=[r for r in R if (r["fp"],r["tgt_s"]>SC)==key]
        if not kk: miss+=w; continue
        tot+=w*np.mean([r["succ"] for r in kk])
    return 100*tot/(1-miss) if miss<1 else float('nan')
WV=Counter((r["fp"],r["tgt_s"]>SC) for r in V); WV={k:v/len(V) for k,v in WV.items()}
print("  standardised to v1bp (anatomy x seam) strata: v1bp %.1f%%  H0 %.1f%%"%(std2(V,WV),std2(H,WV)))

print(); print("="*96); print("D. OOD DEPTH PROFILE (v1bp): how far past the seam did the 55 OOD targets sit?")
print("="*96)
ood=[r for r in V if r["tgt_s"]>SC]
dd=np.array([r["tgt_s"]-SC for r in ood])
print("  mm of grafted anatomy the target required: min %.1f  q25 %.1f  median %.1f  q75 %.1f  max %.1f"
      %(dd.min(),np.percentile(dd,25),np.median(dd),np.percentile(dd,75),dd.max()))
for lo,hi in ((0,10),(10,25),(25,50),(50,100)):
    kk=[r for r in ood if lo<=r["tgt_s"]-SC<hi]
    if kk: P("  target %2d-%3d mm past the seam"%(lo,hi),kk)
print()
print("  fraction of each OOD episode's RCCA route that was grafted:")
fr=np.array([(r["tgt_s"]-SC)/r["tgt_s"] for r in ood])
print("    min %.3f median %.3f max %.3f"%(fr.min(),np.median(fr),fr.max()))
print("  over ALL 98 episodes, grafted share of the target route: %.3f"
      %float(np.mean([max(0.0,r["tgt_s"]-SC)/r["tgt_s"] for r in V])))
