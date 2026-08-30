import pickle, numpy as np, collections, math
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]; fai=[r for r in gt if not r["succ"]]
def logit_fit(X,y,iters=200):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-X@b)); W=np.clip(p*(1-p),1e-9,None)
        z=X@b+(y-p)/W
        b=np.linalg.solve(X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1]), X.T@(W*z))
    p=1/(1+np.exp(-X@b))
    return b,float(np.sum(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))))
def chi2sf(x,k):
    a=k/2.0; xx=x/2.0
    if xx<=0: return 1.0
    if xx<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(20000):
            ap+=1; d*=xx/ap; s+=d
            if abs(d)<abs(s)*1e-15: break
        return 1.0-s*math.exp(-xx+a*math.log(xx)-math.lgamma(a))
    b=xx+1-a; c=1e300; d=1/b; h=d
    for i in range(1,20000):
        an=-i*(i-a); b+=2
        d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
        c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1/d; de=d*c; h*=de
        if abs(de-1)<1e-15: break
    return h*math.exp(-xx+a*math.log(xx)-math.lgamma(a))
anats=sorted(set(r["anat"] for r in gt)); ai={a:i for i,a in enumerate(anats)}
y=[r["succ"] for r in gt]
u=lambda r:(r["tgt_s"]-183.3)/10.0
Xq=[[1.0,u(r),u(r)**2] for r in gt]
Xqa=[row+[1.0 if ai[r["anat"]]==j else 0.0 for j in range(1,len(anats))] for row,r in zip(Xq,gt)]
_,lq=logit_fit(Xq,y); _,lqa=logit_fit(Xqa,y)
print("anatomy(21df) on top of QUADRATIC target depth: dev drop %.2f p=%.4f"%(2*(lqa-lq),chi2sf(2*(lqa-lq),21)))
# binned depth (4 bins) then anatomy
bins=[133.6,160,190,215,300]
def bd(r):
    for i in range(4):
        if bins[i]<=r["tgt_s"]<bins[i+1]: return i
    return 3
Xb=[[1.0]+[1.0 if bd(r)==j else 0.0 for j in (1,2,3)] for r in gt]
Xba=[row+[1.0 if ai[r["anat"]]==j else 0.0 for j in range(1,len(anats))] for row,r in zip(Xb,gt)]
_,lb=logit_fit(Xb,y); _,lba=logit_fit(Xba,y)
print("anatomy(21df) on top of 4-bin target depth  : dev drop %.2f p=%.4f"%(2*(lba-lb),chi2sf(2*(lba-lb),21)))
print()
print("=== DEEP-TARGET CLIFF ===")
d=[r for r in gt if r["tgt_s"]>215]
print("tgt_s>215: teacher %d/%d = %.1f%% ; heuristic %d/%d = %.1f%%"%(
  sum(r['succ'] for r in d),len(d),100*np.mean([r['succ'] for r in d]),
  sum(mh[r['seed']]['succ'] for r in d),len(d),100*np.mean([mh[r['seed']]['succ'] for r in d])))
s=[r for r in gt if r["tgt_s"]<=215]
print("tgt_s<=215: teacher %d/%d = %.1f%% ; heuristic %d/%d = %.1f%%"%(
  sum(r['succ'] for r in s),len(s),100*np.mean([r['succ'] for r in s]),
  sum(mh[r['seed']]['succ'] for r in s),len(s),100*np.mean([mh[r['seed']]['succ'] for r in s])))
print("deep-target failures (n=%d) arrest hist 10mm:"%len([r for r in d if not r['succ']]),
  sorted(collections.Counter(int(r['max_s']//10)*10 for r in d if not r['succ']).items()))
print("  their fold_max med %d ; slack_max med %.1f ; tailv med %.2f ; shortfall med %.1f"%(
  np.median([r['fold_max'] for r in d if not r['succ']]),np.median([r['slack_max'] for r in d if not r['succ']]),
  np.median([r['tailv'] for r in d if not r['succ']]),np.median([r['shortfall'] for r in d if not r['succ']])))
print()
print("=== OSTIUM-FREEZE (arrest s<25) target depths ===")
o=[r for r in fai if r["max_s"]<25]
print(" n=%d ; tgt_s med %.1f range %.1f-%.1f ; %d/%d have tgt_s>200"%(
  len(o),np.median([r['tgt_s'] for r in o]),min(r['tgt_s'] for r in o),max(r['tgt_s'] for r in o),
  sum(1 for r in o if r['tgt_s']>200),len(o)))
print(" base rate of tgt_s>200 among all grafted: %d/%d"%(sum(1 for r in gt if r['tgt_s']>200),len(gt)))
print(" mean |d_ins| in last 300 steps: %.2f mm/step (successes %.2f); max commanded 3.96"%(
  np.median([r['tailv'] for r in o]),np.median([r['tailv'] for r in gt if r['succ']])))
print(" fold_max med %d ; slack_max med %.1f -> no buckling"%(np.median([r['fold_max'] for r in o]),np.median([r['slack_max'] for r in o])))
print(" heuristic on same 17 seeds: succ %d, med max_s %.1f"%(sum(mh[r['seed']]['succ'] for r in o),np.median([mh[r['seed']]['max_s'] for r in o])))
print()
print("=== arrest depth vs target depth (failures) ===")
x=[r["tgt_s"] for r in fai]; z=[r["max_s"] for r in fai]
print(" Pearson r = %+.3f (n=55)"%np.corrcoef(x,z)[0,1])
for lo,hi in ((133.6,160),(160,190),(190,215),(215,300)):
    g=[r for r in fai if lo<=r["tgt_s"]<hi]
    if g: print("  tgt %5.0f-%3.0f  nfail %2d  arrest med %6.1f  frac arrest<25mm %.2f  frac fold>=20 %.2f"%(
      lo,hi,len(g),np.median([r['max_s'] for r in g]),np.mean([r['max_s']<25 for r in g]),np.mean([r['fold_max']>=20 for r in g])))
