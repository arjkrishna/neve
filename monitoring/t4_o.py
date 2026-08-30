import pickle, numpy as np, collections, math, random
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
print("anat      n  rate   tgt_s: min  med  max   n>215  rate|<=215  rate|>215   Hrate")
for a in sorted(by):
    g=by[a]; ts=[r["tgt_s"] for r in g]
    d=[r for r in g if r["tgt_s"]>215]; s=[r for r in g if r["tgt_s"]<=215]
    f1=f"{sum(r['succ'] for r in s)}/{len(s)}" if s else "-"
    f2=f"{sum(r['succ'] for r in d)}/{len(d)}" if d else "-"
    print(f"{a} {len(g):3d} {100*np.mean([r['succ'] for r in g]):5.1f}  {min(ts):5.1f} {np.median(ts):5.1f} {max(ts):5.1f}  {len(d):3d}  {f1:>7s} {f2:>7s}   {100*np.mean([mh[r['seed']]['succ'] for r in g]):5.1f}")
print()
# logistic fit: success ~ 1 + tgt_s  (episode level, n=124), IRLS
def logit_fit(X,y,iters=60):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-X@b)); W=np.clip(p*(1-p),1e-9,None)
        z=X@b+(y-p)/W
        b=np.linalg.solve(X.T@(X*W[:,None])+1e-9*np.eye(X.shape[1]), X.T@(W*z))
    p=1/(1+np.exp(-X@b))
    ll=float(np.sum(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))))
    return b,ll,p
y=[r["succ"] for r in gt]
X0=[[1.0] for r in gt]
X1=[[1.0,(r["tgt_s"]-183.3)/10.0] for r in gt]
X2=[[1.0,(r["tgt_s"]-183.3)/10.0,((r["tgt_s"]-183.3)/10.0)**2] for r in gt]
b0,ll0,_=logit_fit(X0,y); b1,ll1,_=logit_fit(X1,y); b2,ll2,_=logit_fit(X2,y)
def chi2sf(x,k):
    import math as m
    a=k/2.0; xx=x/2.0
    if xx<=0: return 1.0
    if xx<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(10000):
            ap+=1; d*=xx/ap; s+=d
            if abs(d)<abs(s)*1e-15: break
        return 1.0-s*m.exp(-xx+a*m.log(xx)-m.lgamma(a))
    b=xx+1-a; c=1e300; d=1/b; h=d
    for i in range(1,10000):
        an=-i*(i-a); b+=2
        d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
        c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1/d; de=d*c; h*=de
        if abs(de-1)<1e-15: break
    return h*m.exp(-xx+a*m.log(xx)-m.lgamma(a))
print("EPISODE-LEVEL logistic, teacher grafted n=124")
print("  null LL %.2f"%ll0)
print("  +tgt_s        LL %.2f  dev drop %.2f df1 p=%.2e  beta/10mm %+.3f (OR %.3f)"%(ll1,2*(ll1-ll0),chi2sf(2*(ll1-ll0),1),b1[1],math.exp(b1[1])))
print("  +tgt_s+tgt_s^2 LL %.2f  dev drop vs lin %.2f df1 p=%.4f"%(ll2,2*(ll2-ll1),chi2sf(2*(ll2-ll1),1)))
# add anatomy factor after depth
anats=sorted(by); ai={a:i for i,a in enumerate(anats)}
def dummies(rows,base):
    return [base[i]+[1.0 if ai[r["anat"]]==j else 0.0 for j in range(1,len(anats))] for i,r in enumerate(rows)]
XA=dummies(gt,[[x for x in row] for row in X1])
bA,llA,_=logit_fit(XA,y,200)
print("  +anatomy(21 df) on top of tgt_s: LL %.2f dev drop %.2f df21 p=%.4f"%(llA,2*(llA-ll1),chi2sf(2*(llA-ll1),21)))
XA0=dummies(gt,[[1.0] for r in gt]); bA0,llA0,_=logit_fit(XA0,y,200)
print("  anatomy alone (21 df) vs null: dev drop %.2f p=%.4f"%(2*(llA0-ll0),chi2sf(2*(llA0-ll0),21)))
# heuristic same
yh=[mh[r["seed"]]["succ"] for r in gt]
_,h0,_=logit_fit(X0,yh); bh,h1,_=logit_fit(X1,yh)
_,hA,_=logit_fit(dummies(gt,[[x for x in row] for row in X1]),yh,200)
print("HEURISTIC: +tgt_s dev drop %.2f p=%.4f beta/10mm %+.3f ; +anatomy dev drop %.2f p=%.4f"%(
  2*(h1-h0),chi2sf(2*(h1-h0),1),bh[1],2*(hA-h1),chi2sf(2*(hA-h1),21)))
# predicted curve
print("\npredicted teacher success vs target depth (linear-logit model):")
for s in (140,160,180,200,215,230,250):
    p=1/(1+math.exp(-(b1[0]+b1[1]*(s-183.3)/10)))
    print("   tgt_s %3d -> %.3f"%(s,p))
