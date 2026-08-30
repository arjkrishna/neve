import json, math
import numpy as np
from scipy import stats
E=json.load(open("/opt/mon/arj_t3_merged.json"))
R=json.load(open("/opt/mon/arj_t3_rows.json"))
OFF=33.31
print("[FULL 220: success vs s_RCCA in 10 mm bins]  seam s=133.6")
allE=sorted(E,key=lambda e:e["pl"])
sv=np.array([e["pl"]-OFF for e in allE]); T=np.array([e["T"] for e in allE],float); H=np.array([e["H"] for e in allE],float)
edges=np.arange(40,270,10)
print("  s-bin      n  teacher   heur")
for i in range(len(edges)-1):
    m=(sv>=edges[i])&(sv<edges[i+1])
    if m.sum()==0: continue
    print("  [%3d,%3d) %3d  %5.3f    %5.3f %s"%(edges[i],edges[i+1],m.sum(),T[m].mean(),H[m].mean(),
          "<== SEAM" if edges[i]<=133.6<edges[i+1] else ""))
print("\n[GRAFTED 124: dg bins of 15 mm]")
dg=np.array([r["dg"] for r in R]); yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
cum=np.array([r["cum_t"] for r in R])
for lo in range(0,120,15):
    m=(dg>=lo)&(dg<lo+15)
    if m.sum()==0: continue
    print("  dg[%3d,%3d) n=%2d  teacher %2d/%2d=%.3f  heur %2d/%2d=%.3f  mean cum_t=%5.0f"%(
        lo,lo+15,m.sum(),int(yT[m].sum()),m.sum(),yT[m].mean(),int(yH[m].sum()),m.sum(),yH[m].mean(),cum[m].mean()))
def logfit(X,y,ridge=1e-6):
    b=np.zeros(X.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H_=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); st=np.linalg.solve(H_,X.T@(y-p)-ridge*b); b=b+st
        if np.max(np.abs(st))<1e-11: break
    p=1/(1+np.exp(-(X@b)))
    ll=float((y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum())
    W=np.maximum(p*(1-p),1e-9); H_=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1])
    return b,np.sqrt(np.diag(np.linalg.pinv(H_))),ll
print("\n[QUADRATIC DEPTH] logistic success ~ dg + dg^2 (dg in units of 10 mm)")
d=dg/10.0
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    b1,s1,l1=logfit(np.column_stack([np.ones(len(d)),d]),y)
    b2,s2,l2=logfit(np.column_stack([np.ones(len(d)),d,d*d]),y)
    lr=2*(l2-l1)
    print("  %-9s lin ll=%.3f ; quad ll=%.3f LR=%.3f p=%.4f  b_dg=%.3f(%.3f) b_dg2=%.4f(%.4f)  peak at dg=%.1f mm"%(
        lab,l1,l2,lr,stats.chi2.sf(max(lr,0),1),b2[1],s2[1],b2[2],s2[2], (-b2[1]/(2*b2[2])*10 if b2[2]!=0 else float('nan'))))
    for cut in (20,40,60,80,100):
        m=dg<cut
        print("        dg<%3d: teacher/heur = %.3f / %.3f  (n=%d)"%(cut,y[m].mean() if lab=="TEACHER" else y[m].mean(), (yH if lab=="TEACHER" else yT)[m].mean(),m.sum()))
        break
print("\n[CUM TURNING quadratic]")
c=cum/100.0
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    b1,s1,l1=logfit(np.column_stack([np.ones(len(c)),c]),y)
    b2,s2,l2=logfit(np.column_stack([np.ones(len(c)),c,c*c]),y)
    lr=2*(l2-l1)
    print("  %-9s lin ll=%.3f quad ll=%.3f LR=%.3f p=%.4f b1=%.3f(%.3f) b2=%.4f(%.4f) peak cum=%.0f deg"%(
        lab,l1,l2,lr,stats.chi2.sf(max(lr,0),1),b2[1],s2[1],b2[2],s2[2],(-b2[1]/(2*b2[2])*100 if b2[2] else float('nan'))))
print("\n[HARD CUTPOINT SCAN] teacher success above/below dg cut")
best=None
for cut in range(10,110,5):
    a=yT[dg<cut]; b=yT[dg>=cut]
    if len(a)<8 or len(b)<8: continue
    tab=[[int(a.sum()),len(a)-int(a.sum())],[int(b.sum()),len(b)-int(b.sum())]]
    p=stats.fisher_exact(tab)[1]
    print("  cut dg=%3d: below %2d/%2d=%.3f  above %2d/%2d=%.3f  fisher p=%.2e | heur below=%.3f above=%.3f"%(
        cut,int(a.sum()),len(a),a.mean(),int(b.sum()),len(b),b.mean(),p,yH[dg<cut].mean(),yH[dg>=cut].mean()))
