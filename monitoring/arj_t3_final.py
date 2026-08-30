import json, math, collections
import numpy as np
from scipy import stats
R=json.load(open("/opt/mon/arj_t3_rows.json"))
n=len(R); ANAT=sorted({r["anat"] for r in R}); aid=np.array([ANAT.index(r["anat"]) for r in R])
yT=np.array([r["T"] for r in R],float); yH=np.array([r["H"] for r in R],float)
dg=np.array([r["dg"] for r in R]); cum=np.array([r["cum_t"] for r in R]); s=np.array([r["s"] for r in R])
def logfit(X,y,ridge=1e-6):
    b=np.zeros(X.shape[1])
    for _ in range(400):
        p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9)
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); st=np.linalg.solve(H,X.T@(y-p)-ridge*b); b=b+st
        if np.max(np.abs(st))<1e-11: break
    p=1/(1+np.exp(-(X@b))); W=np.maximum(p*(1-p),1e-9); H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1])
    return b,np.sqrt(np.diag(np.linalg.pinv(H)))
print("[NATURAL UNITS]")
for lab,y in (("TEACHER",yT),("HEURISTIC",yH)):
    b,se=logfit(np.column_stack([np.ones(n),dg/10.0]),y)
    print("  %-9s depth-into-graft: b=%.4f per 10mm (se %.4f) OR=%.3f/10mm  p=%.4f"%(lab,b[1],se[1],math.exp(b[1]),2*stats.norm.sf(abs(b[1]/se[1]))))
    b2,se2=logfit(np.column_stack([np.ones(n),cum/100.0]),y)
    print("  %-9s cum turning seam->target: b=%.4f per 100deg (se %.4f) OR=%.3f  p=%.4f"%(lab,b2[1],se2[1],math.exp(b2[1]),2*stats.norm.sf(abs(b2[1]/se2[1]))))
print("\n[DEPTH QUARTILES of dg]  (grafted n=124)")
q=np.quantile(dg,[0,.25,.5,.75,1.0])
print("  quartile edges dg:", np.round(q,1), " cum_t range %.0f-%.0f deg"%(cum.min(),cum.max()))
for i in range(4):
    m=(dg>=q[i])&(dg<=q[i+1]) if i==3 else (dg>=q[i])&(dg<q[i+1])
    print("  Q%d dg[%6.1f,%6.1f) n=%2d  teacher %2d/%2d=%.3f  heur %2d/%2d=%.3f"%(
        i+1,q[i],q[i+1],m.sum(),int(yT[m].sum()),m.sum(),yT[m].mean(),int(yH[m].sum()),m.sum(),yH[m].mean()))
print("\n[CUM TURNING TERTILES]")
qc=np.quantile(cum,[0,1/3,2/3,1.0])
for i in range(3):
    m=(cum>=qc[i])&(cum<=qc[i+1]) if i==2 else (cum>=qc[i])&(cum<qc[i+1])
    print("  T%d cum[%5.0f,%5.0f) n=%2d teacher=%.3f heur=%.3f"%(i+1,qc[i],qc[i+1],m.sum(),yT[m].mean(),yH[m].mean()))
print("\n[nongrafted sanity] full 220-episode split")
E=json.load(open("/opt/mon/arj_t3_merged.json"))
ng=[e for e in E if e["pl"]<=166.91]
print("  non-grafted n=%d teacher=%d (%.3f) heur=%d (%.3f)"%(len(ng),sum(e["T"] for e in ng),
      np.mean([e["T"] for e in ng]),sum(e["H"] for e in ng),np.mean([e["H"] for e in ng])))
bb=sum(1 for e in ng if e["T"] and e["H"]); nn=sum(1 for e in ng if not e["T"] and not e["H"])
to=sum(1 for e in ng if e["T"] and not e["H"]); ho=sum(1 for e in ng if not e["T"] and e["H"])
print("  non-grafted 2x2 BB=%d Tonly=%d Honly=%d NN=%d phi=%.3f"%(bb,to,ho,nn,
      np.corrcoef([e["T"] for e in ng],[e["H"] for e in ng])[0,1]))
print("\n[mr_025 grafted detail]")
for e in sorted([e for e in E if e["anat"]=="topcowmr025"],key=lambda x:x["pl"]):
    print("   pl=%7.1f s=%7.1f T=%d(%3d steps) H=%d(%3d steps) %s"%(e["pl"],e["pl"]-33.31,e["T"],e["Tsteps"],e["H"],e["Hsteps"],
          "GRAFTED" if e["pl"]>166.91 else "prox"))
print("\n[mr_024 grafted detail]")
for e in sorted([e for e in E if e["anat"]=="topcowmr024"],key=lambda x:x["pl"]):
    print("   pl=%7.1f s=%7.1f T=%d(%3d) H=%d(%3d) %s"%(e["pl"],e["pl"]-33.31,e["T"],e["Tsteps"],e["H"],e["Hsteps"],"GRAFTED" if e["pl"]>166.91 else "prox"))
# heuristic-only-succeeds population depth
disc_H=[i for i in range(n) if yH[i]==1 and yT[i]==0]
disc_T=[i for i in range(n) if yT[i]==1 and yH[i]==0]
print("\n[DISCORDANT POPULATIONS] mean dg / cum_t")
for tag,ix in (("teacher-only (n=%d)"%len(disc_T),disc_T),("heuristic-only (n=%d)"%len(disc_H),disc_H),
               ("both-succeed",[i for i in range(n) if yT[i] and yH[i]]),("both-fail",[i for i in range(n) if not yT[i] and not yH[i]])):
    ix=np.array(ix); print("  %-22s dg=%6.1f  cum_t=%6.0f  s=%6.1f"%(tag,dg[ix].mean(),cum[ix].mean(),s[ix].mean()))
