import json, math, collections
import numpy as np
M="monitoring/"
T=json.load(open(M+"h2diff_table.json")); PROF=json.load(open(M+"h2diff_prof.json"))
ROWS=json.load(open(M+"h2diff_eprows.json"))
A=[a for a in T if a!="HOST"]
# HOST clearance corrected: drop stations that fell OUTSIDE the host mesh (clr==0)
h=np.array(PROF["HOST"]["clr"]); nz=h[h>0]
print("HOST clearance: %d/%d graft stations lie OUTSIDE the host mesh (centerline/mesh mismatch, clr forced 0)."%(int((h<=0).sum()),len(h)))
print("  HOST clr excl. those: min=%.3f p05=%.3f p25=%.3f med=%.3f  n<0.35=%d n<0.30=%d n<0.18=%d"
      %(nz.min(),np.percentile(nz,5),np.percentile(nz,25),np.median(nz),(nz<0.35).sum(),(nz<0.30).sum(),(nz<0.18).sum()))
# host on the cohort composite scale
def zc(k,sign=1):
    v=np.array([T[a][k] for a in A],float); m,s=v.mean(),v.std()
    return sign*(T["HOST"][k]-m)/s, sign*(v-m)/s
def build(hostmode=True):
    parts={}
    for nm,ks in [("curv",[("Rc_p25",-1),("n_Rc_lt8",1),("bend_max",1)]),
                  ("turn",[("turn_cum",1),("turn_per_mm",1),("tort_graft",1)]),
                  ("cal",[("clr_p05",-1),("clr_p25",-1),("clr_n_lt_cath",1)]),
                  ("ext",[("graft_len",1)])]:
        hs=[];cs=[]
        for k,s in ks:
            a,b=zc(k,s); hs.append(a); cs.append(b)
        parts[nm]=(float(np.mean(hs)),np.mean(cs,0))
    W={"curv":.40,"turn":.30,"cal":.20,"ext":.10}
    hD=sum(W[k]*parts[k][0] for k in W); cD=sum(W[k]*parts[k][1] for k in W)
    return hD,cD,parts
hD,cD,parts=build()
print("\n=== HOST placed on the cohort difficulty scale (z units vs the 22) ===")
for k in parts: print("  %-5s axis: HOST z=%+.2f   (cohort range %.2f..%.2f)"%(k,parts[k][0],parts[k][1].min(),parts[k][1].max()))
print("  COMPOSITE DIFF: HOST=%+.2f   cohort range %.2f..%.2f, median %.2f  -> host percentile %.0f%%"
      %(hD,cD.min(),cD.max(),np.median(cD),100*np.mean(cD<hD)))
# per-axis ranks
print("\n=== per-axis rank of the 22 (1 = hardest) ===")
axes={"curv":parts["curv"][1],"turn":parts["turn"][1],"cal":parts["cal"][1],"ext":parts["ext"][1],"COMPOSITE":cD}
short=[a.replace("topcow_mr_","") for a in A]
for nm,v in axes.items():
    o=np.argsort(-v); print("  %-9s "%nm+" ".join(short[i] for i in o))
# deepest evaluated target vs geometry limits
print("\n=== evaluated depth coverage per anatomy ===")
by=collections.defaultdict(list)
for r in ROWS: by[r["anat"]].append(r)
print("%-12s %5s %7s %7s %7s %8s %8s"%("anat","n","s_min","s_max","L","frac_L","clr_min_route"))
for a in sorted(by):
    R=by[a]; key=[k for k in T if k!="HOST" and "topcowmr%03d"%int(k.split("_")[-1])==a][0]
    L=T[key]["rcca_len"]; sm=max(r["s_tgt"] for r in R)
    print("%-12s %5d %7.1f %7.1f %7.1f %8.2f %8.3f"%(a,len(R),min(r["s_tgt"] for r in R),sm,L,sm/L,min(r["clr_min"] for r in R)))
