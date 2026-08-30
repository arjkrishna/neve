import json, math, collections
import numpy as np
from scipy import stats
R=json.load(open("/opt/mon/arj_t3_rows.json"))
def het(rows,key,label):
    by=collections.defaultdict(lambda:[0,0])
    for r in rows: by[r["anat"]][0]+=1; by[r["anat"]][1]+=r[key]
    ks=sorted(by); n=sum(by[a][0] for a in ks); y=sum(by[a][1] for a in ks); p=y/n
    G2=0.0; X2=0.0
    for a in ks:
        ni,yi=by[a]; ei=ni*p
        for o,e in ((yi,ei),(ni-yi,ni-ei)):
            if o>0: G2+=2*o*math.log(o/e)
            X2+=(o-e)**2/e
    df=len(ks)-1
    rng=np.random.default_rng(13); ns=np.array([by[a][0] for a in ks]); B=200000
    sim=rng.binomial(ns[None,:],p,size=(B,len(ks))).astype(float); exp=ns[None,:]*p
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=np.where(sim>0,2*sim*np.log(np.where(sim>0,sim/exp,1)),0.0)
        f=ns[None,:]-sim; ef=ns[None,:]-exp
        t2=np.where(f>0,2*f*np.log(np.where(f>0,f/ef,1)),0.0)
    Gs=(t1+t2).sum(1)
    print("[HET] %-32s k=%2d n=%3d y=%3d p=%.3f G2=%6.3f X2=%6.3f df=%2d p_asymG=%.4f p_X=%.4f p_boot=%.4f disp=%.3f"%(
        label,len(ks),n,y,p,G2,X2,df,stats.chi2.sf(G2,df),stats.chi2.sf(X2,df),float(((Gs>=G2).sum()+1)/(B+1)),X2/df))
    return by
sh=[r for r in R if r["dg"]<80]; dp=[r for r in R if r["dg"]>=80]
print("shallow band dg<80 n=%d ; deep band dg>=80 n=%d"%(len(sh),len(dp)))
het(sh,"T","TEACHER dg<80"); het(sh,"H","HEURISTIC dg<80")
het(dp,"T","TEACHER dg>=80"); het(dp,"H","HEURISTIC dg>=80")
sh2=[r for r in sh if r["anat"]!="topcowmr025"]
het(sh2,"T","TEACHER dg<80 no-mr025"); het(sh2,"H","HEURISTIC dg<80 no-mr025")
print("\n[deep band dg>=80 per anatomy]")
by=collections.defaultdict(lambda:[0,0,0])
for r in dp: by[r["anat"]][0]+=1; by[r["anat"]][1]+=r["T"]; by[r["anat"]][2]+=r["H"]
for a in sorted(by): print("   %-12s n=%d T=%d H=%d"%(a,*by[a]))
print("\n[shallow band dg<80 per anatomy]")
by2=collections.defaultdict(lambda:[0,0,0])
for r in sh: by2[r["anat"]][0]+=1; by2[r["anat"]][1]+=r["T"]; by2[r["anat"]][2]+=r["H"]
for a in sorted(by2): print("   %-12s n=%d T=%d H=%d  T%%=%.2f"%(a,*by2[a],by2[a][1]/by2[a][0]))
