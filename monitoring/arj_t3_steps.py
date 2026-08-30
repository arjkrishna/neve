import json, numpy as np
R=json.load(open("/opt/mon/arj_t3_rows.json"))
for lo,hi,tag in ((0,80,"shallow dg<80"),(80,200,"deep dg>=80")):
    f=[r for r in R if lo<=r["dg"]<hi and r["T"]==0]
    s=[r["Tsteps"] for r in f]
    print("TEACHER fails %-14s n=%2d  steps: max=%d, n_at_600=%d, median=%.0f"%(tag,len(f),max(s),sum(1 for x in s if x>=600),np.median(s)))
    f2=[r for r in R if lo<=r["dg"]<hi and r["H"]==0]
    s2=[r["Hsteps"] for r in f2]
    print("HEUR    fails %-14s n=%2d  steps: max=%d, n_at_600=%d, median=%.0f"%(tag,len(f2),max(s2),sum(1 for x in s2 if x>=600),np.median(s2)))
suc=[r["Tsteps"] for r in R if r["T"]==1]
print("TEACHER successes steps median=%.0f p90=%.0f max=%d"%(np.median(suc),np.percentile(suc,90),max(suc)))
suc2=[r["Hsteps"] for r in R if r["H"]==1]
print("HEUR    successes steps median=%.0f p90=%.0f max=%d"%(np.median(suc2),np.percentile(suc2,90),max(suc2)))
