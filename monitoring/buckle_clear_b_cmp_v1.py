"""Head-to-head A vs B on the 98 identical host seeds + coil contrast (gw-slack criterion only,
because cath_slack is dead in the July-2026 A/H0 log build)."""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
rows=[]
for tag,path in [(a,b) for a,b in zip(sys.argv[1::2],sys.argv[2::2])]:
    eps=[json.loads(l) for l in open(path)]
    ev=features(path); idx=[]
    for i,ep in enumerate(eps):
        for e in ep["events"]:
            if e["k"] in ("soft","hard"): idx.append(i)
    for e,i in zip(ev,idx): e["epi"]=i
    steps=sum(len(e["proj"]) for e in eps); nall=sum(len(e["events"]) for e in eps)
    c=collections.Counter(lab(e) for e in ev)
    S=[[g-p for g,p in zip(e["gw"],e["proj"])] for e in eps]
    mgs=[max(s) for s in S]
    coil=[i for i in range(len(eps)) if mgs[i]>100.0]; cset=set(coil)
    cev=[e for e in ev if e["epi"] in cset]
    ccl={e["epi"] for e in cev if lab(e)=="CLEARING"}
    def r(k): return c[k]
    sc=lambda k:(sum(1 for e in ev if lab(e)==k and e["succ"])/max(1,c[k]))
    rows.append((tag,len(eps),sum(1 for e in eps if e["succ"])/len(eps),steps,nall,len(ev),
        r("CLEARING"),r("FUTILE"),r("COSMETIC"),sc("CLEARING"),sc("FUTILE"),sc("COSMETIC"),
        1000.*r("CLEARING")/steps,100.*r("CLEARING")/max(1,nall),
        len(coil),100.*len(coil)/len(eps),sum(1 for i in coil if eps[i]["succ"]),
        len(ccl),sum(1 for i in ccl if eps[i]["succ"])))
h="%-16s %4s %6s %8s %6s %5s | %5s %5s %5s | %6s %6s %6s | %8s %7s | %5s %6s %5s %6s %5s"
print(h%("cell","eps","succ","steps","stalls","s+h","CLR","FUT","COS","P|CLR","P|FUT","P|COS","CLR/1k","%stalls","coil","coil%","cSuc","cCLR","cCsu"))
for t in rows:
    print("%-16s %4d %6.3f %8d %6d %5d | %5d %5d %5d | %6.3f %6.3f %6.3f | %8.3f %7.1f | %5d %6.1f %5d %6d %5d"%t)
