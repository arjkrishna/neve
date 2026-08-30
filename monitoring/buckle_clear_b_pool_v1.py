"""Pooled B-host numbers, coil-success timing, and A-vs-B pooled Fisher."""
import json,sys,math,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
def fisher(a,b,c,d):
    def lc(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
    n=a+b+c+d; p=0.0
    for x in range(0,min(a+b,a+c)+1):
        z=a+c-x; w=d-(a-x); y=a+b-x
        if y<0 or z<0 or w<0: continue
        if x>=a: p+=math.exp(lc(a+b,x)+lc(c+d,z)-lc(n,a+c))
    return p
def load(p):
    eps=[json.loads(l) for l in open(p)]; ev=features(p); idx=[]
    for i,ep in enumerate(eps):
        for e in ep["events"]:
            if e["k"] in ("soft","hard"): idx.append(i)
    for e,i in zip(ev,idx): e["epi"]=i
    return eps,ev
POOL=[]; TOT=dict(steps=0,stalls=0,eps=0)
for p in sys.argv[1:]:
    eps,ev=load(p); POOL+=ev
    TOT["steps"]+=sum(len(e["proj"]) for e in eps); TOT["stalls"]+=sum(len(e["events"]) for e in eps); TOT["eps"]+=len(eps)
    # coil-success timing
    name=p.split("trB_")[-1].replace(".jsonl","")
    S=[[g-p2 for g,p2 in zip(e["gw"],e["proj"])] for e in eps]
    mcs=[max(e["cs"]) if e["cs"] else 0.0 for e in eps]; mgs=[max(s) for s in S]
    coil=[i for i in range(len(eps)) if mcs[i]>50.0 or mgs[i]>100.0]
    cs_ok=[i for i in coil if eps[i]["succ"]]
    print("### %s coiled+SUCCESS episodes n=%d"%(name,len(cs_ok)))
    for i in cs_ok:
        s=S[i]; pr=eps[i]["proj"]; n=len(pr)
        jp=max(range(n),key=lambda k:s[k]); jm=max(range(n),key=lambda k:pr[k])
        print("   ep#%-4d seed=%-8s steps=%-4d maxProj=%6.1f/%6.1f  slackPeak@step %d (%.0f%%)  maxProj@step %d (%.0f%%)  peakSlack=%7.1f finalSlack=%7.1f  peakCath=%7.1f finalCath=%7.1f  coilPeakBeforeTargetReach=%d"%(
            i,eps[i]["seed"],n,max(pr),eps[i]["pl"],jp,100.*jp/n,jm,100.*jm/n,max(s),s[-1],mcs[i],eps[i]["cs"][-1] if eps[i]["cs"] else 0.0,int(jp<jm)))
c=collections.Counter(lab(e) for e in POOL)
cl=[e for e in POOL if lab(e)=="CLEARING"]; fu=[e for e in POOL if lab(e)=="FUTILE"]
a=sum(1 for e in cl if e["succ"]); cc=sum(1 for e in fu if e["succ"])
print("\nPOOLED (%d cells): eps=%d steps=%d stalls=%d soft+hard=%d"%(len(sys.argv)-1,TOT["eps"],TOT["steps"],TOT["stalls"],len(POOL)))
for k in ("CLEARING","FUTILE","COSMETIC"):
    g=[e for e in POOL if lab(e)==k]
    print("  %-8s %3d (%5.1f%% of s+h, %5.1f%% of stalls, %.3f/1k)  P(succ)=%.3f (%d/%d)"%(
        k,len(g),100.*len(g)/len(POOL),100.*len(g)/TOT["stalls"],1000.*len(g)/TOT["steps"],
        sum(1 for e in g if e["succ"])/max(1,len(g)),sum(1 for e in g if e["succ"]),len(g)))
print("  SEPARATION %.3f vs %.3f = %+.3f  Fisher one-sided p=%.4f"%(a/len(cl),cc/len(fu),a/len(cl)-cc/len(fu),fisher(a,len(cl)-a,cc,len(fu)-cc)))
B=[e for e in POOL if e["fold_load"]>=4 or e["slack_load"]>=10]
print("  buckle-present n=%d (%.1f%%)  hard x CLEARING:"%(len(B),100.*len(B)/len(POOL)))
for h in (True,False):
    for k in ("CLEARING","FUTILE"):
        g=[e for e in B if (e["k"]=="hard")==h and lab(e)==k]
        print("    hard=%-5s %-8s n=%2d succ=%s"%(h,k,len(g),"%.3f"%(sum(1 for e in g if e["succ"])/len(g)) if g else "-"))
