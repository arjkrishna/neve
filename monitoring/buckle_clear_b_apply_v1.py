"""Apply the VERBATIM validated buckle-clearing definition (F=4, S=10mm, A=15mm) to family B."""
import json,sys,math,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
def fisher(a,b,c,d):
    def lc(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
    n=a+b+c+d; p=0.0
    for x in range(0,min(a+b,a+c)+1):
        y=a+b-x; z=a+c-x; w=d-(a-x)
        if y<0 or z<0 or w<0: continue
        if x>=a: p+=math.exp(lc(a+b,x)+lc(c+d,z)-lc(n,a+c))
    return p
for path in sys.argv[1:]:
    name=path.split("trB_")[-1].replace(".jsonl","")
    eps=[json.loads(l) for l in open(path)]
    steps=sum(len(e["proj"]) for e in eps); nall=sum(len(e["events"]) for e in eps)
    ev=features(path)
    c=collections.Counter(lab(e) for e in ev)
    print("="*140)
    print("%s   soft+hard n=%d   (all canon stalls=%d, steps=%d, episodes=%d)"%(name,len(ev),nall,steps,len(eps)))
    for k in ("CLEARING","FUTILE","COSMETIC"):
        g=[e for e in ev if lab(e)==k]
        if not g: print("   %-8s   0"%k); continue
        s=sum(1 for e in g if e["succ"])
        print("   %-8s %4d (%5.1f%% of s+h, %5.1f%% of all stalls, %.3f/1k, %.4f/ep)  P(ep succ)=%.3f (%d/%d)  med adv25=%7.1f  med retract=%7.2f  med fold_load=%4.1f  hard-share=%.2f  restall=%d"%(
            k,len(g),100.*len(g)/len(ev),100.*len(g)/max(1,nall),1000.*len(g)/max(1,steps),len(g)/len(eps),
            s/len(g),s,len(g),q([e["adv25"] for e in g],50),q([e["retract"] for e in g],50),
            q([float(e["fold_load"]) for e in g],50),sum(1 for e in g if e["k"]=="hard")/len(g),sum(e["restall_same"] for e in g)))
    cl=[e for e in ev if lab(e)=="CLEARING"]; fu=[e for e in ev if lab(e)=="FUTILE"]
    if cl and fu:
        a=sum(1 for e in cl if e["succ"]); b=len(cl)-a; cc=sum(1 for e in fu if e["succ"]); d=len(fu)-cc
        print("   >> SEPARATION P(succ|CLEARING)=%.3f vs P(succ|FUTILE)=%.3f  diff=%+.3f  ratio=%.2f  Fisher one-sided p=%.4f"%(
            a/len(cl),cc/len(fu),a/len(cl)-cc/len(fu),(a/len(cl))/max(1e-9,cc/len(fu)),fisher(a,b,cc,d)))
    if cl: print("   >> DEFERRED = %d/%d CLEARING in FAILED episode (%.0f%%)"%(sum(1 for e in cl if not e["succ"]),len(cl),100.*sum(1 for e in cl if not e["succ"])/len(cl)))
    # buckle-present marginal + hard x CLEARING
    B=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10]
    if B:
        print("   buckle-present n=%d (%.1f%% of s+h)  P(succ)=%.3f"%(len(B),100.*len(B)/len(ev),sum(1 for e in B if e["succ"])/len(B)))
        for h in (True,False):
            for k in ("CLEARING","FUTILE"):
                g=[e for e in B if (e["k"]=="hard")==h and lab(e)==k]
                print("     hard=%-5s %-8s n=%3d succ=%s"%(h,k,len(g),"%.3f"%(sum(1 for e in g if e["succ"])/len(g)) if g else "-"))
    # episode-level
    seeds={}
    for i,e in enumerate(ev): seeds.setdefault((e["seed"],e["pl"],id(e)) if e["seed"] is None else e["seed"],[]).append(e)
    print("   episodes with >=1 soft/hard event: n=%d succ=%.3f | episodes with none: n=%d succ=%.3f | run-wide %.3f"%(
        0,0,0,0,sum(1 for e in eps if e["succ"])/len(eps)))
