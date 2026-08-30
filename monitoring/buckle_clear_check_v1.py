import sys,math
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features
from buckle_clear_final_v1 import lab
A=features(sys.argv[1])
def fisher(a,b,c,d):
    # P(>= observed) one-sided, hypergeometric
    def lc(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
    n=a+b+c+d; p=0.0
    for x in range(0,min(a+b,a+c)+1):
        y=a+b-x; z=a+c-x; w=d-(a-x)
        if y<0 or z<0 or w<0: continue
        pr=math.exp(lc(a+b,x)+lc(c+d,z)-lc(n,a+c))
        if x>=a: p+=pr
    return p
cl=[e for e in A if lab(e)=="CLEARING"]; fu=[e for e in A if lab(e)=="FUTILE"]
a=sum(1 for e in cl if e["succ"]); b=len(cl)-a; c=sum(1 for e in fu if e["succ"]); d=len(fu)-c
print("CLEARING succ %d/%d, FUTILE succ %d/%d   Fisher one-sided p=%.4f"%(a,len(cl),c,len(fu),fisher(a,b,c,d)))
print()
print("Is CLEARING just 'hard'?  (among the 34 buckle-present events)")
B=[e for e in A if e["fold_load"]>=4 or e["slack_load"]>=10]
for nm,f in [("k==hard",lambda e:e["k"]=="hard"),("CLEARING",lambda e:lab(e)=="CLEARING")]:
    g=[e for e in B if f(e)]; print("  %-10s n=%2d succ=%.3f"%(nm,len(g),sum(1 for e in g if e["succ"])/len(g)))
print("  cross-tab hard x CLEARING:")
for h in (True,False):
    for k in ("CLEARING","FUTILE"):
        g=[e for e in B if (e["k"]=="hard")==h and lab(e)==k]
        print("    hard=%-5s %-8s n=%2d succ=%s"%(h,k,len(g),"%.3f"%(sum(1 for e in g if e["succ"])/len(g)) if g else "-"))
print()
print("Incremental value of the BUCKLE_PRESENT gate (apply REDUCED+PASSED to ALL soft/hard):")
g=[e for e in A if e["fold_close"]==0 and e["adv25"]>=15 and not e["restall_same"]]
print("  REDUCED+PASSED only: n=%d succ=%.3f   | with buckle gate: n=%d succ=%.3f"
      %(len(g),sum(1 for e in g if e["succ"])/len(g),len(cl),a/len(cl)))
g2=[e for e in g if not(e["fold_load"]>=4 or e["slack_load"]>=10)]
print("  dropped by buckle gate (no buckle, but advanced): n=%d succ=%.3f"%(len(g2),sum(1 for e in g2 if e["succ"])/max(1,len(g2))))
print()
print("RATES for downstream use (A ckpt2002292 host, 22,387 steps, 98 eps):")
tot=len(A)
for k in ("CLEARING","FUTILE","COSMETIC"):
    n=sum(1 for e in A if lab(e)==k)
    print("  %-8s %2d events = %.3f/1k steps = %.3f/episode = %.1f%% of soft+hard = %.1f%% of all 136 canon stalls"%(k,n,1000.*n/22387,n/98.,100.*n/tot,100.*n/136))
