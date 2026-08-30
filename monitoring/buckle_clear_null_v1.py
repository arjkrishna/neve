"""NULL CONTROL for 'buckle present before'.

For every soft/hard event of length L, slide a window of the SAME length L over
the parts of the SAME episode that are NOT inside any detected stall window
(stride 10), and compute slack_rise / fold_max exactly as for the event.  Those
windows are stretches of ordinary navigation, so their slack_rise is the
no-buckle reference.  A threshold placed at a high percentile of this null is
the level above which the stall's slack build-up is not ordinary drift.
"""
import json, sys
def med(v): v=sorted(v); return v[len(v)//2]
def q(v,p):
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
nullr=[]; nullf=[]; evr=[]; evf=[]
for line in open(sys.argv[1]):
    ep=json.loads(line)
    proj,gw,fold=ep["proj"],ep["gw"],ep["fold"]; n=len(proj)
    slack=[gw[j]-proj[j] for j in range(n)]
    cov=[False]*n
    spans=[]
    for e in ep["events"]:
        a=max(0,min(e["first"]-1,n-1)); b=(e["close"]-1) if e["close"]>0 else n-1
        b=max(a,min(b,n-1)); spans.append((a,b,e))
        for j in range(a,b+1): cov[j]=True
    for a,b,e in spans:
        if e["k"] not in ("soft","hard"): continue
        L=b-a+1
        pre=slack[max(0,a-20):a] or slack[:1]
        evr.append(max(slack[a:b+1])-med(pre)); evf.append(max(fold[a:b+1]))
        for s in range(0,n-L,10):
            t=s+L-1
            if t>=n or any(cov[s:t+1]): continue
            pre2=slack[max(0,s-20):s] or slack[:1]
            nullr.append(max(slack[s:t+1])-med(pre2)); nullf.append(max(fold[s:t+1]))
print("EVENT slack_rise  n=%d  p50=%.2f p75=%.2f p90=%.2f"%(len(evr),q(evr,50),q(evr,75),q(evr,90)))
print("NULL  slack_rise  n=%d  p50=%.2f p75=%.2f p90=%.2f p95=%.2f p97.5=%.2f p99=%.2f max=%.2f"
      %(len(nullr),q(nullr,50),q(nullr,75),q(nullr,90),q(nullr,95),q(nullr,97.5),q(nullr,99),max(nullr)) if nullr else "NULL none")
print("EVENT fold_max    p50=%.1f p75=%.1f p90=%.1f"%(q(evf,50),q(evf,75),q(evf,90)))
if nullf:
    print("NULL  fold_max    p50=%.1f p75=%.1f p90=%.1f p95=%.1f p99=%.1f max=%d"%(q(nullf,50),q(nullf,75),q(nullf,90),q(nullf,95),q(nullf,99),max(nullf)))
    for t in (1,2,3,4,5,6,8,10):
        print("   frac NULL fold_max>=%2d : %.4f     frac EVENT >= : %.3f"%(t,sum(1 for v in nullf if v>=t)/len(nullf),sum(1 for v in evf if v>=t)/len(evf)))
for t in (0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0):
    print("   frac NULL slack_rise>=%.1f : %.4f    frac EVENT >= : %.3f"%(t,sum(1 for v in nullr if v>=t)/max(1,len(nullr)),sum(1 for v in evr if v>=t)/len(evr)))
