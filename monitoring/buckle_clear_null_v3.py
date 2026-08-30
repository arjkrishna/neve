"""Large-sample strict null + LOADING-PHASE (pre-retraction) features.

The stall window [first..close] is split at the gw PEAK: [first..peak] is the
LOADING phase (wire being fed against the arrest) and [peak..close] is the
RELEASE phase.  'Buckle present BEFORE the retraction' must be measured on the
loading phase only.  Null = every matched-length window of clean advance
(zero stalled steps, no overlap with any detected stall) pooled over ALL
episodes of the run, stride 1.
"""
import json, sys
def med(v): v=sorted(v); return v[len(v)//2]
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
eps=[json.loads(l) for l in open(sys.argv[1])]
P=[]
for ep in eps:
    proj,gw,fold,cmd=ep["proj"],ep["gw"],ep["fold"],ep["cmd"]; n=len(proj)
    slack=[gw[j]-proj[j] for j in range(n)]
    mx=-1e9; stl=[]
    for j in range(n):
        stl.append((proj[j]<mx+0.3) and (cmd[j]>2.0)); mx=max(mx,proj[j])
    cov=[False]*n; spans=[]
    for e in ep["events"]:
        a=max(0,min(e["first"]-1,n-1)); b=(e["close"]-1) if e["close"]>0 else n-1
        b=max(a,min(b,n-1)); spans.append((a,b,e))
        for j in range(a,b+1): cov[j]=True
    ok=[(not cov[j]) and (not stl[j]) for j in range(n)]
    P.append(dict(ep=ep,slack=slack,ok=ok,spans=spans,n=n))
# clean runs per episode
runs=[]
for d in P:
    ok=d["ok"]; s=None
    for j in range(d["n"]+1):
        v=ok[j] if j<d["n"] else False
        if v and s is None: s=j
        elif not v and s is not None:
            runs.append((d,s,j-1)); s=None
E={"rise":[],"foldL":[],"foldA":[],"len":[]}
for d in P:
    ep=d["ep"]; slack=d["slack"]; gw=ep["gw"]; fold=ep["fold"]
    for a,b,e in d["spans"]:
        if e["k"] not in ("soft","hard"): continue
        pk=a+max(range(b-a+1),key=lambda i:gw[a+i])
        pre=slack[max(0,a-20):a] or slack[:1]
        E["rise"].append(max(slack[a:pk+1])-med(pre))
        E["foldL"].append(max(fold[a:pk+1]))
        E["foldA"].append(max(fold[a:b+1]))
        E["len"].append(b-a+1)
N={"rise":[],"foldL":[],"len":[]}
lens=sorted(set(E["len"]))
for L in lens:
    for (d,s0,e0) in runs:
        slack=d["slack"]; fold=d["ep"]["fold"]
        for s in range(s0,e0-L+2):
            t=s+L-1
            pre=slack[max(0,s-20):s] or slack[:1]
            N["rise"].append(max(slack[s:t+1])-med(pre)); N["foldL"].append(max(fold[s:t+1])); N["len"].append(L)
print("clean runs=%d total clean steps=%d   null windows=%d (lengths %d..%d)"%(len(runs),sum(e-s+1 for _,s,e in runs),len(N["rise"]),lens[0],lens[-1]))
for k in ("rise","foldL"):
    print("%-6s EVENT n=%-4d p25=%6.2f p50=%6.2f p75=%6.2f p90=%6.2f | NULL n=%-6d p50=%6.2f p75=%6.2f p90=%6.2f p95=%6.2f p99=%6.2f max=%6.2f"
          %(k,len(E[k]),q(E[k],25),q(E[k],50),q(E[k],75),q(E[k],90),len(N[k]),q(N[k],50),q(N[k],75),q(N[k],90),q(N[k],95),q(N[k],99),max(N[k]) if N[k] else float('nan')))
print("\n  thr | NULL P(>=t) | EVENT P(>=t) | lift")
for k in ("rise","foldL"):
    print(" -- %s --"%k)
    for t in (1,2,3,4,5,6,7,8,10,12,15,20):
        fn=sum(1 for v in N[k] if v>=t)/max(1,len(N[k])); fe=sum(1 for v in E[k] if v>=t)/max(1,len(E[k]))
        print("  %4d | %10.4f | %11.3f | %s"%(t,fn,fe,("%.2f"%(fe/fn)) if fn>0 else "inf"))
print("\nEVENT foldA(all-window) vs foldL(loading only): p50 %.1f/%.1f p75 %.1f/%.1f"%(q(E["foldA"],50),q(E["foldL"],50),q(E["foldA"],75),q(E["foldL"],75)))
