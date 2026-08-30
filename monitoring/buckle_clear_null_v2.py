"""STRICT null: matched-length windows containing ZERO stalled steps.

'stalled' uses extract_stuck.py's own predicate (proj < running_max+0.3 AND
push>2.0), so a strict-null window is a stretch of genuinely clean advance --
the correct no-buckle reference.  Also reports the PINNED-FEED instrument:
gw fed while the tip is arrested = max(gw)-gw[start] over the window.
"""
import json, sys
def med(v): v=sorted(v); return v[len(v)//2]
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
E={"rise":[],"fold":[],"fed":[],"proj_gain":[]}
N={"rise":[],"fold":[],"fed":[],"proj_gain":[]}
for line in open(sys.argv[1]):
    ep=json.loads(line)
    proj,gw,fold,cmd=ep["proj"],ep["gw"],ep["fold"],ep["cmd"]; n=len(proj)
    if n<5: continue
    slack=[gw[j]-proj[j] for j in range(n)]
    mx=-1e9; st=[]
    for j in range(n):
        st.append((proj[j]<mx+0.3) and (cmd[j]>2.0)); mx=max(mx,proj[j])
    cov=[False]*n; spans=[]
    for e in ep["events"]:
        a=max(0,min(e["first"]-1,n-1)); b=(e["close"]-1) if e["close"]>0 else n-1
        b=max(a,min(b,n-1)); spans.append((a,b,e))
        for j in range(a,b+1): cov[j]=True
    def feat(D,s,t):
        pre=slack[max(0,s-20):s] or slack[:1]
        D["rise"].append(max(slack[s:t+1])-med(pre))
        D["fold"].append(max(fold[s:t+1]))
        D["fed"].append(max(gw[s:t+1])-gw[s])
        D["proj_gain"].append(max(proj[s:t+1])-proj[s])
    for a,b,e in spans:
        if e["k"] not in ("soft","hard"): continue
        feat(E,a,b); L=b-a+1
        for s in range(0,n-L,5):
            t=s+L-1
            if t>=n or any(cov[s:t+1]) or any(st[s:t+1]): continue
            feat(N,s,t)
for k in ("rise","fold","fed","proj_gain"):
    print("%-10s EVENT n=%-4d p25=%6.2f p50=%6.2f p75=%6.2f p90=%6.2f  |  NULL n=%-5d p50=%6.2f p75=%6.2f p90=%6.2f p95=%6.2f p99=%6.2f max=%6.2f"
          %(k,len(E[k]),q(E[k],25),q(E[k],50),q(E[k],75),q(E[k],90),len(N[k]),q(N[k],50),q(N[k],75),q(N[k],90),q(N[k],95),q(N[k],99),max(N[k]) if N[k] else float('nan')))
print()
print("thr    fracNULL>=t   fracEVENT>=t   lift")
for k,ts in [("rise",(1,2,3,4,5,6,8,10)),("fold",(1,2,3,4,5,6,8,10)),("fed",(1,2,3,4,5,6,8,10))]:
    print(" -- %s --"%k)
    for t in ts:
        fn=sum(1 for v in N[k] if v>=t)/max(1,len(N[k])); fe=sum(1 for v in E[k] if v>=t)/max(1,len(E[k]))
        print("  %5.1f  %8.4f    %8.3f    %6.2f"%(t,fn,fe,fe/max(fn,1e-9)))
