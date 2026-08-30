"""Strict null control, scaled version for the 5,902-episode explore stream.
Same construction as buckle_clear_null_v3.py (matched-length windows of clean advance:
zero steps satisfying extract_stuck.py's stalled predicate, no overlap with any detected
stall) but with documented subsampling: distinct event lengths reduced to NL evenly
spaced quantiles, window start stride STR, episodes subsampled every EVERY-th."""
import json,sys,random
NL=int(sys.argv[2]); STR=int(sys.argv[3]); EVERY=int(sys.argv[4])
def med(v): v=sorted(v); return v[len(v)//2]
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
E={"rise":[],"foldL":[],"len":[]}; runs=[]
for li,line in enumerate(open(sys.argv[1])):
    ep=json.loads(line); proj,gw,fold,cmd=ep["proj"],ep["gw"],ep["fold"],ep["cmd"]; n=len(proj)
    slack=[gw[j]-proj[j] for j in range(n)]
    mx=-1e9; stl=[]
    for j in range(n):
        stl.append((proj[j]<mx+0.3) and (cmd[j]>2.0)); mx=max(mx,proj[j])
    cov=[False]*n; spans=[]
    for e in ep["events"]:
        a=max(0,min(e["first"]-1,n-1)); b=(e["close"]-1) if e["close"]>0 else n-1
        b=max(a,min(b,n-1)); spans.append((a,b,e))
        for j in range(a,b+1): cov[j]=True
    for a,b,e in spans:
        if e["k"] not in ("soft","hard"): continue
        pk=a+max(range(b-a+1),key=lambda i:gw[a+i])
        pre=slack[max(0,a-20):a] or slack[:1]
        E["rise"].append(max(slack[a:pk+1])-med(pre)); E["foldL"].append(max(fold[a:pk+1])); E["len"].append(b-a+1)
    if li%EVERY: continue
    ok=[(not cov[j]) and (not stl[j]) for j in range(n)]
    s=None
    for j in range(n+1):
        v=ok[j] if j<n else False
        if v and s is None: s=j
        elif not v and s is not None:
            if j-s>=5: runs.append((slack,fold,s,j-1))
            s=None
L=sorted(set(E["len"]))
sel=sorted(set(L[int(round(i*(len(L)-1)/max(1,NL-1)))] for i in range(min(NL,len(L)))))
N={"rise":[],"foldL":[]}
for Lw in sel:
    for (slack,fold,s0,e0) in runs:
        for s in range(s0,e0-Lw+2,STR):
            t=s+Lw-1
            pre=slack[max(0,s-20):s] or slack[:1]
            N["rise"].append(max(slack[s:t+1])-med(pre)); N["foldL"].append(max(fold[s:t+1]))
print("clean runs=%d clean steps=%d  null windows=%d  event lengths %d..%d (%d distinct, %d sampled) stride=%d ep_every=%d"%(
    len(runs),sum(e-s+1 for _,_,s,e in runs),len(N["rise"]),L[0],L[-1],len(L),len(sel),STR,EVERY))
for k in ("rise","foldL"):
    print("%-6s EVENT n=%-5d p25=%7.2f p50=%7.2f p75=%7.2f p90=%7.2f | NULL n=%-8d p50=%6.2f p75=%6.2f p90=%6.2f p95=%6.2f p99=%6.2f max=%6.2f"%(
        k,len(E[k]),q(E[k],25),q(E[k],50),q(E[k],75),q(E[k],90),len(N[k]),q(N[k],50),q(N[k],75),q(N[k],90),q(N[k],95),q(N[k],99),max(N[k]) if N[k] else float('nan')))
print("\n  thr | NULL P(>=t) | EVENT P(>=t) | lift")
for k in ("rise","foldL"):
    print(" -- %s --"%k)
    for t in (1,2,3,4,5,6,8,10,12,15,20):
        fn=sum(1 for v in N[k] if v>=t)/max(1,len(N[k])); fe=sum(1 for v in E[k] if v>=t)/max(1,len(E[k]))
        print("  %4d | %10.5f | %11.3f | %s"%(t,fn,fe,("%.2f"%(fe/fn)) if fn>0 else "inf"))
