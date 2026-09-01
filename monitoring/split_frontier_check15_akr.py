exec(open(r"D:\Arjun\workspace\neve\monitoring\split_opt2_check15_akr.py").read().split("print(\"=== 2-WAY")[0])
import json, math, random
from collections import defaultdict

def anneal_sub(Lset, Pset, tgt, iters, seed, w):
    """2-way anneal restricted to donor subsets; returns assignment dicts + counts."""
    Ls=sorted(Lset); Ps=sorted(Pset)
    li={n:k for k,n in enumerate(Ls)}; pi={n:k for k,n in enumerate(Ps)}
    la=defaultdict(list); pa=defaultdict(list)
    for i in Ls:
        for j in Ladj[i]:
            if j in Pset: la[li[i]].append(pi[j]); pa[pi[j]].append(li[i])
    nl,np_=len(Ls),len(Ps)
    rnd=random.Random(seed)
    al=[rnd.randrange(2) for _ in range(nl)]; ap=[rnd.randrange(2) for _ in range(np_)]
    cnt=[0,0]
    for i in range(nl):
        for j in la[i]:
            if al[i]==ap[j]: cnt[al[i]]+=1
    obj=lambda c: c[0]-w*max(0,tgt[1]-c[1])
    s=obj(cnt); best=(s,list(al),list(ap),list(cnt))
    T0,T1=8.0,0.02
    for it in range(iters):
        T=T0*(T1/T0)**(it/iters)
        if rnd.random()<nl/(nl+np_): idx=rnd.randrange(nl); arr,adj,oth=al,la[idx],ap
        else: idx=rnd.randrange(np_); arr,adj,oth=ap,pa[idx],al
        old=arr[idx]; new=1-old
        d=[0,0]
        for m in adj:
            g=oth[m]
            if g==old: d[old]-=1
            if g==new: d[new]+=1
        nc=[cnt[0]+d[0],cnt[1]+d[1]]; s2=obj(nc)
        if s2>=s or rnd.random()<math.exp((s2-s)/T):
            arr[idx]=new; cnt=nc; s=s2
            if s>best[0]: best=(s,list(al),list(ap),list(cnt))
    return best,Ls,Ps

ALLL=set(range(NL)); ALLP=set(range(NP))
print("=== 2-WAY FRONTIER (feasible only): maximise TRAIN s.t. TEST >= B ===")
frontier={}
for B in [43,40,36,32,30,26,22,18]:
    bb=None
    for sd in range(6):
        b,Ls,Ps=anneal_sub(ALLL,ALLP,(0,B),250000,sd,60.0)
        c=b[3]
        if c[1]<B: continue
        if bb is None or c[0]>bb[0][3][0]: bb=(b,Ls,Ps)
    if bb is None: print("  test>=%2d : INFEASIBLE at this budget"%B); continue
    b,Ls,Ps=bb; c=b[3]
    print("  test>=%2d -> train=%3d test=%3d kept=%3d dropped=%3d (%.1f%%)"%(B,c[0],c[1],sum(c),216-sum(c),100*(216-sum(c))/216))
    frontier[B]=(b,Ls,Ps)
json.dump({str(k):{"al":v[0][1],"ap":v[0][2],"Ls":v[1],"Ps":v[2],"cnt":v[0][3]} for k,v in frontier.items()}
          |{"lows":lows,"pats":pats}, open(r"D:\Arjun\workspace\neve\monitoring\_frontier.json","w"))
