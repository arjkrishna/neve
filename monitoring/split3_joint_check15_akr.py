exec(open(r"D:\Arjun\workspace\neve\monitoring\split_opt2_check15_akr.py").read().split("print(\"=== 2-WAY")[0])
import math, random, json
def run3(V,T,iters,seed,w=80.0):
    rnd=random.Random(seed)
    al=[rnd.randrange(3) for _ in range(NL)]; ap=[rnd.randrange(3) for _ in range(NP)]
    cnt=[0,0,0]
    for i in range(NL):
        for j in Ladj[i]:
            if al[i]==ap[j]: cnt[al[i]]+=1
    obj=lambda c: c[0]-w*(max(0,V-c[1])+max(0,T-c[2]))
    s=obj(cnt); best=(s,list(al),list(ap),list(cnt))
    T0,T1=8.0,0.02
    for it in range(iters):
        Tp=T0*(T1/T0)**(it/iters)
        if rnd.random()<NL/(NL+NP): idx=rnd.randrange(NL); arr,adj,oth=al,Ladj[idx],ap
        else: idx=rnd.randrange(NP); arr,adj,oth=ap,Padj[idx],al
        old=arr[idx]; new=rnd.choice([k for k in range(3) if k!=old])
        d=[0,0,0]
        for m in adj:
            g=oth[m]
            if g==old: d[old]-=1
            if g==new: d[new]+=1
        nc=[cnt[k]+d[k] for k in range(3)]; s2=obj(nc)
        if s2>=s or rnd.random()<math.exp((s2-s)/Tp):
            arr[idx]=new; cnt=nc; s=s2
            if s>best[0]: best=(s,list(al),list(ap),list(cnt))
    return best
print("=== JOINT 3-WAY: maximise TRAIN s.t. val>=V, test>=T ===")
sols={}
for V,T in [(22,43),(22,32),(32,32),(22,22),(16,32),(16,22)]:
    bb=None
    for sd in range(8):
        b=run3(V,T,400000,sd)
        c=b[3]
        if c[1]<V or c[2]<T: continue
        if bb is None or c[0]>bb[3][0]: bb=b
    if bb is None: print("  V>=%2d T>=%2d : no feasible solution found"%(V,T)); continue
    c=bb[3]; sols[(V,T)]=bb
    print("  V>=%2d T>=%2d -> train=%3d val=%3d test=%3d kept=%3d dropped=%3d (%.1f%%)"
          %(V,T,c[0],c[1],c[2],sum(c),216-sum(c),100*(216-sum(c))/216))
json.dump({"al":sols[(22,43)][1],"ap":sols[(22,43)][2],"lows":lows,"pats":pats},
          open(r"D:\Arjun\workspace\neve\monitoring\_joint3.json","w"))
