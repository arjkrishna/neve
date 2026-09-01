import json, os, random, math
from collections import defaultdict
ROOT = r"D:\Arjun\workspace\neve\carotid_data\anatomies"
rows=[]
for d in sorted(os.listdir(ROOT)):
    j=json.load(open(os.path.join(ROOT,d,"provenance.json"),encoding="utf-8"))
    sip=j["siphon"]; pat=sip[:-2] if sip.endswith("_L") else sip
    rows.append((d,j["lower"],sip,pat))
lows=sorted(set(r[1] for r in rows)); pats=sorted(set(r[3] for r in rows))
LI={n:i for i,n in enumerate(lows)}; PI={n:i for i,n in enumerate(pats)}
NL,NP=len(lows),len(pats)
Ladj=defaultdict(list); Padj=defaultdict(list)
for _,l,_,p in rows:
    Ladj[LI[l]].append(PI[p]); Padj[PI[p]].append(LI[l])

def run(K,targets,iters,seed,w):
    rnd=random.Random(seed)
    al=[rnd.randrange(K) for _ in range(NL)]; ap=[rnd.randrange(K) for _ in range(NP)]
    cnt=[0]*K
    for i in range(NL):
        for j in Ladj[i]:
            if al[i]==ap[j]: cnt[al[i]]+=1
    def obj(c): return sum(c)-w*sum(max(0,targets[k]-c[k]) for k in range(K))
    s=obj(cnt); best=(s,list(al),list(ap),list(cnt))
    T0,T1=8.0,0.02
    for it in range(iters):
        T=T0*(T1/T0)**(it/iters)
        if rnd.random()<NL/(NL+NP): idx=rnd.randrange(NL); arr,adj,oth=al,Ladj[idx],ap
        else: idx=rnd.randrange(NP); arr,adj,oth=ap,Padj[idx],al
        old=arr[idx]; new=rnd.randrange(K)
        if new==old: continue
        d=[0]*K
        for m in adj:
            g=oth[m]
            if g==old: d[old]-=1
            if g==new: d[new]+=1
        nc=[cnt[k]+d[k] for k in range(K)]; s2=obj(nc)
        if s2>=s or rnd.random()<math.exp((s2-s)/T):
            arr[idx]=new; cnt=nc; s=s2
            if s>best[0] and all(cnt[k]>=targets[k] for k in range(K)): best=(s,list(al),list(ap),list(cnt))
            elif s>best[0] and not all(best[3][k]>=targets[k] for k in range(K)): best=(s,list(al),list(ap),list(cnt))
    return best

def bestof(K,tgt,iters=120000,seeds=4,w=12.0):
    bb=None
    for sd in range(seeds):
        b=run(K,list(tgt),iters,sd,w)
        ok=all(b[3][k]>=tgt[k] for k in range(K))
        key=(1 if ok else 0, sum(b[3]))
        if bb is None or key>bb[0]: bb=(key,b)
    return bb[1]

print("=== 2-WAY sweep: require test>=T, maximise total kept ===")
for T in [43,40,35,30,25,20]:
    b=bestof(2,(0,T)); c=b[3]
    print("  test>=%2d -> train=%3d test=%3d total=%3d dropped=%3d" % (T,c[0],c[1],sum(c),216-sum(c)))
print("\n=== 3-WAY sweep: require val>=V and test>=T ===")
out=None
for (V,T) in [(33,33),(30,30),(25,25),(22,22),(20,20),(16,16)]:
    b=bestof(3,(0,V,T)); c=b[3]
    print("  val>=%2d test>=%2d -> train=%3d val=%3d test=%3d total=%3d dropped=%3d"%(V,T,c[0],c[1],c[2],sum(c),216-sum(c)))
    if (V,T)==(25,25): out=b
json.dump({"al":out[1],"ap":out[2],"lows":lows,"pats":pats},open(r"D:\Arjun\workspace\neve\monitoring\_split3.json","w"))
