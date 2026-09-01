exec(open(r"D:\Arjun\workspace\neve\monitoring\split_opt2_check15_akr.py").read().split("print(\"=== 2-WAY")[0])
import json
def bo(K,tgt,iters=250000,seeds=6,w=12.0):
    bb=None
    for sd in range(seeds):
        b=run(K,list(tgt),iters,sd,w)
        ok=all(b[3][k]>=tgt[k] for k in range(K)); key=(1 if ok else 0,sum(b[3]))
        if bb is None or key>bb[0]: bb=(key,b)
    return bb[1]
print("=== 2-WAY: train>=A, test>=B ===")
best2={}
for A,B in [(150,43),(140,43),(160,40),(150,35),(160,35),(150,30),(165,30),(170,25),(175,20)]:
    b=bo(2,(A,B)); c=sorted(b[3],reverse=True)
    print("  train>=%3d test>=%2d -> %s total=%3d dropped=%3d"%(A,B,c,sum(c),216-sum(c)))
    best2[(A,B)]=b
print("\n=== 3-WAY: train>=A, val>=B, test>=C ===")
best3={}
for A,B,C in [(140,25,25),(130,30,30),(150,20,20),(145,22,22),(120,33,33),(135,25,25)]:
    b=bo(3,(A,B,C)); c=b[3]
    print("  train>=%3d val>=%2d test>=%2d -> %s total=%3d dropped=%3d"%(A,B,C,sorted(c,reverse=True),sum(c),216-sum(c)))
    best3[(A,B,C)]=b
json.dump({k:{"al":v[1],"ap":v[2],"cnt":v[3]} for k,v in [("2way",best2[(150,43)]),("3way",best3[(140,25,25)])]}|{"lows":lows,"pats":pats},
          open(r"D:\Arjun\workspace\neve\monitoring\_splits.json","w"))
