exec(open(r"D:\Arjun\workspace\neve\monitoring\split_opt2_check15_akr.py").read().split("print(\"=== 2-WAY")[0])
import json, statistics as st
def bo(K,tgt,iters,seeds,w):
    bb=None
    for sd in range(seeds):
        b=run(K,list(tgt),iters,sd,w)
        ok=all(b[3][k]>=tgt[k] for k in range(K)); key=(1 if ok else 0,sum(b[3]))
        if bb is None or key>bb[0]: bb=(key,b)
    return bb[1]
print("=== 3-WAY hard search (w=60, 1M iters, 8 seeds) ===")
sols={}
for A,B,C in [(140,25,25),(130,30,30)]:
    b=bo(3,(A,B,C),600000,6,60.0); c=b[3]
    ok=all(c[k]>=t for k,t in enumerate((A,B,C)))
    print("  need(%d,%d,%d) -> %s total=%3d dropped=%3d feasible=%s"%(A,B,C,c,sum(c),216-sum(c),ok))
    sols[(A,B,C)]=b
json.dump({"al":sols[(140,25,25)][1],"ap":sols[(140,25,25)][2],"lows":lows,"pats":pats,"rows":rows},
          open(r"D:\Arjun\workspace\neve\monitoring\_split3_final.json","w"))
