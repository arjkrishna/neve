import json,sys
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
ev=features(sys.argv[1])
B=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10]
print("BUCKLE-PRESENT n=%d  (of %d soft/hard)"%(len(B),len(ev)))
for fld,edges in [("adv25",[i*1.0 for i in range(0,31)]),("adv50",[i*2.0 for i in range(0,31)])]:
    print("\n%s histogram (buckle-present only)"%fld)
    for i in range(len(edges)-1):
        g=[e for e in B if edges[i]<=e[fld]<edges[i+1]]
        if g or i<20: print("  [%5.1f,%5.1f) %2d %s   succ=%s"%(edges[i],edges[i+1],len(g),"#"*len(g),"".join(str(int(e["succ"])) for e in g)))
    g=[e for e in B if e[fld]>=edges[-1]]
    print("  [%5.1f,  inf) %2d %s   succ=%s"%(edges[-1],len(g),"#"*len(g),"".join(str(int(e["succ"])) for e in g)))
print("\nsorted adv25 with labels:")
for e in sorted(B,key=lambda x:x["adv25"]):
    print("  adv25=%7.2f adv50=%7.2f adv_end=%7.2f retract=%7.2f k=%-4s fold_load=%2d fold_close=%2d restall=%d succ=%d seed=%s"
          %(e["adv25"],e["adv50"],e["adv_end"],e["retract"],e["k"],e["fold_load"],e["fold_close"],e["restall_same"],e["succ"],e["seed"]))
