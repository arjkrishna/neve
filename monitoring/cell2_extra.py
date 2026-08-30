import json, statistics as stx
from collections import Counter
import sys
sys.path.insert(0,"D:/Arjun/workspace/neve/monitoring")
from cell2_report import load, pct, q

A,_=load("A_all22"); H,_=load("H_all22"); A4,_=load("A_holdout4")
a={e["seed"]:e for e in A}; h={e["seed"]:e for e in H}
print("holdout4 seed overlap with all22: %d/98 ; anatomies=%s" % (
  len(set(e["seed"] for e in A4)&set(a)), sorted(set(e["mfp"] for e in A4))))
print("all22 anatomies n=%d" % len(set(e["mfp"] for e in A)))
print("path_len range shared=%.1f..%.1f grafted=%.1f..%.1f" % (
  min(e["pl"] for e in A if e["pl"]<=166.91), max(e["pl"] for e in A if e["pl"]<=166.91),
  min(e["pl"] for e in A if e["pl"]>166.91), max(e["pl"] for e in A if e["pl"]>166.91)))

def onsets(eps, k="cmd12"):
    return [(o["on"], e["pl"], o["k"]) for e in eps for o in e["ev"][k]]

for nm, eps in (("A_all22",A),("A_all22 GRAFTED",[e for e in A if e["pl"]>166.91]),
                ("H_all22",H),("H_all22 GRAFTED",[e for e in H if e["pl"]>166.91]),
                ("H_all22 SHARED",[e for e in H if e["pl"]<=166.91]),
                ("A_holdout4",A4)):
    o=onsets(eps)
    if not o: continue
    b=Counter(int(x//5)*5 for x,_,_ in o)
    mode=b.most_common(1)[0]
    rem=[p-x for x,p,_ in o]
    print("\n%s: n_onset=%d modal 5mm proj_s bin=%d-%d (n=%d, %s of onsets)" % (nm,len(o),mode[0],mode[0]+5,mode[1],pct(mode[1],len(o))))
    print("   top bins: %s" % " ".join("%d:%d"%(k2,v) for k2,v in sorted(b.items(), key=lambda kv:-kv[1])[:8]))
    print("   remaining-to-target (path_len-proj_s at onset): med=%.1f p10=%.1f p90=%.1f ; %%onsets with <=33.31 remaining=%s" % (
      stx.median(rem), q(rem,.1), q(rem,.9), pct(sum(1 for r in rem if r<=33.31), len(rem))))
    unr=[x for x,p,k in o if k=="unrec"]
    if unr:
        bu=Counter(int(x//10)*10 for x in unr)
        print("   UNREC onsets n=%d modal 10mm bin=%s ; med proj_s=%.1f ; med remaining=%.1f" % (
          len(unr), bu.most_common(1)[0], stx.median(unr), stx.median([p-x for x,p,k in o if k=="unrec"])))

# how far short do A's grafted failures die
print("\nA grafted failures: maxp vs path_len")
gf=[e for e in A if e["pl"]>166.91 and not e["succ"]]
print("  n=%d med maxp=%.1f med path_len=%.1f med shortfall=%.1f ; %%reaching within 33.31 of end=%s" % (
  len(gf), stx.median([e["maxp"] for e in gf]), stx.median([e["pl"] for e in gf]),
  stx.median([e["pl"]-e["maxp"] for e in gf]), pct(sum(1 for e in gf if e["pl"]-e["maxp"]<=33.31), len(gf))))
gs=[e for e in A if e["pl"]>166.91 and e["succ"]]
print("  grafted successes n=%d med steps=%.0f ; grafted fails med steps=%.0f" % (
  len(gs), stx.median([e["steps"] for e in gs]), stx.median([e["steps"] for e in gf])))
hf=[e for e in H if e["pl"]>166.91 and not e["succ"]]
print("  H grafted fails n=%d med shortfall=%.1f ; %%within 33.31=%s" % (
  len(hf), stx.median([e["pl"]-e["maxp"] for e in hf]), pct(sum(1 for e in hf if e["pl"]-e["maxp"]<=33.31), len(hf))))
