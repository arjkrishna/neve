import json,sys
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features, med, q
eps=[json.loads(l) for l in open(sys.argv[1])]
ev=features(sys.argv[1])
byseed={}
for e in ev: byseed.setdefault(e["seed"],[]).append(e)
n_all=len(eps); s_all=sum(1 for e in eps if e["succ"])
withev=[e for e in eps if any(x["k"] in ("soft","hard") for x in e["events"])]
noev=[e for e in eps if not any(x["k"] in ("soft","hard") for x in e["events"])]
print("episodes=%d succ=%d (%.3f)"%(n_all,s_all,s_all/n_all))
print("  with >=1 soft/hard event: n=%d succ=%.3f"%(len(withev),sum(1 for e in withev if e["succ"])/len(withev)))
print("  with none              : n=%d succ=%.3f"%(len(noev),sum(1 for e in noev if e["succ"])/len(noev)))
# episodes with >=1 buckle-present (fold_load>=4 or slack_load>=10)
bp=set(s for s,g in byseed.items() if any(x["fold_load"]>=4 or x["slack_load"]>=10 for x in g))
we=set(byseed)
print("  event-eps with a BUCKLE-PRESENT event: n=%d succ=%.3f"%(len(bp),sum(1 for e in eps if e["seed"] in bp and e["succ"])/len(bp)))
print("  event-eps with NO buckle-present     : n=%d succ=%.3f"%(len(we-bp),sum(1 for e in eps if e["seed"] in (we-bp) and e["succ"])/len(we-bp)))
print()
print("Among BUCKLE-PRESENT events (n=%d): outcome vs each candidate criterion"%sum(1 for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10))
B=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10]
def split(name,f):
    a=[e for e in B if f(e)]; b=[e for e in B if not f(e)]
    ga=sum(1 for e in a if e["succ"])/len(a) if a else float('nan')
    gb=sum(1 for e in b if e["succ"])/len(b) if b else float('nan')
    print("  %-26s TRUE n=%2d succ=%.3f medfrac=%.3f | FALSE n=%2d succ=%.3f medfrac=%.3f | diff %+0.3f"
          %(name,len(a),ga,q([e['frac_path'] for e in a],50) if a else float('nan'),
            len(b),gb,q([e['frac_path'] for e in b],50) if b else float('nan'),ga-gb))
for A_ in (5,10,15,25,40):
    split("adv25>=%d"%A_, lambda e,A_=A_: e["adv25"]>=A_)
for A_ in (10,25,50):
    split("adv50>=%d"%A_, lambda e,A_=A_: e["adv50"]>=A_)
for A_ in (25,50,75):
    split("adv_end>=%d"%A_, lambda e,A_=A_: e["adv_end"]>=A_)
split("no restall_same", lambda e: not e["restall_same"])
split("retract>8 (hard)", lambda e: e["k"]=="hard")
split("slack_resid<=0", lambda e: e["slack_resid"]<=0)
split("fold_close==0", lambda e: e["fold_close"]==0)
split("stall_len<=70", lambda e: e["stall_len"]<=70)
print()
print("PER-EPISODE: does the episode contain a buckle-present event that is FOLLOWED by a durable pass?")
# durable: after close, the episode never again stalls within 2mm of p0
for line in []: pass
