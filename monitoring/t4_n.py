import pickle, numpy as np, collections, random
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]; fai=[r for r in gt if not r["succ"]]
# target-depth vs arrest
print("target depth (s_RCCA) of grafted episodes: min %.1f med %.1f max %.1f"%(
  min(r['tgt_s'] for r in gt),np.median([r['tgt_s'] for r in gt]),max(r['tgt_s'] for r in gt)))
for lo,hi in ((133.6,160),(160,190),(190,215),(215,300)):
    g=[r for r in gt if lo<=r["tgt_s"]<hi]
    if not g: continue
    print("  tgt %5.0f-%3.0f n=%3d T succ %5.1f%%  H succ %5.1f%%  T arrest(fails) med %6.1f"%(
      lo,hi,len(g),100*np.mean([r['succ'] for r in g]),100*np.mean([mh[r['seed']]['succ'] for r in g]),
      np.median([r['max_s'] for r in g if not r['succ']]) if any(not r['succ'] for r in g) else float('nan')))
print()
# permutation: within-anatomy clustering of arrest stations
by=collections.defaultdict(list)
for r in fai: by[r["anat"]].append(r["max_s"])
def stat(assign):
    d=collections.defaultdict(list)
    for a,v in assign: d[a].append(v)
    num=0.0;den=0
    for a,vs in d.items():
        if len(vs)>=2: num+=np.var(vs)*len(vs); den+=len(vs)
    return num/den
obs=stat([(r["anat"],r["max_s"]) for r in fai])
vals=[r["max_s"] for r in fai]; anats=[r["anat"] for r in fai]
rng=random.Random(0); cnt=0; N=20000
for _ in range(N):
    v=vals[:]; rng.shuffle(v)
    if stat(list(zip(anats,v)))<=obs: cnt+=1
print("within-anatomy arrest-station clustering permutation test:")
print("  observed pooled within-anatomy var of arrest s = %.1f (sd %.1f); total var %.1f (sd %.1f)"%(
  obs,obs**.5,np.var(vals),np.var(vals)**.5))
print("  p(perm <= obs) = %.4f  over %d permutations"%(cnt/N,N))
print()
# same, restricted to anatomies with >=3 failures
sub=[r for r in fai if len(by[r["anat"]])>=3]
obs2=stat([(r["anat"],r["max_s"]) for r in sub]); vals2=[r["max_s"] for r in sub]; an2=[r["anat"] for r in sub]
cnt=0
for _ in range(N):
    v=vals2[:]; rng.shuffle(v)
    if stat(list(zip(an2,v)))<=obs2: cnt+=1
print("  restricted to anatomies with >=3 failures (n=%d eps, %d anat): obs var %.1f p=%.4f"%(
  len(sub),len(set(an2)),obs2,cnt/N))
print()
# heuristic own arrests
hf=[mh[r["seed"]] for r in gt if not mh[r["seed"]]["succ"]]
print("HEURISTIC grafted failures n=%d: arrest hist 10mm:"%len(hf),sorted(collections.Counter(int(r['max_s']//10)*10 for r in hf).items()))
print("  heuristic fails at s<=130: %d/%d ; timeouts (n=600): %d"%(
  sum(1 for r in hf if r['max_s']<=130),len(hf),sum(1 for r in hf if r['n']>=600)))
print("  heuristic fail arrest med %.1f ; teacher fail arrest med %.1f"%(
  np.median([r['max_s'] for r in hf]),np.median([r['max_s'] for r in fai])))
print()
# paired depth
d=[(r["max_s"]-mh[r["seed"]]["max_s"]) for r in gt]
print("paired max_s (teacher - heuristic) over 124 grafted: med %+.1f ; T deeper in %d, H deeper in %d"%(
  np.median(d),sum(1 for x in d if x>2),sum(1 for x in d if x<-2)))
db=[(r["max_s"]-mh[r["seed"]]["max_s"]) for r in fai]
print("  restricted to teacher failures (55): med %+.1f ; H deeper by >20mm in %d"%(np.median(db),sum(1 for x in db if x<-20)))
