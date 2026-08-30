import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows.pkl","rb")); T=D["T"]; H=D["H"]
epsT={e["seed"]:e for e in pickle.load(open("_t4_teacher.pkl","rb"))}
epsH={e["seed"]:e for e in pickle.load(open("_t4_heur.pkl","rb"))}
def enrich(rows, eps):
    for r in rows:
        e=eps[r["seed"]]; d1=np.array(e["dins1"]); n=len(d1)
        w=min(300,n)
        r["tailv"]=float(np.nanmean(np.abs(d1[-w:])))
        r["allv"]=float(np.nanmean(np.abs(d1)))
        r["v50"]=float(np.nanmean(np.abs(d1[:50])))
        ps=np.array(e["projs"])-33.31
        r["s_hist"]=ps
        # last-200 dwell mode in 5mm bins
        t=ps[-min(200,n):]
        b=np.floor(t/5.0)*5
        r["dwell_mode"]=float(collections.Counter(b).most_common(1)[0][0])+2.5
        r["dwell_iqr"]=float(np.percentile(t,75)-np.percentile(t,25))
        fo=np.array(e["fold"])
        # s at first time fold hits 20
        r["s_at_fold20"]=float(ps[np.argmax(fo>=20)]) if (fo>=20).any() else None
        r["i_fold20"]=int(np.argmax(fo>=20)) if (fo>=20).any() else None
    return rows
T=enrich(T,epsT); H=enrich(H,epsH)
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
suc=[r for r in gt if r["succ"]]; fai=[r for r in gt if not r["succ"]]
print("mean |d_ins_gw| per step, teacher grafted:")
print("  successes: allv med %.2f  ; failures allv med %.2f ; failures tail300 med %.2f"%(
   np.median([r["allv"] for r in suc]),np.median([r["allv"] for r in fai]),np.median([r["tailv"] for r in fai])))
print("  failures tailv distribution:", np.round(np.percentile([r["tailv"] for r in fai],[0,10,25,50,75,90,100]),2))
def mode(r):
    if r["tailv"]<0.5: return "FREEZE"
    if r["fold_max"]>=20: return "BUCKLE"
    return "STALL"
c=collections.Counter(mode(r) for r in fai); print("failure modes:",c)
for m in ("FREEZE","BUCKLE","STALL"):
    g=[r for r in fai if mode(r)==m]
    if not g: continue
    print(f"\n--- {m} n={len(g)} ---")
    print("  max_s: med %.1f  [%.1f,%.1f]"%(np.median([r['max_s'] for r in g]),min(r['max_s'] for r in g),max(r['max_s'] for r in g)))
    print("  dwell_mode: med %.1f ; fold_max med %d ; slack_max med %.1f ; tailv med %.2f"%(
      np.median([r['dwell_mode'] for r in g]),np.median([r['fold_max'] for r in g]),
      np.median([r['slack_max'] for r in g]),np.median([r['tailv'] for r in g])))
    print("  push/pull/hold med: %.2f/%.2f/%.2f"%(np.median([r['push_frac'] for r in g]),
      np.median([r['pull_frac'] for r in g]),np.median([r['hold_frac'] for r in g])))
    print("  heuristic on same seeds: succ %d/%d, med max_s %.1f"%(
      sum(mh[r['seed']]['succ'] for r in g),len(g),np.median([mh[r['seed']]['max_s'] for r in g])))
    print("  anat:",collections.Counter(r['anat'] for r in g).most_common())
pickle.dump(dict(T=T,H=H),open("_t4_rows2.pkl","wb"))
