import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows2.pkl","rb")); T=D["T"]; H=D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]; fai=[r for r in gt if not r["succ"]]
def bnd(s):
    if s<25: return "A_ostium<25"
    if s<100: return "B_25-100"
    if s<138.6: return "C_100-138.6(preseam)"
    if s<167: return "D_138.6-167"
    if s<=200: return "E_167-200"
    return "F_>200"
c=collections.Counter(bnd(r["max_s"]) for r in fai)
tot=len(fai)
print("ARREST BAND (deepest s_RCCA) for 55 grafted failures:")
for k in sorted(c): print(f"  {k:22s} {c[k]:3d}  {100*c[k]/tot:5.1f}%")
print()
print("per band: fold/slack/tailv/heuristic")
for k in sorted(c):
    g=[r for r in fai if bnd(r["max_s"])==k]
    print(f"  {k:22s} n={len(g):2d} foldmax med {np.median([r['fold_max'] for r in g]):5.0f} "
          f"fold>=20 {sum(1 for r in g if r['fold_max']>=20):2d} slackmax med {np.median([r['slack_max'] for r in g]):6.1f} "
          f"tailv {np.median([r['tailv'] for r in g]):.2f} push {np.median([r['push_frac'] for r in g]):.2f} "
          f"| H succ {sum(mh[r['seed']]['succ'] for r in g):2d}/{len(g):2d} H maxs med {np.median([mh[r['seed']]['max_s'] for r in g]):6.1f}")
print()
print("=== per-anatomy arrest clustering (failures only) ===")
by=collections.defaultdict(list)
for r in fai: by[r["anat"]].append(r)
print("anat     nf  arrest s_RCCA (sorted)                                        spread  modal-band")
for a in sorted(by):
    g=sorted(by[a],key=lambda x:x["max_s"]); v=[r["max_s"] for r in g]
    sp=max(v)-min(v)
    mb=collections.Counter(bnd(x) for x in v).most_common(1)[0]
    print(f"{a} {len(v):3d}  "+" ".join(f"{x:6.1f}" for x in v)+f"   sd={np.std(v):6.1f} rng={sp:6.1f}  {mb[0]}({mb[1]}/{len(v)})")
print()
print("=== fold>=20 per anatomy (ALL grafted episodes, teacher vs heuristic) ===")
byg=collections.defaultdict(list)
for r in gt: byg[r["anat"]].append(r)
print("anat      n  T:f>=20  rate   succ|f>=20  succ|f<20   H:f>=20 rate")
tf=sum(1 for r in gt if r['fold_max']>=20); hf=sum(1 for r in gt if mh[r['seed']]['fold_max']>=20)
for a in sorted(byg):
    g=byg[a]; n=len(g); k=sum(1 for r in g if r["fold_max"]>=20)
    s1=[r["succ"] for r in g if r["fold_max"]>=20]; s0=[r["succ"] for r in g if r["fold_max"]<20]
    hk=sum(1 for r in g if mh[r["seed"]]["fold_max"]>=20)
    f1=f"{sum(s1)}/{len(s1)}" if s1 else "-"
    f0=f"{sum(s0)}/{len(s0)}" if s0 else "-"
    print(f"{a} {n:3d} {k:5d} {100*k/n:7.1f}   {f1:>9s}  {f0:>9s}     {hk:3d} {100*hk/n:6.1f}")
print(f"TOTAL     {len(gt)} {tf} {100*tf/len(gt):.1f}%  heur {hf} {100*hf/len(gt):.1f}%")
s1=[r["succ"] for r in gt if r["fold_max"]>=20]; s0=[r["succ"] for r in gt if r["fold_max"]<20]
print("teacher succ | fold>=20: %d/%d = %.1f%% ; fold<20: %d/%d = %.1f%%"%(sum(s1),len(s1),100*np.mean(s1),sum(s0),len(s0),100*np.mean(s0)))
h1=[mh[r["seed"]]["succ"] for r in gt if mh[r["seed"]]["fold_max"]>=20]
h0=[mh[r["seed"]]["succ"] for r in gt if mh[r["seed"]]["fold_max"]<20]
print("heur    succ | fold>=20: %d/%d ; fold<20: %d/%d = %.1f%%"%(sum(h1),len(h1),sum(h0),len(h0),100*np.mean(h0)))
print()
print("buckle onset station s_at_fold20 (teacher grafted eps with fold>=20):")
v=[r["s_at_fold20"] for r in gt if r["s_at_fold20"] is not None]
print("  n=%d med %.1f  p10 %.1f p25 %.1f p75 %.1f p90 %.1f"%(len(v),np.median(v),*np.percentile(v,[10,25,75,90])))
print("  hist 10mm:",sorted(collections.Counter(int(x//10)*10 for x in v).items()))
