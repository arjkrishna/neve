import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
f=[r for r in gt if not r["succ"]]
early=[r for r in f if r["max_s"]<25]
print("EARLY-ARREST (max_s<25mm) failures:", len(early))
print("anat seed tgt_s maxS foldmax slkmax nearest curbr dtgtfin Hsucc HmaxS")
for r in sorted(early,key=lambda x:x["anat"]):
    h=mh[r["seed"]]
    print(f"{r['anat']} {r['seed']} {r['tgt_s']:6.1f} {r['max_s']:5.1f} {r['fold_max']:4d} {r['slack_max']:6.1f} "
          f"{r['nearest']:6s} {str(r['curbr'])[:28]:28s} {r['d_tgt_final']:6.1f} {int(h['succ'])} {h['max_s']:6.1f} {h['fold_max']:4d}")
print()
print("their success counterparts in same anatomies: nearest at end for SUCCESSES")
suc=[r for r in gt if r["succ"]]
print(collections.Counter(r["nearest"] for r in suc).most_common())
print("fails nearest:", collections.Counter(r["nearest"] for r in f).most_common())
print("early nearest:", collections.Counter(r["nearest"] for r in early).most_common())
print()
print("Does the EARLY mode also occur in non-grafted (proximal) targets?")
ng=[r for r in T if not r["grafted"]]
print(" non-grafted n=%d succ=%d ; fails with max_s<25: %d"%(len(ng),sum(r['succ'] for r in ng),
      sum(1 for r in ng if not r['succ'] and r['max_s']<25)))
print(" non-grafted fails:", sum(1 for r in ng if not r['succ']))
for r in ng:
    if not r["succ"]:
        print("  ",r['anat'],r['seed'],round(r['tgt_s'],1),round(r['max_s'],1),r['fold_max'],round(r['slack_max'],1))
