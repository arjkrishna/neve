import pickle,statistics as st
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
OFF=33.314
F=[r for r in recs if not r["succ"]]
print("failures",len(F))
print("log step counts: min",min(len(r['S']) for r in F),"max",max(len(r['S']) for r in F))
print("csv steps distribution:",Counter(r["nst"] for r in F).most_common(6))
print()
# Look at a proximal-arrest example and a deep one
for tag,sel in [("PROXIMAL ~11.5",lambda r:r["max_s"]<20),("DEEP mr025",lambda r:r["mesh"]=="topcowmr025" and r["max_s"]>160)]:
    ex=[r for r in F if sel(r)][0]
    print(f"--- {tag}: {ex['mesh']} seed={ex['seed']} tgt_s={ex['tgt_s']:.1f} max_s={ex['max_s']:.1f} nsteps={len(ex['S'])}")
    for s in ex["S"][-6:]:
        print(f"   n={s['n']:4d} proj_s={s['proj_s']:7.1f} s_r={s['proj_s']-OFF:7.1f} d_tgt={s['d_tgt']:6.1f} ins={s['ins']} dins={s['dins']} cmd={s['cmd']} on_br={s['on_br']} on_path={s['on_path']} off_br={s['off_br']} br={s['br'][:34]}")
    print()
