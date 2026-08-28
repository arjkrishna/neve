import pickle,json,statistics as st,math
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
GG=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_geo.pkl","rb")); G=GG["G"]
def prof(a,s):
    g=G[a]["g"]; d=G[a]["d"]; r=G[a]["r_decl"]
    i=min(range(len(g)),key=lambda k:abs(g[k]-s))
    return d[i],r[i]
def nearest_below(a,s,thr):
    g=G[a]["g"]; d=G[a]["d"]
    c=[g[i] for i in range(len(g)) if d[i]<thr]
    if not c: return None
    return min(c,key=lambda x:abs(x-s))
F=[r for r in recs if not r["succ"]]; S_=[r for r in recs if r["succ"]]

print("=== D. CLEARANCE AT THE ARREST POINT (every failure) ===")
print(f"{'mesh':<12}{'seed':>7}{'mode':<24}{'arr_s':>7}{'clr@arr':>8}{'r_decl':>7}{'nearest<0.30':>13}{'dist':>7}{'tgt_s':>7}{'short_by':>9}")
for r in sorted(F,key=lambda x:(x["mesh"],x["seed"])):
    c,rd=prof(r["mesh"],r["max_s"])
    nb=nearest_below(r["mesh"],r["max_s"],0.30)
    r["clr_at_arrest"]=c
    print(f"{r['mesh']:<12}{r['seed']:>7}{r['mode']:<24}{r['max_s']:>7.1f}{c:>8.3f}{rd:>7.2f}"
          f"{('%.1f'%nb) if nb is not None else '-':>13}{('%+.1f'%(nb-r['max_s'])) if nb is not None else '-':>7}"
          f"{r['tgt_s']:>7.1f}{r['tgt_s']-r['max_s']:>9.1f}")
cl=[r["clr_at_arrest"] for r in F]
print(f"\nclearance at arrest, all {len(F)} failures: min={min(cl):.3f} p05={sorted(cl)[len(cl)//20]:.3f} "
      f"median={st.median(cl):.3f} max={max(cl):.3f}   below 0.30: {sum(1 for x in cl if x<0.30)}  "
      f"below 0.35: {sum(1 for x in cl if x<0.35)}  below 1.0mm: {sum(1 for x in cl if x<1.0)}")
w=[r for r in F if r["mode"]=="i_wall_feed_no_advance"]
clw=[r["clr_at_arrest"] for r in w]
print(f"mode-i 'wall' subset (n={len(w)}): clr@arrest min={min(clw):.3f} median={st.median(clw):.3f} max={max(clw):.3f}"
      f"  below 0.30: {sum(1 for x in clw if x<0.30)}")

print()
print("=== E. CORRELATION TESTS ACROSS ANATOMIES ===")
def pear(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy) if sx*sy else float('nan')
byA=defaultdict(list)
for r in recs: byA[r["mesh"]].append(r)
X=[]
for a in sorted(byA):
    fs=[r for r in byA[a] if not r["succ"]]
    if not fs: continue
    g=G[a]; L=g["L"]
    m=[r for r in g["s030"] if r["s1"]<L-8.0]
    X.append(dict(a=a,mode_s=st.median(sorted(r["max_s"] for r in fs)),
        med_arr=st.median([r["max_s"] for r in fs]),
        blk=(min(r["s_at_min"] for r in m) if m else None),
        minclr=min(g["d"]), L=L, nf=len(fs), n=len(byA[a]),
        med_tgt=st.median([r["tgt_s"] for r in fs]),
        sr=100*sum(1 for r in byA[a] if r["succ"])/len(byA[a])))
have=[x for x in X if x["blk"] is not None]
print(f"anatomies with >=1 failure: {len(X)}   of these, with a MID-VESSEL <0.30 blockage: {len(have)}")
for x in have: print(f"   {x['a']}: blk_s={x['blk']:.1f}  median_arrest={x['med_arr']:.1f}  delta={x['med_arr']-x['blk']:+.1f} mm")
print("   -> Pearson r(blk_s, modal_arrest) across anatomies: UNDEFINED (n=2, predictor null for 20/22)")
print(f"\nr(median_failure_arrest, median_failure_TARGET depth) over {len(X)} anatomies = "
      f"{pear([x['med_arr'] for x in X],[x['med_tgt'] for x in X]):+.3f}")
print(f"r(median_failure_arrest, branch terminus L)            = {pear([x['med_arr'] for x in X],[x['L'] for x in X]):+.3f}")
print(f"r(median_failure_arrest, branch min clearance)         = {pear([x['med_arr'] for x in X],[x['minclr'] for x in X]):+.3f}")
print(f"r(anatomy success rate, branch min clearance)          = {pear([x['sr'] for x in X],[x['minclr'] for x in X]):+.3f}")
# episode level
print(f"\nEPISODE-LEVEL over {len(F)} failures: r(arrest_s, target_s) = {pear([r['max_s'] for r in F],[r['tgt_s'] for r in F]):+.3f}")
Fd=[r for r in F if r["max_s"]>60]
print(f"  excluding the 15 proximal (<60mm) stalls, n={len(Fd)}: r = {pear([r['max_s'] for r in Fd],[r['tgt_s'] for r in Fd]):+.3f}")
pickle.dump(recs,open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","wb"))
