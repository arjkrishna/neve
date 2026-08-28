import pickle,statistics as st
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
OFF=33.314
F=[r for r in recs if not r["succ"]]
DT=0.132  # empirical dins/cmd scale
for r in recs:
    S=r["S"]; n=len(S)
    for W in (50,100):
        seg=S[-W:]
        r[f"dproj{W}"]=seg[-1]["proj_s"]-seg[0]["proj_s"]
        r[f"dins0_{W}"]=seg[-1]["ins"][0]-seg[0]["ins"][0]
        r[f"dins1_{W}"]=seg[-1]["ins"][1]-seg[0]["ins"][1]
        r[f"absins0_{W}"]=sum(abs(s["dins"][0]) for s in seg)
        r[f"absins1_{W}"]=sum(abs(s["dins"][1]) for s in seg)
        r[f"cmdpos0_{W}"]=sum(max(s["cmd"][0],0) for s in seg)*DT
        r[f"cmdpos1_{W}"]=sum(max(s["cmd"][2],0) for s in seg)*DT
    r["offbr_frac"]=sum(1 for s in S if s["off_br"]==1)/n
    r["onpath_frac"]=sum(1 for s in S if s["on_path"]==1)/n
    r["frac_at_max"]=r["imax"]/n
    r["retreat"]=r["max_proj"]-S[-1]["proj_s"]

print("=== FAILURE MODE CLASSIFICATION (window = last 100 steps) ===")
print(f"{'mesh':<12}{'seed':>7}{'tgt_s':>7}{'max_s':>7}{'fin_s':>7}{'dproj100':>9}{'cmdIns':>8}{'gotIns':>8}{'|ins|':>7}{'off%':>6}{'@max':>6}{'retr':>7}  mode")
def classify(r):
    dp=r["dproj100"]; cmd=r["cmdpos0_100"]+r["cmdpos1_100"]; got=abs(r["dins0_100"])+abs(r["dins1_100"])
    absmv=r["absins0_100"]+r["absins1_100"]
    if r["offbr_frac"]>0.10: return "iv_offpath"
    if dp>3.0: return "iii_timeout_advancing"
    if cmd>8.0 and dp<1.0 and (r["dins0_100"]+r["dins1_100"])>2.0: return "i_wall_feed_no_advance"
    if absmv>8.0 and dp<3.0: return "ii_oscillate"
    return "v_idle_low_command"
cnt=Counter()
for r in sorted(F,key=lambda x:(x["mesh"],x["seed"])):
    m=classify(r); cnt[m]+=1; r["mode"]=m
    print(f"{r['mesh']:<12}{r['seed']:>7}{r['tgt_s']:>7.1f}{r['max_s']:>7.1f}{r['final_s']:>7.1f}{r['dproj100']:>9.1f}"
          f"{r['cmdpos0_100']+r['cmdpos1_100']:>8.1f}{r['dins0_100']+r['dins1_100']:>8.1f}"
          f"{r['absins0_100']+r['absins1_100']:>7.1f}{100*r['offbr_frac']:>6.1f}{100*r['frac_at_max']:>6.0f}{r['retreat']:>7.1f}  {m}")
print()
for k,v in cnt.most_common(): print(f"  {k:<26}{v:>4}  ({100*v/len(F):.1f}%)")
pickle.dump(recs,open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","wb"))
