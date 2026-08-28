import os,re,glob,json,statistics as st
from collections import Counter,defaultdict
exec(open(r"d:/Arjun/workspace/neve/monitoring/attack3_e.py").read().split("vb=collect")[0])
V1BP=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
H0=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
vb=collect(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
h0=[r for r in collect(sorted(glob.glob(os.path.join(H0,"diagnostics","logs_subprocesses","worker_*.log"))),
    "2026-08-28 03:51:29","2026-08-28 04:15:20") if r["mesh"] in HOLD]
# H0 official rule: EPISODE_OUTCOME reason == success
for r in h0: r["succ_off"]= (r["outcome"]=="success")
for r in vb: r["succ_off"]= r["succ"]   # verified == official jsonl
print("H0 official-rule k=",sum(r["succ_off"] for r in h0),"/",len(h0))
def blocks(rows,key):
    d=defaultdict(lambda:[0,0])
    for r in rows: d[key(r)][0]+=1; d[key(r)][1]+=r["succ_off"]
    return d
OFF=33.4  # path_len(insertion) - RCCA arclength, upper bound from min path_len 73.4 with min_arclength=40
print("\nDerived insertion->ostium offset <= %.1f mm (min pooled path_len 73.4, min_arclength_from_start=40)"%OFF)
print("  section cut 146 -> arclength %.1f ; cut 210 -> arclength %.1f ; graft seam arclength 130 -> path_len %.1f"%(146-OFF,210-OFF,130+OFF))
SEAM=130+OFF
for tag,rows in (("V1BP",vb),("H0",h0)):
    pre=[r for r in rows if r["pl"] is not None and r["pl"]<SEAM]
    post=[r for r in rows if r["pl"] is not None and r["pl"]>=SEAM]
    print("%s  targets PRE-seam(host geometry): %d/%d = %.3f | POST-seam(graft): %d/%d = %.3f"%(
        tag,sum(r['succ_off'] for r in pre),len(pre),sum(r['succ_off'] for r in pre)/len(pre),
        sum(r['succ_off'] for r in post),len(post),sum(r['succ_off'] for r in post)/len(post)))
# stratified anatomy x section, common weights = pooled 196 episodes
cells=defaultdict(lambda:{"V1BP":[0,0],"H0":[0,0]})
for tag,rows in (("V1BP",vb),("H0",h0)):
    for r in rows:
        if r["pl"] is None: continue
        c=(r["mesh"],sec(r["pl"]))
        cells[c][tag][0]+=1; cells[c][tag][1]+=r["succ_off"]
tot=sum(sum(v[t][0] for t in ("V1BP","H0")) for v in cells.values())
sv=sh=w=0.0
for c,v in sorted(cells.items()):
    if v["V1BP"][0]==0 or v["H0"][0]==0: continue
    wt=v["V1BP"][0]+v["H0"][0]
    sv+=wt*v["V1BP"][1]/v["V1BP"][0]; sh+=wt*v["H0"][1]/v["H0"][0]; w+=wt
print("\nstratified (anatomy x section, cells present in both, weight=pooled n):")
print("  cells used=%d covering %d of %d episode-slots"%(sum(1 for v in cells.values() if v['V1BP'][0] and v['H0'][0]),int(w),tot))
print("  V1BP=%.4f  H0=%.4f  gap=%.1f pp"%(sv/w,sh/w,100*(sv-sh)/w))
for c,v in sorted(cells.items()):
    print("   %-14s %-8s V1BP %d/%-3d  H0 %d/%-3d"%(c[0],c[1],v["V1BP"][1],v["V1BP"][0],v["H0"][1],v["H0"][0]))
# direct vs recovered
print("\nV1BP successes: advance rate mm/step")
d=[(r["pl"]/r["steps"],r["steps"],r["pl"]) for r in vb if r["succ_off"]]
fast=[x for x in d if x[0]>=2.0]; slow=[x for x in d if x[0]<2.0]
print("  direct (>=2 mm/step): %d  steps p50=%d max=%d"%(len(fast),sorted(x[1] for x in fast)[len(fast)//2],max(x[1] for x in fast)))
print("  slow   (<2 mm/step) : %d  steps p50=%d max=%d  pl p50=%.0f"%(len(slow),sorted(x[1] for x in slow)[len(slow)//2],max(x[1] for x in slow),sorted(x[2] for x in slow)[len(slow)//2]))
print("  steps>200: %d of %d successes; >400: %d; >500: %d"%(sum(1 for x in d if x[1]>200),len(d),sum(1 for x in d if x[1]>400),sum(1 for x in d if x[1]>500)))
# section of the long successes
print("  long(>200 steps) successes by section:",Counter(sec(x[2]) for x in d if x[1]>200))
print("  long(>200) by anatomy:",Counter(r["mesh"] for r in vb if r["succ_off"] and r["steps"]>200))
# max-steps failures detail
print("\nV1BP max_steps failures:")
for r in vb:
    if not r["succ_off"]:
        print("   seed=%s mesh=%s pl=%.1f sec=%s final d_tgt=%s proj_s=%s"%(r["seed"],r["mesh"],r["pl"],sec(r["pl"]),r["last"]["d_tgt"],r["last"]["proj_s"]))
