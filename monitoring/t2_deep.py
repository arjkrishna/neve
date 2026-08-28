import pickle,json,statistics as st
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
G=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_geo.pkl","rb"))["G"]
OFF=33.314
print("=== F. BAND CHECK (path_len bands, s_RCCA = path_len-33.31) ===")
bands=[(0,167,"pre-seam"),(167,200,"167-200"),(200,240,"200-240"),(240,1e9,">240")]
for lo,hi,nm in bands:
    b=[r for r in recs if lo<=r["pl"]<hi]
    s=sum(1 for r in b if r["succ"])
    print(f"  path_len {nm:<9} n={len(b):>3} succ={s:>3} ({100*s/len(b):.1f}%)")
DEEP=[r for r in recs if r["pl"]>240]
DF=[r for r in DEEP if not r["succ"]]
print(f"\n=== G. DEEP BAND path_len>240 (tgt_s>{240-OFF:.1f}): n={len(DEEP)} succ={len(DEEP)-len(DF)} fail={len(DF)} ===")
print(f"{'mesh':<12}{'seed':>7}{'mode':<24}{'tgt_s':>7}{'arr_s':>7}{'short':>7}{'clr@arr':>8}{'L':>7}{'L-arr':>7}{'L-tgt':>7}{'cap030':>7}{'dproj100':>9}")
capinfo={}
for a,g in G.items():
    L=g["L"]; t=[r for r in g["s030"] if r["s1"]>=L-8.0]
    capinfo[a]=(L-min(r["s0"] for r in t)) if t else 0.0
def prof(a,s):
    g=G[a]["g"]; d=G[a]["d"]
    i=min(range(len(g)),key=lambda k:abs(g[k]-s)); return d[i]
mc=Counter()
for r in sorted(DF,key=lambda x:(x["mesh"],x["seed"])):
    L=G[r["mesh"]]["L"]; mc[r["mode"]]+=1
    print(f"{r['mesh']:<12}{r['seed']:>7}{r['mode']:<24}{r['tgt_s']:>7.1f}{r['max_s']:>7.1f}"
          f"{r['tgt_s']-r['max_s']:>7.1f}{prof(r['mesh'],r['max_s']):>8.3f}{L:>7.1f}{L-r['max_s']:>7.1f}"
          f"{L-r['tgt_s']:>7.1f}{capinfo[r['mesh']]:>7.2f}{r['dproj100']:>9.1f}")
print("\n deep-band failure modes:")
for k,v in mc.most_common(): print(f"   {k:<26}{v:>3} ({100*v/len(DF):.0f}%)")
d=[L for L in (G[r["mesh"]]["L"]-r["max_s"] for r in DF)]
print(f"\n distance from arrest to branch TERMINUS: min={min(d):.1f} med={st.median(d):.1f} max={max(d):.1f} mm")
print(f" arrests within the eroded end-cap (cap_030 extent, max 6.9mm anywhere): "
      f"{sum(1 for r in DF if (G[r['mesh']]['L']-r['max_s'])<=max(capinfo[r['mesh']],0.5))}")
print(f" arrests at a real mid-vessel <0.30 obstruction: "
      f"{sum(1 for r in DF if prof(r['mesh'],r['max_s'])<0.30)}")
print(f" still advancing at cap (dproj over last 100 steps >3mm): {sum(1 for r in DF if r['dproj100']>3.0)}")
print(f" shortfall tgt_s - arrest_s: min={min(r['tgt_s']-r['max_s'] for r in DF):.1f} "
      f"med={st.median([r['tgt_s']-r['max_s'] for r in DF]):.1f} max={max(r['tgt_s']-r['max_s'] for r in DF):.1f}")

print("\n=== H. HOW CLOSE DO SUCCESSES GET TO THE TERMINUS? (upper bound on real reach) ===")
SU=[r for r in recs if r["succ"]]
byA=defaultdict(list)
for r in SU: byA[r["mesh"]].append(r)
print(f"{'anat':<12}{'L':>7}{'max_tgt_s_SOLVED':>18}{'L-that':>8}{'max_arrest_any':>16}{'nsucc':>7}")
for a in sorted(byA):
    L=G[a]["L"]; mt=max(r["tgt_s"] for r in byA[a])
    ma=max(r["max_s"] for r in recs if r["mesh"]==a)
    print(f"{a:<12}{L:>7.1f}{mt:>18.1f}{L-mt:>8.1f}{ma:>16.1f}{len(byA[a]):>7}")

print("\n=== I. FINAL MODE COUNTS (all 55 failures; every one truncated at the 600-step cap) ===")
cnt=Counter(r["mode"] for r in recs if not r["succ"])
lbl={"iii_timeout_advancing":"(iii) TIMEOUT - still advancing at cap",
     "i_wall_feed_no_advance":"(i)   feed-without-tip-advance (buckle/stall)",
     "ii_oscillate":"(ii)  oscillating / retracting",
     "v_idle_low_command":"(v)   idle - policy commands ~no insertion",
     "iv_offpath":"(iv)  off-path / wrong branch"}
for k in ["iii_timeout_advancing","i_wall_feed_no_advance","ii_oscillate","v_idle_low_command","iv_offpath"]:
    print(f"   {lbl[k]:<48}{cnt.get(k,0):>3}  ({100*cnt.get(k,0)/55:.1f}%)")
print(f"\n   proximal stalls (arrest s_RCCA < 60 mm, target 130-215 mm away): "
      f"{sum(1 for r in recs if not r['succ'] and r['max_s']<60)}")
