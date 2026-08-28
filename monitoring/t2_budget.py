import pickle,statistics as st
from collections import defaultdict,Counter
recs=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","rb"))
G=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_geo.pkl","rb"))["G"]
F=[r for r in recs if not r["succ"]]
print("=== J. TARGETS DISTAL TO A MID-VESSEL <0.30 BLOCKAGE (the only defensible 'unreachable' set) ===")
BLK={"topcowmr025":166.8,"topcowmr004":219.5}
tot=0;fl=0
for a,b in BLK.items():
    rs=[r for r in recs if r["mesh"]==a]
    past=[r for r in rs if r["tgt_s"]>b]; pre=[r for r in rs if r["tgt_s"]<=b]
    tot+=len(past); fl+=sum(1 for r in past if not r["succ"])
    print(f"  {a}  blk_s={b}:  targets PAST blockage n={len(past)} succ={sum(1 for r in past if r['succ'])}"
          f"  |  targets BEFORE n={len(pre)} succ={sum(1 for r in pre if r['succ'])}")
    print(f"      past-blockage target s: {sorted(round(r['tgt_s'],1) for r in past)}")
    print(f"      pre-blockage  target s: {sorted(round(r['tgt_s'],1) for r in pre)}")
print(f"  TOTAL episodes with target distal to a real mid-vessel blockage: {tot}  (failed {fl})")
print(f"  => corrected denominator 220-{tot}={220-tot}; corrected rate "
      f"{165}/{220-tot} = {100*165/(220-tot):.1f}%  (vs 75.0%)")

print("\n=== K. BUDGET DEFICIT: extrapolate the tail advance rate to the target ===")
T=[r for r in F if r["mode"]=="iii_timeout_advancing"]
print(f"{'mesh':<12}{'seed':>7}{'remain_mm':>10}{'rate_mm/step':>13}{'steps_needed':>13}{'cap':>6}{'x_over':>8}")
need=[]
for r in sorted(T,key=lambda x:-x["dproj100"]):
    rate=r["dproj100"]/100.0; rem=r["tgt_s"]-r["max_s"]
    n=rem/rate if rate>0 else float('inf'); need.append(n)
    print(f"{r['mesh']:<12}{r['seed']:>7}{rem:>10.1f}{rate:>13.3f}{n:>13.0f}{600:>6}{n/600:>8.2f}")
fin=[n for n in need if n!=float('inf')]
print(f"\n  n={len(T)} still-advancing timeouts. extra steps needed: min={min(fin):.0f} "
      f"median={st.median(fin):.0f} max={max(fin):.0f}")
for cap in (1200,1800,2400):
    print(f"   would finish within a {cap}-step cap (600 used + {cap-600} more): "
          f"{sum(1 for n in fin if n<=cap-600)}/{len(T)}")

print("\n=== L. MODAL-ARREST TIGHTNESS vs GEOMETRY (the signature test) ===")
byA=defaultdict(list)
for r in F: byA[r["mesh"]].append(r)
print(f"{'anat':<12}{'nf':>3}{'arrest sd':>11}{'span':>8}   {'mid-blk?':>9}  {'clr@modal_arrest':>18}")
for a in sorted(byA):
    d=[r["max_s"] for r in byA[a]]
    g=G[a]; L=g["L"]; m=[x for x in g["s030"] if x["s1"]<L-8.0]
    md=st.median(d)
    gi=min(range(len(g["g"])),key=lambda k:abs(g["g"][k]-md))
    print(f"{a:<12}{len(d):>3}{(st.pstdev(d) if len(d)>1 else 0):>11.1f}{max(d)-min(d):>8.1f}   "
          f"{('YES %.1f'%m[0]['s_at_min']) if m else 'no':>9}  {g['d'][gi]:>18.3f}")
