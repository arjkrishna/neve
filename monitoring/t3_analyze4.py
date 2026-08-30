import pickle, math, statistics as st
from collections import defaultdict
D=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb")); eps=D["eps"]
SN=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_snap.pkl","rb"))
def pct(a,p):
    a=sorted(a)
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(a)-1); f=i-lo
    return a[lo]*(1-f)+a[hi]*f
blocks=defaultdict(list)
for k,e in eps.items():
    r=SN.get((e["pid"],int(e["ep"]))); e["res"]=r[0] if r else None; e["png"]=r[3] if r else None
    blocks[e["block"]].append(e)
DT=0.132
print("DT fit = 0.1320 s  (max |d_ins| 3.96 mm = 30 mm/s * 0.132)")
print("="*70); print("SATURATION OF THE VELOCITY LIMIT")
for b in (1,2,3):
    dg=[abs(s["dgw"]) for e in blocks[b] for s in e["steps"] if s["dgw"]==s["dgw"]]
    sat=sum(1 for v in dg if v>=3.90); sat99=sum(1 for v in dg if v>=3.95)
    print(f"eval{b}: median |d_ins_gw| = {pct(dg,50):.3f} mm/step = {pct(dg,50)/DT:.2f} mm/s ; mean = {st.mean(dg):.3f} mm = {st.mean(dg)/DT:.2f} mm/s ; max = {max(dg):.3f} mm = {max(dg)/DT:.2f} mm/s")
    print(f"        frac steps |d|>=3.90mm (>=29.5mm/s) = {sat/len(dg):.4f} ; >=3.95 = {sat99/len(dg):.4f}")
print()
print("="*70); print("DECISIVE TEST: within eval2/3, does per-step speed predict cross-track error?")
for b in (2,3):
    bins=defaultdict(lambda: ([],0,0))
    rows=[]
    for e in blocks[b]:
        if e["res"]!="success": continue
        S=sorted(e["steps"],key=lambda s:s["n"])
        for i,s in enumerate(S):
            d=abs(s["dgw"]);
            if d!=d: continue
            nxt=S[i+1] if i+1<len(S) else s
            x=abs(nxt["xt"]); lr=nxt["lr"]
            if x!=x or lr!=lr or lr<=0: continue
            k=min(int(d),3)
            xs,tot,ex=bins[k]; xs.append(x/lr); bins[k]=(xs,tot+1,ex+(1 if x>lr else 0))
    print(f"eval{b} (successes; bin = floor(|d_ins_gw| mm/step), xt of NEXT step):")
    for k in sorted(bins):
        xs,tot,ex=bins[k]
        lbl={0:"0-1 mm (0-7.6 mm/s)",1:"1-2 mm (7.6-15 mm/s)",2:"2-3 mm (15-23 mm/s)",3:">=3 mm (>=23 mm/s)"}[k]
        print(f"   {lbl:26s} n={tot:6d}  med |xt|/local_r={pct(xs,50):.3f}  p95={pct(xs,95):.3f}  frac>1 = {ex/tot:.4f}")
print()
print("="*70); print("EVAL1 (heuristic) COMMAND CEILING")
for b in (1,2):
    a=[abs(s["ains_gw"]) for e in blocks[b] for s in e["steps"] if s["ains_gw"]==s["ains_gw"]]
    print(f"eval{b}: |cmd_action[ins_gw]| med={pct(a,50):.2f} p90={pct(a,90):.2f} p99={pct(a,99):.2f} max={max(a):.2f} mm/s (env limit 30)")
print()
print("="*70); print("FAILURE EPISODES IN EVAL2/EVAL3")
for b in (2,3):
    for e in blocks[b]:
        if e["res"]=="max_steps":
            S=sorted(e["steps"],key=lambda s:s["n"])
            print(f"eval{b} {e['w']} pid={e['pid']} ep={e['ep']} mesh={e['mesh']} png={e['png']}")
            print("   proj_s every 25 steps:", [round(S[i]['ps'],1) for i in range(0,len(S),25)])
            print("   d_tgt every 25:", [round(S[i]['dtgt'],1) for i in range(0,len(S),25)])
print()
print("="*70); print("PNG PICKS")
for b in (1,2,3):
    L=[e for e in blocks[b] if e["res"]=="success"]
    L.sort(key=lambda e: len(e["steps"]))
    print(f"eval{b} shortest: {L[0]['png']} ({L[0]['mesh']})  longest: {L[-1]['png']} ({L[-1]['mesh']})")
