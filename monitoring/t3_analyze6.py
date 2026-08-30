import pickle, statistics as st
from collections import defaultdict
D=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb")); eps=D["eps"]
SN=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_snap.pkl","rb"))
def pct(a,p):
    a=sorted(a)
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i);hi=min(lo+1,len(a)-1);f=i-lo
    return a[lo]*(1-f)+a[hi]*f
blocks=defaultdict(list)
for k,e in eps.items():
    r=SN.get((e["pid"],int(e["ep"]))); e["res"]=r[0] if r else None
    blocks[e["block"]].append(e)
print("PER-STEP rates among SUCCESS episodes")
for b in (1,2,3):
    n=0;sl=0;bu=0;fo=0
    for e in blocks[b]:
        if e["res"]!="success": continue
        for s in e["steps"]:
            n+=1
            if s["slack"]==s["slack"] and s["slack"]>20: sl+=1
            if s["buck"]==s["buck"] and abs(s["buck"])>0.3: bu+=1
            try:
                if int(s["fold"].split("/")[0])>0: fo+=1
            except: pass
    print(f"eval{b}: steps={n} cath_slack>20mm {sl}({sl/n:.4f}) |buckle_phi|>0.3 {bu}({bu/n:.4f}) fold>0 {fo}({fo/n:.4f})")
print()
print("eval2/3 success episodes with long flat proj_s runs (>=10 steps):")
for b in (2,3):
    for e in blocks[b]:
        if e["res"]!="success": continue
        S=sorted(e["steps"],key=lambda s:s["n"]); prev=None;run=0;mx=0
        for s in S:
            if prev is not None and s["ps"]==s["ps"]:
                if abs(s["ps"]-prev)<0.05: run+=1
                else: mx=max(mx,run);run=0
            prev=s["ps"]
        mx=max(mx,run)
        if mx>=10: print(f"  eval{b} {e['w']} pid={e['pid']} ep={e['ep']} mesh={e['mesh']} steps={len(S)} longest_flat_run={mx}")
print()
print("terminal fine-approach: steps with |d_ins_gw|<1mm as frac of episode (successes)")
for b in (1,2,3):
    fr=[]
    for e in blocks[b]:
        if e["res"]!="success": continue
        S=e["steps"]; c=sum(1 for s in S if s["dgw"]==s["dgw"] and abs(s["dgw"])<1.0)
        fr.append(c/max(1,len(S)))
    print(f"eval{b}: med={pct(fr,50):.4f} mean={st.mean(fr):.4f} p90={pct(fr,90):.4f}")
