import pickle, math, statistics as st, os, re
from collections import defaultdict
D=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb")); eps=D["eps"]
TIP=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_tip.pkl","rb"))
SN=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_snap.pkl","rb"))
def pct(a,p):
    a=sorted(a)
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(a)-1); f=i-lo
    return a[lo]*(1-f)+a[hi]*f
blocks=defaultdict(list)
for k,e in eps.items():
    r=SN.get((e["pid"],int(e["ep"])))
    e["res"]= r[0] if r else None
    e["key"]=k
    blocks[e["block"]].append(e)

print("="*70); print("TRAJECTORY LENGTH / SPEED RECONCILIATION (CSV: eval1 236.771 @3.64 ; eval2 225.589 @25.725 ; eval3 225.344 @25.877)")
CSV={1:(236.771,3.64,497.439),2:(225.589,25.725,68.214),3:(225.344,25.877,71.398)}
for b in (1,2,3):
    L3=[];Lins=[];SP=[];nst=[]
    for e in blocks[b]:
        S=TIP.get((e["w"],e["pid"],int(e["ep"]),e["block"]),[])
        S=sorted(S,key=lambda x:x[0])
        d=0
        for i in range(1,len(S)):
            a,bb=S[i-1][1],S[i][1]
            if a and bb: d+=math.dist(a,bb)
        L3.append(d); nst.append(len(S))
        ins=sum(abs(s["dgw"]) for s in e["steps"] if s["dgw"]==s["dgw"]); Lins.append(ins)
    tl,sp,stp=CSV[b]
    print(f"eval{b}: mean 3D tip path = {st.mean(L3):.3f} mm  | mean sum|d_ins_gw| = {st.mean(Lins):.3f} mm | CSV traj_len={tl}")
    # implied dt per episode
    dts=[l/(n*sp) for l,n in zip(L3,nst) if n]
    print(f"        CSV traj_len/(steps) = {tl/stp:.4f} mm/step ; /CSV_speed = dt = {tl/stp/sp:.5f} s")
    print(f"        mean(3D_len/steps)   = {st.mean([l/n for l,n in zip(L3,nst) if n]):.4f} mm/step ; mean per-ep speed at dt=0.1300: {st.mean([l/(n*0.13) for l,n in zip(L3,nst) if n]):.3f}")
    print(f"        mean(ins_len/steps)  = {st.mean([l/n for l,n in zip(Lins,nst) if n]):.4f} mm/step ; at dt=0.1300: {st.mean([l/(n*0.13) for l,n in zip(Lins,nst) if n]):.3f}")

print(); print("="*70); print("1b. XT_TRUE BY SNAPSHOT-TRUTH OUTCOME")
for b in (1,2,3):
    for res in ("success","max_steps"):
        xs=[];ex=0;tot=0;exT=0
        E=[e for e in blocks[b] if e["res"]==res]
        for e in E:
            for s in e["steps"]:
                x=s["xt"];lr=s["lr"];tl=s["tol"]
                if x!=x: continue
                xs.append(abs(x))
                if lr==lr and lr>0:
                    tot+=1
                    if abs(x)>lr: ex+=1
                    if tl==tl and tl>0 and abs(x)>tl: exT+=1
        if not xs: continue
        print(f"eval{b} {res:9s} eps={len(E):3d} steps={len(xs):6d} |xt| med={pct(xs,50):.2f} p90={pct(xs,90):.2f} p99={pct(xs,99):.2f} max={max(xs):.2f} | >local_r {ex/tot:.4f} | >tol {exT/tot:.4f}")

print(); print("="*70); print("1c. XT_TRUE RESTRICTED TO SIPHON (proj_s such that s_RCCA=path_len-33.31 ; use arc past seam: proj_s>133.6+33.31=166.9)")
for b in (1,2,3):
    xs=[];ex=0;tot=0
    for e in blocks[b]:
        if e["res"]!="success": continue
        for s in e["steps"]:
            if s["ps"]!=s["ps"] or s["ps"]<166.9: continue
            x=s["xt"];lr=s["lr"]
            if x!=x: continue
            xs.append(abs(x))
            if lr==lr and lr>0:
                tot+=1
                if abs(x)>lr: ex+=1
    if xs: print(f"eval{b} successes, distal (proj_s>166.9mm): n={len(xs)} med={pct(xs,50):.2f} p90={pct(xs,90):.2f} p99={pct(xs,99):.2f} max={max(xs):.2f} frac>local_r={ex/tot:.4f}")

print(); print("="*70); print("2b. ADVANCE PROFILE, snapshot-truth successes")
for b in (1,2,3):
    dps=[];flat=0;tot=0;runs=[];back=0
    for e in blocks[b]:
        if e["res"]!="success": continue
        S=sorted(e["steps"],key=lambda s:s["n"]); prev=None;run=0;mx=0
        for s in S:
            if prev is not None and s["ps"]==s["ps"]:
                d=s["ps"]-prev; dps.append(d); tot+=1
                if abs(d)<0.05: flat+=1;run+=1
                else: mx=max(mx,run);run=0
                if d<-0.05: back+=1
            prev=s["ps"]
        runs.append(max(mx,run))
    print(f"eval{b}: d_proj_s med={pct(dps,50):.3f} p10={pct(dps,10):.3f} p90={pct(dps,90):.3f} max={max(dps):.3f} min={min(dps):.3f} CV={st.pstdev(dps)/st.mean(dps):.3f}")
    print(f"        flat(<0.05mm) {flat/tot:.4f} backward(<-0.05) {back/tot:.4f} ; longest flat run per ep med={pct(runs,50):.0f} p90={pct(runs,90):.0f} max={max(runs)}")

print(); print("="*70); print("ANOMALOUS EVAL3 EPISODES (fold/cath_slack extremes)")
for b in (2,3):
    for e in blocks[b]:
        mf=0;ms=0
        for s in e["steps"]:
            try: mf=max(mf,int(s["fold"].split("/")[0]))
            except: pass
            if s["slack"]==s["slack"]: ms=max(ms,s["slack"])
        if mf>10 or ms>60:
            print(f"eval{b} {e['w']} pid={e['pid']} ep={e['ep']} mesh={e['mesh']} steps={len(e['steps'])} res={e['res']} maxfold={mf} max_slack={ms:.1f}")
