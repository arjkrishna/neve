import pickle, math, statistics as st
from collections import defaultdict
D = pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb"))
eps = D["eps"]
def pct(a,p):
    a=sorted(a)
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(a)-1); f=i-lo
    return a[lo]*(1-f)+a[hi]*f
blocks=defaultdict(list)
for k,e in eps.items(): blocks[e["block"]].append(e)
def ok(e): return e["outcome"] is not None and e["outcome"].get("grader_success")=="1"

print("="*70); print("5b. SPEED-COLUMN RECONCILIATION")
for b in (1,2,3):
    absd=[]; traj=[]; nst=[]; ratios=[]
    for e in blocks[b]:
        S=e["steps"]; s_abs=0
        for s in S:
            if s["dgw"]==s["dgw"]:
                absd.append(abs(s["dgw"])); s_abs+=abs(s["dgw"])
                if abs(s["ains_gw"])>1e-6: ratios.append(abs(s["dgw"])/abs(s["ains_gw"]))
        traj.append(s_abs); nst.append(len(S))
    mn=st.mean(absd)
    print(f"eval{b}: mean|d_ins_gw| per step = {mn:.4f} mm ; mean sum|d_ins_gw| per ep = {st.mean(traj):.2f} mm ; mean steps = {st.mean(nst):.1f}")
    print(f"        implied dt from CSV: (mean|d|)/CSV_speed ; realized/commanded ratio med={pct(ratios,50):.4f} mean={st.mean(ratios):.4f} p10={pct(ratios,10):.4f} p90={pct(ratios,90):.4f}")

print(); print("="*70); print("2. ADVANCE PROFILE proj_s vs ep_step  (successes)")
for b in (1,2):
    E=[e for e in blocks[b] if ok(e)]
    dps=[]; flat=0; tot=0; mono=0; nback=[]; jumps=[]
    per_ep_flatrun=[]
    for e in E:
        S=sorted(e["steps"], key=lambda s:s["n"])
        prev=None; run=0; maxrun=0; nb=0
        for s in S:
            if prev is not None and s["ps"]==s["ps"] and prev==prev:
                d=s["ps"]-prev; dps.append(d); tot+=1
                if abs(d)<0.05: flat+=1; run+=1
                else:
                    maxrun=max(maxrun,run); run=0
                if d< -0.05: nb+=1
            prev=s["ps"]
        maxrun=max(maxrun,run)
        per_ep_flatrun.append(maxrun); nback.append(nb/max(1,len(S)))
    print(f"eval{b}: n_succ_eps={len(E)} n_deltas={tot}")
    print(f"   d(proj_s)/step: med={pct(dps,50):.3f} mean={st.mean(dps):.3f} p10={pct(dps,10):.3f} p90={pct(dps,90):.3f} min={min(dps):.3f} max={max(dps):.3f}")
    print(f"   frac |d|<0.05mm (flat) = {flat/tot:.4f} ; frac d<-0.05 (backward) = {sum(1 for d in dps if d<-0.05)/tot:.4f}")
    print(f"   longest flat RUN per episode: med={pct(per_ep_flatrun,50):.0f} p90={pct(per_ep_flatrun,90):.0f} max={max(per_ep_flatrun)}")
    # coefficient of variation of forward progress
    fwd=[d for d in dps if d>0]
    print(f"   forward d(proj_s): med={pct(fwd,50):.3f} CV={st.pstdev(dps)/abs(st.mean(dps)):.3f}")
    # normalized: how much of s-range covered in first/last thirds
    fr=[]
    for e in E:
        S=sorted(e["steps"], key=lambda s:s["n"])
        if len(S)<9: continue
        ps=[s["ps"] for s in S if s["ps"]==s["ps"]]
        if len(ps)<9: continue
        tot_s=ps[-1]-ps[0]
        if tot_s<=0: continue
        t3=len(ps)//3
        fr.append(((ps[t3]-ps[0])/tot_s, (ps[2*t3]-ps[t3])/tot_s, (ps[-1]-ps[2*t3])/tot_s))
    a=[x[0] for x in fr]; bb=[x[1] for x in fr]; c=[x[2] for x in fr]
    print(f"   fraction of arclength covered in thirds of episode: {st.mean(a):.3f} / {st.mean(bb):.3f} / {st.mean(c):.3f}")

print(); print("="*70); print("SAMPLE proj_s TRACES (eval2 successes, longest & shortest)")
E2=sorted([e for e in blocks[2] if ok(e)], key=lambda e: len(e["steps"]))
for e in [E2[0], E2[len(E2)//2], E2[-1]]:
    S=sorted(e["steps"], key=lambda s:s["n"])
    ps=[round(s["ps"],1) for s in S]
    print(f" {e['w']} pid={e['pid']} ep={e['ep']} mesh={e['mesh']} steps={len(S)} path_len={S[0]['pl']:.1f}")
    print("   proj_s:", ps)
E1=sorted([e for e in blocks[1] if ok(e)], key=lambda e: len(e["steps"]))
print(" -- eval1 median-length success (subsampled every 10 steps):")
e=E1[len(E1)//2]; S=sorted(e["steps"], key=lambda s:s["n"])
print(f" {e['w']} pid={e['pid']} ep={e['ep']} mesh={e['mesh']} steps={len(S)}")
print("   proj_s:", [round(S[i]['ps'],1) for i in range(0,len(S),10)])

print(); print("="*70); print("MECHANICAL LOAD PROXIES")
for b in (1,2,3):
    bu=[];sl=[];fo=[];op=0;tot=0;fmax=[]
    for e in blocks[b]:
        mf=0
        for s in e["steps"]:
            tot+=1
            if s["onpath"]=="1": op+=1
            if s["buck"]==s["buck"]: bu.append(abs(s["buck"]))
            if s["slack"]==s["slack"]: sl.append(s["slack"])
            try:
                f=int(s["fold"].split("/")[0]); fo.append(f); mf=max(mf,f)
            except: pass
        fmax.append(mf)
    print(f"eval{b}: on_path frac={op/tot:.4f} | |buckle_phi| med={pct(bu,50):.3f} p90={pct(bu,90):.3f} max={max(bu):.3f}")
    print(f"        cath_slack med={pct(sl,50):.2f} p90={pct(sl,90):.2f} max={max(sl):.2f} min={min(sl):.2f}")
    print(f"        fold count med={pct(fo,50):.1f} p99={pct(fo,99):.1f} max={max(fo)} | per-ep max fold: med={pct(fmax,50):.0f} p90={pct(fmax,90):.0f} max={max(fmax)}")
