import pickle, math, statistics as st
from collections import defaultdict
D = pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb"))
eps = D["eps"]

def pct(a,p):
    a=sorted(a); 
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(a)-1); f=i-lo
    return a[lo]*(1-f)+a[hi]*f

blocks = defaultdict(list)
for k,e in eps.items(): blocks[e["block"]].append(e)

def outcome_ok(e):
    o=e["outcome"]
    if o is None: return None
    return o.get("grader_success")=="1"

print("="*70)
print("EPISODE COUNTS / OUTCOMES")
for b in (1,2,3):
    E=blocks[b]; n=len(E)
    hav=[e for e in E if e["outcome"]]
    succ=[e for e in hav if outcome_ok(e)]
    print(f"eval{b}: eps={n} outcome_lines={len(hav)} (shortfall {n-len(hav)}) success_among_logged={len(succ)}/{len(hav)}={len(succ)/max(1,len(hav)):.3f}")
    # steps distribution
    sl=[len(e["steps"]) for e in E]
    print(f"   steps/ep: mean={st.mean(sl):.1f} med={pct(sl,50):.0f} p10={pct(sl,10):.0f} p90={pct(sl,90):.0f} max={max(sl)}")

print()
print("="*70)
print("1. CROSS-TRACK: xt_true / cross_tr / xt_true vs local_r  (ALL eval steps)")
for b in (1,2,3):
    xs=[];xr=[];ratio=[];exceed=0;tot=0;exceed_tol=0
    for e in blocks[b]:
        for s in e["steps"]:
            x=s["xt"]; lr=s["lr"]; tl=s["tol"]
            if x==x: xs.append(abs(x))
            if s["xtr"]==s["xtr"]: xr.append(abs(s["xtr"]))
            if x==x and lr==lr and lr>0:
                tot+=1; ratio.append(abs(x)/lr)
                if abs(x)>lr: exceed+=1
                if tl==tl and tl>0 and abs(x)>tl: exceed_tol+=1
    print(f"eval{b}: n={len(xs)}")
    print(f"   |xt_true| mean={st.mean(xs):.2f} med={pct(xs,50):.2f} p75={pct(xs,75):.2f} p90={pct(xs,90):.2f} p99={pct(xs,99):.2f} max={max(xs):.2f}")
    print(f"   |cross_tr| mean={st.mean(xr):.2f} med={pct(xr,50):.2f} p90={pct(xr,90):.2f} p99={pct(xr,99):.2f} max={max(xr):.2f}")
    print(f"   |xt|/local_r mean={st.mean(ratio):.3f} med={pct(ratio,50):.3f} p90={pct(ratio,90):.3f} p99={pct(ratio,99):.3f}")
    print(f"   frac steps |xt_true|>local_r = {exceed}/{tot} = {exceed/tot:.4f}   |xt_true|>tol = {exceed_tol}/{tot} = {exceed_tol/tot:.4f}")

# per-episode max xt
print()
print("   per-episode MAX |xt_true|:")
for b in (1,2,3):
    m=[max([abs(s["xt"]) for s in e["steps"] if s["xt"]==s["xt"]] or [float('nan')]) for e in blocks[b]]
    m=[v for v in m if v==v]
    print(f"   eval{b}: med={pct(m,50):.2f} p90={pct(m,90):.2f} max={max(m):.2f}")

# successes only
print()
print("   SUCCESSES ONLY (grader_success=1):")
for b in (1,2,3):
    xs=[];exceed=0;tot=0
    for e in blocks[b]:
        if not outcome_ok(e): continue
        for s in e["steps"]:
            x=s["xt"];lr=s["lr"]
            if x==x: xs.append(abs(x))
            if x==x and lr==lr and lr>0:
                tot+=1
                if abs(x)>lr: exceed+=1
    if tot: print(f"   eval{b}: n={tot} med|xt|={pct(xs,50):.2f} p90={pct(xs,90):.2f} p99={pct(xs,99):.2f} max={max(xs):.2f} frac>local_r={exceed/tot:.4f}")

print()
print("="*70)
print("3. OVERSHOOT")
for b in (1,2,3):
    stot=0; sover=0; epover=0; n=0
    for e in blocks[b]:
        n+=1; any_o=False
        for s in e["steps"]:
            stot+=1
            if s["over"]: sover+=1; any_o=True
        if any_o: epover+=1
    print(f"eval{b}: steps_with_overshoot={sover}/{stot}={sover/stot:.4f}  episodes_with_any_overshoot={epover}/{n}={epover/n:.3f}")
    # also outcome field overshoot=
    ov=[e["outcome"].get("overshoot") for e in blocks[b] if e["outcome"]]
    print(f"        EPISODE_OUTCOME overshoot=1 : {sum(1 for v in ov if v=='1')}/{len(ov)}")

print()
print("="*70)
print("5. INSERTION PER STEP (delta_ins) and speed reconciliation")
for b in (1,2,3):
    dg=[];dc=[];dmax=[];tl=[];sl=[]
    for e in blocks[b]:
        S=e["steps"]
        for s in S:
            if s["dgw"]==s["dgw"]: dg.append(s["dgw"])
            if s["dcath"]==s["dcath"]: dc.append(s["dcath"])
            if s["dgw"]==s["dgw"] and s["dcath"]==s["dcath"]: dmax.append(max(abs(s["dgw"]),abs(s["dcath"])))
        if S:
            tl.append(max(s["igw"] for s in S if s["igw"]==s["igw"]))
            sl.append(len(S))
    print(f"eval{b}: n={len(dg)}")
    print(f"   d_ins_gw   : med={pct(dg,50):.3f} mean={st.mean(dg):.3f} p90={pct(dg,90):.3f} p99={pct(dg,99):.3f} max={max(dg):.3f} min={min(dg):.3f} frac_neg={sum(1 for v in dg if v<-1e-6)/len(dg):.3f} frac_zero={sum(1 for v in dg if abs(v)<1e-6)/len(dg):.3f}")
    print(f"   d_ins_cath : med={pct(dc,50):.3f} mean={st.mean(dc):.3f} p90={pct(dc,90):.3f} max={max(dc):.3f} frac_neg={sum(1 for v in dc if v<-1e-6)/len(dc):.3f}")
    print(f"   max(|dgw|,|dcath|): med={pct(dmax,50):.3f} mean={st.mean(dmax):.3f} p90={pct(dmax,90):.3f}")
    print(f"   final inserted_gw: med={pct(tl,50):.1f} mean={st.mean(tl):.1f}")
    # candidate speed metric: total |delta| path / (steps * dt)
    print(f"   mean steps={st.mean(sl):.1f}")

print()
print("   cmd_action insertion channel (raw units, gw / cath):")
for b in (1,2,3):
    ag=[];ac=[]
    for e in blocks[b]:
        for s in e["steps"]:
            if s["ains_gw"]==s["ains_gw"]: ag.append(s["ains_gw"])
            if s["ains_cath"]==s["ains_cath"]: ac.append(s["ains_cath"])
    print(f"   eval{b}: a_ins_gw med={pct(ag,50):.3f} mean={st.mean(ag):.3f} p90={pct(ag,90):.3f} max={max(ag):.3f} min={min(ag):.3f} | a_ins_cath med={pct(ac,50):.3f} mean={st.mean(ac):.3f} max={max(ac):.3f}")
