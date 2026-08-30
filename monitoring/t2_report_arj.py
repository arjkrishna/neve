import json,collections,math,statistics as st
rows=json.load(open(r"D:/Arjun/workspace/neve/monitoring/_t2_eval_rows.json"))

# impute outcome for the 48 missing (last-per-worker) episodes from term/trunc
for r in rows:
    if r["reason"]:
        r["succ"]=int(r["grader_success"]); r["src"]="log"
    else:
        r["succ"]=1 if r["last_term"]=="True" else 0
        r["reason"]="success" if r["succ"] else "max_steps"
        r["src"]="imputed"
    r["steps_final"]=int(r["out_steps"]) if r["out_steps"] else r["n_steps"]
    r["ret_final"]=float(r["out_return"]) if r["out_return"] else r["cum_reward"]
    r["seed"]=int(r["seed"])

def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))

byev=collections.defaultdict(list)
for r in rows: byev[r["eval"]].append(r)

print("="*70); print("1. PER-ANATOMY SUCCESS (n=98 per eval; 82 logged + 16 imputed from term/trunc)")
for ev in (1,2,3):
    R=byev[ev]; k=sum(r["succ"] for r in R); n=len(R); lo,hi=wilson(k,n)
    print(f"\n-- eval {ev}: overall {k}/{n} = {100*k/n:.1f}%  Wilson95 [{lo:.1f},{hi:.1f}]")
    bym=collections.defaultdict(list)
    for r in R: bym[r["mesh"]].append(r)
    for m in sorted(bym):
        v=bym[m]; kk=sum(x["succ"] for x in v); nn=len(v); a,b=wilson(kk,nn)
        print(f"   {m:14s} {kk:3d}/{nn:<3d} = {100*kk/nn:5.1f}%  Wilson95 [{a:5.1f},{b:5.1f}]")

print()
print("="*70); print("2. TARGET DEPTH (path_len, mm) per eval; seam at path_len=166.91")
for ev in (1,2,3):
    R=byev[ev]; pl=sorted(r["path_len"] for r in R if r["path_len"] is not None)
    nvar=sum(1 for r in R if len(r["path_lens"])>1)
    below=[r for r in R if r["path_len"] is not None and r["path_len"]<166.91]
    print(f" eval {ev}: n={len(pl)} min={min(pl):.1f} p25={st.quantiles(pl,n=4)[0]:.1f} median={st.median(pl):.1f} p75={st.quantiles(pl,n=4)[2]:.1f} max={max(pl):.1f}  | episodes with path_len<166.91: {len(below)} | eps w/ non-constant path_len: {nvar}")
    print(f"          s_RCCA = path_len-33.31 -> min={min(pl)-33.31:.1f} median={st.median(pl)-33.31:.1f} max={max(pl)-33.31:.1f}")

print()
print("="*70); print("3. SEED MATCHING ACROSS EVALS")
seeds={ev:{r["seed"]:r for r in byev[ev]} for ev in (1,2,3)}
for ev in (1,2,3):
    print(f" eval {ev}: {len(byev[ev])} episodes, {len(seeds[ev])} distinct seeds")
s1=set(seeds[1]); s2=set(seeds[2]); s3=set(seeds[3])
print(" seed sets identical 1==2==3:", s1==s2==s3, "| |1|,|2|,|3| =",len(s1),len(s2),len(s3))
mism_a=[]; mism_t=[]; mism_p=[]
for s in sorted(s1&s2&s3):
    A=[seeds[e][s] for e in (1,2,3)]
    if len({a["mesh"] for a in A})>1: mism_a.append(s)
    if len({a["target"] for a in A})>1: mism_t.append(s)
    if len({round(a["path_len"],1) for a in A})>1: mism_p.append(s)
print(" seeds w/ anatomy mismatch:",len(mism_a),mism_a[:10])
print(" seeds w/ target-coord mismatch:",len(mism_t),mism_t[:10])
print(" seeds w/ path_len mismatch:",len(mism_p),mism_p[:10])
print(" anatomy is pure fn of seed:", len(mism_a)==0)

print("\n PAIRED TRANSITIONS eval1 -> eval2 (n=%d)"%len(s1&s2))
tr=collections.Counter((seeds[1][s]["succ"],seeds[2][s]["succ"]) for s in s1&s2)
print("  fail->success:",tr[(0,1)]," success->fail:",tr[(1,0)]," success->success:",tr[(1,1)]," fail->fail:",tr[(0,0)])
n01,n10=tr[(0,1)],tr[(1,0)]
if n01+n10>0:
    chi=(abs(n01-n10)-1)**2/(n01+n10)
    print("  McNemar exact-ish: b=%d c=%d  chi2_cc=%.2f"%(n01,n10,chi))
    # exact binomial two-sided
    from math import comb
    n=n01+n10; k=min(n01,n10)
    p=2*sum(comb(n,i) for i in range(0,k+1))/2**n
    print("  McNemar exact two-sided p = %.3g"%min(p,1.0))
print("\n PAIRED TRANSITIONS eval2 -> eval3")
tr23=collections.Counter((seeds[2][s]["succ"],seeds[3][s]["succ"]) for s in s2&s3)
print("  fail->success:",tr23[(0,1)]," success->fail:",tr23[(1,0)]," success->success:",tr23[(1,1)]," fail->fail:",tr23[(0,0)])
print("\n PAIRED eval1 -> eval3")
tr13=collections.Counter((seeds[1][s]["succ"],seeds[3][s]["succ"]) for s in s1&s3)
print("  fail->success:",tr13[(0,1)]," success->fail:",tr13[(1,0)]," succ->succ:",tr13[(1,1)]," fail->fail:",tr13[(0,0)])

print()
print("="*70); print("4. FAILURES")
for ev in (1,2,3):
    F=[r for r in byev[ev] if not r["succ"]]
    print(f"\n eval {ev}: {len(F)} failures")
    for r in sorted(F,key=lambda x:x["seed"]):
        print(f"   seed={r['seed']:4d} {r['mesh']:12s} path_len={r['path_len']:6.1f} s_RCCA={r['path_len']-33.31:6.1f} reason={r['reason']:9s} steps={r['steps_final']:4d} max_proj_s={r['max_proj_s']:6.1f} last_proj_s={r['last_proj_s']:6.1f} last_d_tgt={r['last_d_tgt']:5.1f} ret={r['ret_final']:.3f} frac_path={r['max_proj_s']/r['path_len']:.3f} src={r['src']}")

print()
print("="*70); print("5. RETURN / STEP DISTRIBUTIONS")
for ev in (1,2,3):
    R=byev[ev]
    rets=sorted(r["ret_final"] for r in R); stp=sorted(r["steps_final"] for r in R)
    sr=sorted(r["ret_final"] for r in R if r["succ"]); fr=sorted(r["ret_final"] for r in R if not r["succ"])
    ss=sorted(r["steps_final"] for r in R if r["succ"])
    print(f"\n eval {ev}: return  min={rets[0]:.3f} p25={st.quantiles(rets,n=4)[0]:.3f} med={st.median(rets):.3f} p75={st.quantiles(rets,n=4)[2]:.3f} max={rets[-1]:.3f} mean={st.mean(rets):.3f} sd={st.pstdev(rets):.3f}")
    print(f"          success-only return: n={len(sr)} min={sr[0]:.3f} med={st.median(sr):.3f} max={sr[-1]:.3f} sd={st.pstdev(sr):.3f}")
    if fr: print(f"          failure-only return: {['%.3f'%x for x in fr]}")
    print(f"          steps   min={stp[0]} p25={st.quantiles(stp,n=4)[0]:.0f} med={st.median(stp):.0f} p75={st.quantiles(stp,n=4)[2]:.0f} max={stp[-1]}  mean={st.mean(stp):.1f}")
    print(f"          success-only steps: min={ss[0]} med={st.median(ss):.0f} max={ss[-1]}")
    # per anatomy median steps
    bym=collections.defaultdict(list)
    for r in R:
        if r["succ"]: bym[r["mesh"]].append(r["steps_final"])
    print("          median success steps by anatomy: "+"  ".join(f"{m}:{st.median(v):.0f}(n={len(v)})" for m,v in sorted(bym.items())))
