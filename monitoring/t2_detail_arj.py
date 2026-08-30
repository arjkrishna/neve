import json,collections,statistics as st
rows=json.load(open(r"D:/Arjun/workspace/neve/monitoring/_t2_eval_rows.json"))
for r in rows:
    r["seed"]=int(r["seed"])
    r["succ"]= (1 if r["grader_success"]=="1" else 0) if r["reason"] else (1 if r["last_term"]=="True" else 0)
byev={e:{r["seed"]:r for r in rows if r["eval"]==e} for e in (1,2,3)}
print("--- 6 seeds whose anatomy/target differ across evals ---")
for s in [8,17,68,80,143,159]:
    for e in (1,2,3):
        r=byev[e][s]
        print(f" seed={s:4d} eval{e} {r['worker']:9s} mesh={r['mesh']:12s} anat={r['anatomy']} tgt={r['target']} path_len={r['path_len']:.1f} succ={r['succ']} steps={r['n_steps']}")
    print()
print("--- anatomy count per eval (episode counts) ---")
for e in (1,2,3):
    print(" eval",e,collections.Counter(r["mesh"] for r in byev[e].values()))
print("--- worker->mesh mapping stability for the 6 seeds: seed->worker per eval ---")
for s in [8,17,68,80,143,159]:
    print(" seed",s,[byev[e][s]["worker"] for e in (1,2,3)])
print()
print("--- targets with path_len < 166.91 (seam) ---")
for e in (1,2,3):
    for r in byev[e].values():
        if r["path_len"]<166.91: print("  eval",e,"seed",r["seed"],r["mesh"],"path_len",r["path_len"],"s_RCCA",round(r["path_len"]-33.31,2))
print("--- min s_RCCA per eval and count within [133.0,133.6) ---")
for e in (1,2,3):
    v=[r["path_len"]-33.31 for r in byev[e].values()]
    print("  eval",e,"min s_RCCA=%.2f"%min(v),"n<133.0:",sum(1 for x in v if x<133.0),"n in [133.0,133.6):",sum(1 for x in v if 133.0<=x<133.6))
print()
print("--- eval1 failure taxonomy by max_proj_s ---")
F=[r for r in byev[1].values() if not r["succ"]]
early=[r for r in F if r["max_proj_s"]<100]; deep=[r for r in F if r["max_proj_s"]>=100]
print("  early-stall (max_proj_s<100): n=%d, max_proj_s range %.1f-%.1f, s_RCCA %.1f-%.1f"%(len(early),min(r['max_proj_s'] for r in early),max(r['max_proj_s'] for r in early),min(r['max_proj_s'] for r in early)-33.31,max(r['max_proj_s'] for r in early)-33.31))
print("  deep near-miss (>=100): n=%d seeds %s, last_d_tgt %s"%(len(deep),[r['seed'] for r in deep],[r['last_d_tgt'] for r in deep]))
print("  early-stall meshes:",collections.Counter(r["mesh"] for r in early))
print("  fate of the 6 eval1 deep near-misses in eval2/eval3:")
for r in deep:
    s=r["seed"]; print("   seed",s,r["mesh"],"eval2 succ=",byev[2][s]["succ"],"steps",byev[2][s]["n_steps"],"| eval3 succ=",byev[3][s]["succ"],"steps",byev[3][s]["n_steps"])
print("  fate of the 20 early-stall seeds in eval2:",collections.Counter(byev[2][r["seed"]]["succ"] for r in early))
print()
print("--- success terminal d_tgt distribution (tolerance check) ---")
for e in (1,2,3):
    d=sorted(r["last_d_tgt"] for r in byev[e].values() if r["succ"])
    print("  eval",e,"min=%.1f med=%.1f max=%.1f  n at exactly 5.0: %d"%(d[0],st.median(d),d[-1],sum(1 for x in d if x>4.95)))
print()
print("--- max inserted length (gw/cath) vs path_len, successes ---")
for e in (1,2,3):
    v=[(r["max_ins"],r["path_len"]) for r in byev[e].values() if r["succ"] and r["max_ins"]]
    ratio=sorted(a/b for a,b in v)
    print("  eval",e,"max_inserted/path_len: min=%.3f med=%.3f max=%.3f"%(ratio[0],st.median(ratio),ratio[-1]))
