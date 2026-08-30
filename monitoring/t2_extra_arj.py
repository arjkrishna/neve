import json,collections,statistics as st
rows=json.load(open(r"D:/Arjun/workspace/neve/monitoring/_t2_eval_rows.json"))
for r in rows:
    r["seed"]=int(r["seed"])
    r["succ"]= (1 if r["grader_success"]=="1" else 0) if r["reason"] else (1 if r["last_term"]=="True" else 0)
byev={e:{r["seed"]:r for r in rows if r["eval"]==e} for e in (1,2,3)}
MIS={8,17,68,80,143,159}
print("--- paired transitions restricted to the 92 seeds with IDENTICAL anatomy+target across evals ---")
m=[s for s in byev[1] if s not in MIS]
t=collections.Counter((byev[1][s]["succ"],byev[2][s]["succ"]) for s in m)
print(" 1->2 n=%d  fail->succ=%d succ->fail=%d succ->succ=%d fail->fail=%d"%(len(m),t[(0,1)],t[(1,0)],t[(1,1)],t[(0,0)]))
t=collections.Counter((byev[2][s]["succ"],byev[3][s]["succ"]) for s in m)
print(" 2->3 n=%d  fail->succ=%d succ->fail=%d succ->succ=%d fail->fail=%d"%(len(m),t[(0,1)],t[(1,0)],t[(1,1)],t[(0,0)]))
print("\n--- insertion-excess outliers (max_inserted/path_len) ---")
for e in (1,2,3):
    v=sorted(((r["max_ins"]/r["path_len"]),r) for r in byev[e].values() if r["max_ins"])
    print(" eval",e,"top5:",[("seed%d"%r["seed"],r["mesh"],"%.3f"%x,"succ%d"%r["succ"],"steps%d"%r["n_steps"]) for x,r in v[-5:]])
    print("        n>1.05:",sum(1 for x,_ in v if x>1.05),"n>1.20:",sum(1 for x,_ in v if x>1.20))
print("\n--- overshoot flag at last step ---")
for e in (1,2,3):
    print(" eval",e,collections.Counter(r["overshoot"] for r in byev[e].values()))
print("\n--- eval3 anomalous long success (steps>200) ---")
for e in (2,3):
    for r in byev[e].values():
        if r["succ"] and r["n_steps"]>150:
            print("  eval",e,"seed",r["seed"],r["mesh"],"steps",r["n_steps"],"path_len",r["path_len"],"ret",r["cum_reward"],"max_ins",r["max_ins"])
print("\n--- per-anatomy median steps & speed proxy (path_len/steps) successes ---")
for e in (1,2,3):
    bym=collections.defaultdict(list)
    for r in byev[e].values():
        if r["succ"]: bym[r["mesh"]].append(r["path_len"]/r["n_steps"])
    print(" eval",e," ".join("%s:%.2f mm/step"%(k,st.median(v)) for k,v in sorted(bym.items())))
