import pickle, collections, statistics, json
P = r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/h0.pkl"
D = pickle.load(open(P,"rb"))
eps, steps, snap = D["eps"], D["steps"], D["snap"]
SEAM = 166.91
rows=[]
for k in sorted(eps):
    st = eps[k]["start"]; ss = steps[k]
    pl = [float(s["path_len"]) for s in ss]
    ps = [float(s["proj_s"]) for s in ss]
    row = dict(pid=k[0], ep=k[1], seed=int(st["seed"]), mesh=st["mesh_fp"],
               anat=st["anatomy"], target=st["target"],
               path_len=statistics.median(pl), path_len_min=min(pl), path_len_max=max(pl),
               nsteps=len(ss), succ = 1 if snap[k]["cls"]=="success" else 0,
               R=snap[k]["R"], snap_steps=snap[k]["steps"],
               proj_s_max=max(ps), proj_s_final=ps[-1],
               d_tgt_final=float(ss[-1]["d_tgt"]), d_tgt_min=min(float(s["d_tgt"]) for s in ss),
               fb = eps[k]["outcome"]["final_branch"] if eps[k]["outcome"] else None,
               reason = eps[k]["outcome"]["reason"] if eps[k]["outcome"] else None)
    rows.append(row)
print("n=",len(rows), "succ=",sum(r["succ"] for r in rows))
# path_len constant per episode?
print("max within-episode path_len spread:", max(r["path_len_max"]-r["path_len_min"] for r in rows))
print()
print("mesh counts:", collections.Counter(r["mesh"] for r in rows))
print()
print("=== ALL 98 H0 EVAL EPISODES ===")
print(f"{'pid':>4} {'ep':>3} {'seed':>4} {'mesh':>12} {'path_len':>8} {'succ':>4} {'steps':>5} {'sMax':>7} {'sFin':>7} {'dtgtF':>6} {'dtgtMin':>7} {'fb':>5} {'reason':>9}")
for r in sorted(rows, key=lambda r:(r["mesh"], r["path_len"])):
    print(f"{r['pid']:>4} {r['ep']:>3} {r['seed']:>4} {r['mesh']:>12} {r['path_len']:>8.1f} {r['succ']:>4} {r['nsteps']:>5} {r['proj_s_max']:>7.1f} {r['proj_s_final']:>7.1f} {r['d_tgt_final']:>6.1f} {r['d_tgt_min']:>7.1f} {str(r['fb']):>5} {str(r['reason']):>9}")
json.dump(rows, open(P.replace("h0.pkl","h0_rows.json"),"w"), indent=0)
