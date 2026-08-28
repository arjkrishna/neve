import pickle, collections, statistics, math
P = r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/h0.pkl"
D = pickle.load(open(P,"rb")); eps,steps,snap = D["eps"],D["steps"],D["snap"]
fails = [k for k in sorted(eps) if snap[k]["cls"]=="max_steps"]
def t3(s):
    return tuple(float(x) for x in s.strip("()").split(","))
print(f"{'pid/ep':>8} {'mesh':>11} {'pl':>6} {'sMax':>6} {'sFin':>6} {'xtMax':>6} {'xtFin':>6} {'xt_p95':>6} {'offbr%':>6} {'onpath%':>7} {'inserted_max':>12} {'insFin':>7} {'tipRange_mm':>11} {'lastbranch':>34} {'nearest(last200)':>22}")
detail={}
for k in fails:
    ss = steps[k]; st=eps[k]["start"]
    xt=[float(s["xt_true"]) for s in ss]
    ins=[float(s["inserted"].strip("[]").split(",")[0]) for s in ss]  # device0
    ins1=[float(s["inserted"].strip("[]").split(",")[1]) for s in ss]
    off=[int(s["off_br"]) for s in ss]; onp=[int(s["on_path"]) for s in ss]
    tips=[t3(s["tip3d"]) for s in ss]
    last=tips[-200:]
    cx=[sum(p[i] for p in last)/len(last) for i in range(3)]
    rng=max(math.dist(p,cx) for p in last)
    br=collections.Counter(s["cur_branch"] for s in ss[-200:]).most_common(2)
    nn=collections.Counter(s["nearest_named"] for s in ss[-200:]).most_common(2)
    print(f"{k[0]}/{k[1]:<4} {st['mesh_fp'][-5:]:>11} {float(ss[0]['path_len']):>6.1f} {max(float(s['proj_s']) for s in ss):>6.1f} {float(ss[-1]['proj_s']):>6.1f} {max(xt):>6.1f} {xt[-1]:>6.1f} {sorted(xt)[int(.95*len(xt))]:>6.1f} {100*sum(off)/len(off):>6.1f} {100*sum(onp)/len(onp):>7.1f} {max(ins1):>12.1f} {ins1[-1]:>7.1f} {rng:>11.2f}  {str(br):>32}  {str(nn)}")
    detail[k]=dict(xt=xt,tips=tips,ins1=ins1,ss=ss)
pickle.dump(detail, open(P.replace("h0.pkl","h0_faildetail.pkl"),"wb"))
