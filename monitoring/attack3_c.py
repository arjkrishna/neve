import os,csv,json,pickle,statistics as st
from collections import Counter,defaultdict
V1BP = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
SP=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad\a3.pkl"
D=pickle.load(open(SP,"rb")); vb=D["vb"]; h0=D["h0"]
rows=list(csv.DictReader(open(os.path.join(V1BP,"episodes.csv"))))
jl=[json.loads(l) for l in open(os.path.join(V1BP,"episodes_official_20260828_045651.jsonl"))]
print("csv rows",len(rows),"jsonl",len(jl))
jm={j["seed"]:j for j in jl}
print("jsonl unique seeds",len(jm))
# agreement
dis=0; miss=0
for r in rows:
    s=int(r["seed"]); j=jm.get(s)
    if j is None: miss+=1; continue
    if int(r["success"])!=int(bool(j["success"])) or bool(j["success"])!=bool(j["grader_success"]) or int(r["steps"])!=int(j["steps"]):
        dis+=1; print("DISAGREE",r["seed"],r["success"],j["success"],j["grader_success"],r["steps"],j["steps"])
print("missing",miss,"disagreements",dis)
print("jsonl success",sum(1 for j in jl if j["success"]),"grader_success",sum(1 for j in jl if j["grader_success"]))
print("final_branch_short:",Counter(j["final_branch_short"] for j in jl))
# log-derived per-episode steps vs csv
lg={int(e["seed"]):e for e in vb}
print("log seeds == csv seeds:", set(lg)==set(int(r["seed"]) for r in rows))
bad=0
for r in rows:
    e=lg[int(r["seed"])]
    if e["maxstep"]!=int(r["steps"]): bad+=1; print("STEPMISMATCH",r["seed"],e["maxstep"],r["steps"])
print("step mismatches log-vs-csv:",bad)
print("sum steps csv:",sum(int(r["steps"]) for r in rows),"sum logged steps:",sum(e["nsteps"] for e in vb))

succ=[int(r["steps"]) for r in rows if r["success"]=="1"]
fail=[int(r["steps"]) for r in rows if r["success"]!="1"]
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
print("\nSUCCESS steps: n=%d min=%d p10=%d p25=%d p50=%d p75=%d p90=%d max=%d"%(
    len(succ),min(succ),q(succ,.1),q(succ,.25),q(succ,.5),q(succ,.75),q(succ,.9),max(succ)))
print("FAIL steps:",sorted(fail))
for cap in (200,250,300,350,400,450,500,550,600):
    print("  successes with steps<=%d: %d  -> rate %.4f"%(cap,sum(1 for s in succ if s<=cap),sum(1 for s in succ if s<=cap)/98))
# per anatomy
pa=defaultdict(lambda:[0,0])
for r in rows:
    pa[r["anatomy"]][0]+=1; pa[r["anatomy"]][1]+=int(r["success"])
print("\nper-anatomy:", {k:(v[1],v[0],round(v[1]/v[0],4)) for k,v in pa.items()})
rates=[v[1]/v[0] for v in pa.values()]
print("pooled=%.4f (%d/98)  unweighted mean=%.4f  sd=%.4f"%(sum(v[1] for v in pa.values())/98,sum(v[1] for v in pa.values()),sum(rates)/4,st.pstdev(rates)))
# section split
sec=defaultdict(lambda:[0,0])
for r in rows:
    sec[r["section"]][0]+=1; sec[r["section"]][1]+=int(r["success"])
print("sections:",{k:(v[1],v[0],round(v[1]/v[0],4)) for k,v in sec.items()})
# path len
pl=[float(r["path_len_mm"]) for r in rows]
print("path_len_mm: min=%.1f p50=%.1f max=%.1f"%(min(pl),q(pl,.5),max(pl)))
# advance per step for successes
adv=[float(r["path_len_mm"])/int(r["steps"]) for r in rows if r["success"]=="1"]
print("path_len/steps (successes) mm/step: min=%.2f p10=%.2f p50=%.2f p90=%.2f max=%.2f"%(min(adv),q(adv,.1),q(adv,.5),q(adv,.9),max(adv)))
