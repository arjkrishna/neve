import os, re, glob, json, math, statistics as st
from collections import defaultdict, Counter

V1BP = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
H0   = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
HOLD = {"topcowmr004","topcowmr008","topcowmr017","topcowmr023"}

def kv(line):
    d={}
    for part in line.split(" | "):
        if "=" in part:
            k,v=part.split("=",1); d[k.strip()]=v.strip()
    return d

def parse(files):
    """returns dict (pid, envkey, ep) -> episode dict. Uses mesh_fp to split envs."""
    eps={}
    for f in files:
        # state per pid: current open episode key
        cur={}
        for line in open(f, errors="replace"):
            if "EPISODE_START" in line:
                d=kv(line[line.index("EPISODE_START"):])
                pid=d.get("pid"); mesh=d.get("mesh_fp","?")
                envk = "hold" if mesh in HOLD else "train"
                key=(f,pid,envk,d.get("ep"),len(eps))
                # need stable: use running counter per (pid,envk)
                key=(f,pid,envk,d.get("ep"))
                while key in eps: key=key+("d",)
                eps[key]={"file":f,"pid":pid,"env":envk,"mesh":mesh,
                          "anatomy":d.get("anatomy"),"seed":d.get("seed"),
                          "target":d.get("target"),"ep":d.get("ep"),
                          "acts":[], "steps_logged":0, "last":None, "first":None}
                cur[(pid,envk)]=key
            elif "STEP |" in line:
                d=kv(line[line.index("STEP |"):])
                pid=d.get("pid")
                # determine env: STEP has no mesh; attach to most recent open ep for pid.
                # disambiguate by ep number match
                cand=[k for k in (cur.get((pid,"hold")),cur.get((pid,"train"))) if k and k[3]==d.get("ep")]
                if len(cand)!=1:
                    cand=[k for k in (cur.get((pid,"hold")),cur.get((pid,"train"))) if k]
                    if not cand: continue
                    # ambiguous -> skip
                    if len(cand)>1: continue
                k=cand[0]; e=eps[k]
                a=d.get("cmd_action","")
                m=re.match(r"\[([-\d\.,e]+)\]",a)
                if m:
                    try: e["acts"].append([float(x) for x in m.group(1).split(",")])
                    except: pass
                e["steps_logged"]+=1
                if e["first"] is None: e["first"]=d
                e["last"]=d
            elif "EPISODE_END" in line:
                d=kv(line[line.index("EPISODE_END"):])
                pid=d.get("pid")
                cand=[k for k in (cur.get((pid,"hold")),cur.get((pid,"train"))) if k and k[3]==d.get("ep")]
                if len(cand)==1:
                    eps[cand[0]]["end"]=d
                    cur.pop((cand[0][1],cand[0][2]),None)
    return eps

def dist(vals,name):
    if not vals: return f"{name}: n=0"
    vals=sorted(vals)
    n=len(vals)
    q=lambda p: vals[min(n-1,int(p*n))]
    return (f"{name}: n={n} mean={sum(vals)/n:.3f} sd={(st.pstdev(vals) if n>1 else 0):.3f} "
            f"min={vals[0]:.3f} p10={q(.10):.3f} p50={q(.50):.3f} p90={q(.90):.3f} max={vals[-1]:.3f}")

print("#### V1BP EVAL RUN 20260828_045651")
v=parse(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
v_hold={k:e for k,e in v.items() if e["env"]=="hold"}
print("episodes parsed:", len(v), "hold:", len(v_hold), "train-env:", len(v)-len(v_hold))
print("mesh counts:", Counter(e["mesh"] for e in v.values()))
print("mesh->hash:", {e["mesh"]:e["anatomy"] for e in v.values()})
seeds=[e["seed"] for e in v.values()]
print("n seeds:", len(seeds), "unique:", len(set(seeds)))
