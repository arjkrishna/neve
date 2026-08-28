import os,re,json,csv,math
from collections import defaultdict
RUN="D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
LOG=os.path.join(RUN,"logs","20260828_053306")
OFF=33.314

# --- episodes.csv ---
rows=list(csv.DictReader(open(os.path.join(RUN,"episodes.csv"))))
by_seed={}
for r in rows:
    by_seed[int(r["seed"])]=dict(path_len=float(r["path_len_mm"]),section=r["section"],
        steps=int(r["steps"]),success=int(r["success"]),anat=r["anatomy"],gh=r["geometry_hash"])
print("csv rows",len(rows),"unique seeds",len(by_seed))

# --- jsonl ---
jl={}
for ln in open(os.path.join(RUN,"episodes_official_20260828_053306.jsonl")):
    d=json.loads(ln); jl[d["seed"]]=d
print("jsonl rows",len(jl))
mis=[s for s in by_seed if s in jl and jl[s]["success"]!=bool(by_seed[s]["success"])]
print("csv/jsonl success mismatches:",len(mis))
print("csv successes",sum(v["success"] for v in by_seed.values()),"jsonl",sum(1 for d in jl.values() if d["success"]))

# --- parse STEP logs ---
def kv(line):
    d={}
    for part in line.split(" | "):
        if "=" in part:
            k,v=part.split("=",1); d[k.strip()]=v.strip()
    return d

eps={}  # (worker,ep) -> dict
for fn in sorted(os.listdir(LOG)):
    if not fn.startswith("worker_"): continue
    path=os.path.join(LOG,fn)
    cur=None
    for line in open(path,errors="replace"):
        if "EPISODE_START" in line:
            d=kv(line); ep=d.get("ep")
            key=(fn,ep)
            cur=dict(worker=fn,ep=ep,seed=int(d["seed"]),mesh=d.get("mesh_fp"),
                     anat=d.get("anatomy"),steps=[],outcome=None)
            eps[key]=cur
        elif "STEP |" in line and cur is not None:
            d=kv(line)
            if d.get("ep")!=cur["ep"]: continue
            try:
                cur["steps"].append(dict(
                    n=int(d["ep_step"]), proj_s=float(d["proj_s"]), path_len=float(d["path_len"]),
                    d_tgt=float(d["d_tgt"]), xt=float(d["xt_true"]),
                    ins=[float(x) for x in d["inserted"].strip("[]").split(",")],
                    dins=[float(x) for x in d["delta_ins"].strip("[]").split(",")],
                    cmd=[float(x) for x in d["cmd_action"].strip("[]").split(",")],
                    on_path=int(d.get("on_path",-1)), on_br=int(d.get("on_br",-1)),
                    off_br=int(d.get("off_br",-1)), br=d.get("cur_branch",""),
                    lr=float(d.get("local_r",0)), tol=float(d.get("tol",0)),
                    term=d.get("term"), trunc=d.get("trunc")))
            except Exception: pass
        elif "EPISODE_OUTCOME" in line and cur is not None:
            d=kv(line)
            if d.get("ep")==cur["ep"]: cur["outcome"]=d

print("parsed episodes",len(eps))
seeds_log={v["seed"] for v in eps.values()}
print("log seeds",len(seeds_log),"missing from csv",len(seeds_log-set(by_seed)),
      "csv seeds missing from log",len(set(by_seed)-seeds_log))
nostep=[ (v["worker"],v["ep"],v["seed"]) for v in eps.values() if not v["steps"]]
print("episodes with 0 STEP lines:",len(nostep),nostep[:5])
import pickle
pickle.dump({"eps":eps,"by_seed":by_seed,"jl":jl},open("D:/Arjun/workspace/neve/monitoring/_t2_parsed.pkl","wb"))
