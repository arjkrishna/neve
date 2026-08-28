import os, glob, json, collections
LOGD = r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260828_045651"
def kv(line):
    d = {}
    for part in line.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1); d[k.strip()] = v.strip()
    return d
eps = []
for f in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    cur = None
    for line in open(f, errors="replace"):
        if "EPISODE_START |" in line:
            d = kv(line)
            cur = dict(worker=os.path.basename(f), ep=int(d.get("ep",-1)), pid=d.get("pid"),
                       seed=d.get("seed"), mesh=d.get("mesh_fp"), rows=[], outcome=None)
            eps.append(cur)
        elif "STEP |" in line and cur is not None:
            cur["rows"].append(kv(line))
        elif "EPISODE_OUTCOME |" in line and cur is not None:
            cur["outcome"] = kv(line); cur = None
print("starts", len(eps), "with seed", sum(1 for e in eps if e["seed"]), "with outcome", sum(1 for e in eps if e["outcome"]))
print("meshes", collections.Counter(e["mesh"] for e in eps))
OFF=33.314
out=[]
for e in eps:
    R=e["rows"]
    if not R: continue
    ps=[float(r["proj_s"]) for r in R]
    out.append(dict(mesh=e["mesh"], seed=int(e["seed"]) if e["seed"] else None, ep=e["ep"], pid=e["pid"],
        path_len=float(R[0]["path_len"]), maxps=max(ps), finps=ps[-1], n=len(R),
        reason=e["outcome"]["reason"] if e["outcome"] else None,
        succ=int(e["outcome"]["grader_success"]) if e["outcome"] else None,
        final_branch=e["outcome"]["final_branch"] if e["outcome"] else None))
json.dump(out, open(r"D:\Arjun\workspace\neve\monitoring\h0_teacher98.json","w"), indent=1)
print("rows", len(out), "outcome-missing", sum(1 for r in out if r["succ"] is None))
# join jsonl by seed
JL=r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\episodes_official_20260828_045651.jsonl"
jl={}
for l in open(JL):
    l=l.strip()
    if l:
        d=json.loads(l); jl[d["seed"]]=d
print("jsonl seeds", len(jl), "succ", sum(1 for d in jl.values() if d["success"]))
m=[r for r in out if r["seed"] in jl]
print("matched", len(m))
mismatch=[r for r in m if r["succ"] is not None and int(jl[r["seed"]]["success"])!=r["succ"]]
print("succ mismatch log vs jsonl:", len(mismatch))
