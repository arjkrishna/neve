import os, glob, json
BASE = r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-08-28_034549_rcca_topbrain_smoke"
LOGD = os.path.join(BASE, "diagnostics", "logs_subprocesses")
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
            cur = dict(kind="eval" if "seed" in d else "heatup", ep=int(d.get("ep",-1)),
                       pid=d.get("pid"), seed=d.get("seed"), mesh=d.get("mesh_fp"),
                       tgt=d.get("target"), rows=[], outcome=None)
            eps.append(cur)
        elif "STEP |" in line and cur is not None:
            cur["rows"].append(kv(line))
        elif "EPISODE_OUTCOME |" in line and cur is not None:
            cur["outcome"] = kv(line); cur = None
ev=[e for e in eps if e["kind"]=="eval"]
out=[]
for e in ev:
    R=e["rows"]
    succ=int(e["outcome"]["grader_success"])
    tips=[]
    for r in R:
        t=r["tip3d"].strip("()").split(",")
        tips.append([float(t[0]),float(t[1]),float(t[2])])
    out.append(dict(mesh=e["mesh"], ep=e["ep"], pid=int(e["pid"]), seed=int(e["seed"]),
        succ=succ, reason=e["outcome"]["reason"], final_branch=e["outcome"]["final_branch"],
        path_len=float(R[0]["path_len"]),
        tgt=[float(x) for x in e["tgt"].strip("()").split(",")],
        proj_s=[round(float(r["proj_s"]),2) for r in R],
        phys=[r.get("phys") for r in R],
        cur_branch=[r.get("cur_branch") for r in R],
        on_path=[int(r.get("on_path","0")) for r in R],
        off_br=[int(r.get("off_br","0")) for r in R],
        xt=[round(float(r.get("xt_true","0")),2) for r in R],
        tips=[[round(v,2) for v in t] for t in tips]))
json.dump(out, open(r"D:\Arjun\workspace\neve\monitoring\h0_traj.json","w"))
print("episodes", len(out), "failures", sum(1 for r in out if not r["succ"]))
print("size MB", os.path.getsize(r"D:\Arjun\workspace\neve\monitoring\h0_traj.json")/1e6)
