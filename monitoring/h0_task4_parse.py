import os, re, glob, json, collections

BASE = r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-08-28_034549_rcca_topbrain_smoke"
LOGD = os.path.join(BASE, "diagnostics", "logs_subprocesses")

def kv(line):
    d = {}
    for part in line.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

episodes = []
for f in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    cur = None
    with open(f, "r", errors="replace") as fh:
        for line in fh:
            if "EPISODE_START |" in line:
                d = kv(line)
                cur = dict(kind="eval" if "seed" in d else "heatup",
                           worker=os.path.basename(f), ep=int(d.get("ep", -1)),
                           pid=d.get("pid"), seed=d.get("seed"),
                           target=d.get("target"), mesh=d.get("mesh_fp"),
                           anat=d.get("anatomy"), steps_rows=[], outcome=None)
                episodes.append(cur)
            elif "STEP |" in line and cur is not None:
                d = kv(line)
                cur["steps_rows"].append(d)
            elif "EPISODE_OUTCOME |" in line and cur is not None:
                cur["outcome"] = kv(line)
                cur = None

ev = [e for e in episodes if e["kind"] == "eval"]
print("total episodes parsed:", len(episodes), " eval:", len(ev), " heatup:", len(episodes)-len(ev))
print("eval with outcome:", sum(1 for e in ev if e["outcome"]))
print("mesh counts:", collections.Counter(e["mesh"] for e in ev))
oc = collections.Counter((e["outcome"]["reason"] if e["outcome"] else "MISSING") for e in ev)
print("reasons:", oc)
print("final_branch:", collections.Counter((e["outcome"]["final_branch"] if e["outcome"] else "MISSING") for e in ev))
# path_len from first STEP
import statistics
pl = [float(e["steps_rows"][0]["path_len"]) for e in ev if e["steps_rows"]]
print("n path_len:", len(pl), "min", min(pl), "max", max(pl))
json.dump([{k: v for k, v in e.items() if k != "steps_rows"} | {
    "n_steps": len(e["steps_rows"]),
    "path_len": float(e["steps_rows"][0]["path_len"]) if e["steps_rows"] else None,
    "max_proj_s": max((float(r["proj_s"]) for r in e["steps_rows"]), default=None),
    "final_proj_s": float(e["steps_rows"][-1]["proj_s"]) if e["steps_rows"] else None,
} for e in ev], open(r"D:\Arjun\workspace\neve\monitoring\h0_eval_index.json", "w"), indent=1)
