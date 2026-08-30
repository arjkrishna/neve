import os,re,glob,json,math,statistics as st

LOGDIR = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"

# eval round wall-clock windows from main.log (unix ts via wall_time field is safer; use log timestamp string)
# eval1 08:11:52 -> 08:36:09 ; eval2 12:33:09 -> 12:39:42 ; eval3 16:31:26 -> 16:39:00
WIN = [(1,"08:11:00","08:37:00"),(2,"12:32:00","12:41:00"),(3,"16:30:00","16:40:00")]

def which_eval(tstr):
    for i,a,b in WIN:
        if a <= tstr <= b: return i
    return None

kv = re.compile(r"([A-Za-z0-9_]+)=([^|]*)")
def parse_kv(s):
    d={}
    for m in kv.finditer(s):
        d[m.group(1)]=m.group(2).strip()
    return d

eps = {}   # key -> dict
order = []

for fp in sorted(glob.glob(os.path.join(LOGDIR,"worker_*.log"))):
    wname = os.path.basename(fp).replace(".log","")
    with open(fp,"r",errors="replace") as f:
        for line in f:
            if "step_logger_eval" not in line: continue
            if ("EPISODE_START" not in line) and ("STEP |" not in line) and ("EPISODE_OUTCOME" not in line): continue
            tstr = line[11:19]
            ev = which_eval(tstr)
            if ev is None: continue
            body = line.split(" - INFO - ",1)
            if len(body)<2: continue
            body = body[1]
            d = parse_kv(body)
            pid = d.get("pid"); ep = d.get("ep")
            if pid is None or ep is None: continue
            key = (wname, ev, pid, ep)
            if body.startswith("EPISODE_START"):
                e = {"worker":wname,"eval":ev,"pid":pid,"ep":ep,
                     "seed":d.get("seed"),"mesh":d.get("mesh_fp"),"anatomy":d.get("anatomy"),
                     "target":d.get("target"),"path_len":None,"path_lens":set(),
                     "max_proj_s":None,"last_proj_s":None,"last_d_tgt":None,"n_steps":0,
                     "cum_reward":None,"last_term":None,"last_trunc":None,
                     "reason":None,"grader_success":None,"is_clean":None,
                     "out_steps":None,"out_return":None,"t0":tstr,"overshoot":None,
                     "max_ins":None}
                eps[key]=e; order.append(key)
            elif body.startswith("STEP |"):
                e = eps.get(key)
                if e is None: continue
                e["n_steps"] = max(e["n_steps"], int(d.get("ep_step","0") or 0))
                try:
                    pl=float(d["path_len"]); e["path_lens"].add(round(pl,1))
                    if e["path_len"] is None: e["path_len"]=pl
                except Exception: pass
                try:
                    ps=float(d["proj_s"])
                    e["last_proj_s"]=ps
                    e["max_proj_s"]= ps if e["max_proj_s"] is None else max(e["max_proj_s"],ps)
                except Exception: pass
                try: e["last_d_tgt"]=float(d["d_tgt"])
                except Exception: pass
                try: e["cum_reward"]=float(d["cum_reward"])
                except Exception: pass
                e["last_term"]=d.get("term"); e["last_trunc"]=d.get("trunc")
                e["overshoot"]=d.get("overshoot")
                m=re.search(r"inserted=\[([-\d.]+),([-\d.]+)\]",body)
                if m:
                    v=max(float(m.group(1)),float(m.group(2)))
                    e["max_ins"]= v if e["max_ins"] is None else max(e["max_ins"],v)
            elif body.startswith("EPISODE_OUTCOME"):
                e = eps.get(key)
                if e is None: continue
                e["reason"]=d.get("reason"); e["grader_success"]=d.get("grader_success")
                e["is_clean"]=d.get("is_clean"); e["out_steps"]=d.get("steps")
                e["out_return"]=d.get("return")

rows=[eps[k] for k in order]
for r in rows: r["path_lens"]=sorted(r["path_lens"])
out = r"D:/Arjun/workspace/neve/monitoring/_t2_eval_rows.json"
with open(out,"w") as f: json.dump(rows,f)
print("episodes parsed:",len(rows))
from collections import Counter
print("per eval:",Counter(r["eval"] for r in rows))
print("with outcome:",Counter(r["eval"] for r in rows if r["reason"]))
print("seed missing:",sum(1 for r in rows if not r["seed"]))
