import os, re, glob, pickle
from collections import defaultdict
LOGS=r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"
def parse_kv(s):
    d={}
    for p in s.split(" | "):
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
out={}
for path in sorted(glob.glob(os.path.join(LOGS,"worker_*.log"))):
    w=os.path.basename(path)[:-4]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            if "step_logger_eval" not in line: continue
            hour=int(line[11:13]); block=1 if hour<10 else (2 if hour<14 else 3)
            try: body=line.split(" - INFO - ",1)[1]
            except IndexError: continue
            if not body.startswith("STEP"): continue
            d=parse_kv(body)
            key=(w,d.get("pid"),int(d.get("ep")),block)
            t=d.get("tip3d","").strip("()").split(",")
            try: t=(float(t[0]),float(t[1]),float(t[2]))
            except: t=None
            try: stt=float(d.get("step_time","0s").rstrip("s"))
            except: stt=float("nan")
            out.setdefault(key,[]).append((int(d.get("ep_step",0)),t,stt))
pickle.dump(out,open(r"D:/Arjun/workspace/neve/monitoring/_t3_tip.pkl","wb"))
print("episodes:",len(out),"steps:",sum(len(v) for v in out.values()))
