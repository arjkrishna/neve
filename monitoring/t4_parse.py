import os, sys, json, glob, pickle
import numpy as np

def parse_kv(line):
    d = {}
    for part in line.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1); d[k.strip()] = v.strip()
    return d

def f2(s):
    # "[0.00,3.96]" -> (0.0, 3.96)
    s = s.strip().strip("[]")
    p = s.split(",")
    return float(p[0]), float(p[1])

def parse_dir(logdir):
    eps = {}; order = []; cur = {}
    for f in sorted(glob.glob(os.path.join(logdir, "worker_*.log"))):
        fb = os.path.basename(f)
        for line in open(f, errors="replace"):
            if "EPISODE_START" in line:
                d = parse_kv(line)
                key = (fb, d.get("pid"), d.get("seed"), d.get("ep"))
                cur[(fb, d.get("pid"))] = key
                eps[key] = dict(seed=int(d["seed"]), mesh=d.get("mesh_fp"), anat=d.get("anatomy"),
                                target=d.get("target"), worker=fb, pid=d.get("pid"),
                                path_len=None,
                                projs=[], fold=[], slack=[], dins0=[], dins1=[],
                                ins0=[], ins1=[], buck=[], onpath=[], dtgt=[], cmd=[], last=None)
                order.append(key)
            elif "STEP |" in line:
                d = parse_kv(line)
                key = cur.get((fb, d.get("pid")))
                if key is None: continue
                e = eps[key]
                if e["path_len"] is None and d.get("path_len"):
                    try: e["path_len"] = float(d["path_len"])
                    except: pass
                try: e["projs"].append(float(d["proj_s"]))
                except: e["projs"].append(np.nan)
                try: e["fold"].append(int(d.get("fold","0/0").split("/")[0]))
                except: e["fold"].append(0)
                try: e["slack"].append(float(d.get("cath_slack","nan")))
                except: e["slack"].append(np.nan)
                try:
                    a,b = f2(d.get("delta_ins","[nan,nan]")); e["dins0"].append(a); e["dins1"].append(b)
                except: e["dins0"].append(np.nan); e["dins1"].append(np.nan)
                try:
                    a,b = f2(d.get("inserted","[nan,nan]")); e["ins0"].append(a); e["ins1"].append(b)
                except: e["ins0"].append(np.nan); e["ins1"].append(np.nan)
                try:
                    ca=[float(x) for x in d.get("cmd_action","[nan,nan,nan,nan]").strip().strip("[]").split(",")]
                    e["cmd"].append(ca)
                except: e["cmd"].append([float("nan")]*4)
                try: e["buck"].append(abs(float(d.get("buckle_phi","nan"))))
                except: e["buck"].append(np.nan)
                try: e["onpath"].append(int(d.get("on_path","0")))
                except: e["onpath"].append(0)
                try: e["dtgt"].append(float(d.get("d_tgt","nan")))
                except: e["dtgt"].append(np.nan)
                e["last"] = d
    return [eps[k] for k in order]

if __name__ == "__main__":
    logdir, out = sys.argv[1], sys.argv[2]
    eps = parse_dir(logdir)
    print("episodes parsed:", len(eps))
    pickle.dump(eps, open(out, "wb"), protocol=4)
