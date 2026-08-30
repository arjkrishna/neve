import os, re, glob, json, pickle, sys
from collections import defaultdict

RUN = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1"
LOGS = os.path.join(RUN, "logs_subprocesses")
OUT  = r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl"

fld = re.compile(r"([a-z_0-9]+)=([^|]*)")

def parse_kv(s):
    d = {}
    for part in s.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

# episodes: key (worker, pid, ep, block) -> dict
eps = {}
order = []

for path in sorted(glob.glob(os.path.join(LOGS, "worker_*.log"))):
    w = os.path.basename(path).replace(".log", "")
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "step_logger_eval" not in line:
                continue
            ts = line[:19]
            hour = int(ts[11:13])
            block = 1 if hour < 10 else (2 if hour < 14 else 3)
            try:
                body = line.split(" - INFO - ", 1)[1]
            except IndexError:
                continue
            if body.startswith("EPISODE_START"):
                d = parse_kv(body)
                key = (w, d.get("pid"), d.get("ep"), block)
                cur = {"w": w, "pid": d.get("pid"), "ep": d.get("ep"), "block": block,
                       "seed": d.get("seed"), "mesh": d.get("mesh_fp"), "t0": ts,
                       "steps": [], "outcome": None}
                eps[key] = cur
                order.append(key)
            elif body.startswith("STEP"):
                d = parse_kv(body)
                key = (w, d.get("pid"), d.get("ep"), block)
                e = eps.get(key)
                if e is None:
                    continue
                def fl(k, dflt=float("nan")):
                    try: return float(d.get(k))
                    except: return dflt
                ca = d.get("cmd_action", "[]").strip("[]").split(",")
                di = d.get("delta_ins", "[]").strip("[]").split(",")
                ins = d.get("inserted", "[]").strip("[]").split(",")
                def g(a, i):
                    try: return float(a[i])
                    except: return float("nan")
                e["steps"].append(dict(
                    n=int(d.get("ep_step", 0)),
                    xt=fl("xt_true"), xtr=fl("cross_tr"), lr=fl("local_r"),
                    ps=fl("proj_s"), pl=fl("path_len"), dtgt=fl("d_tgt"),
                    tol=fl("tol"), buck=fl("buckle_phi"), slack=fl("cath_slack"),
                    onpath=d.get("on_path"), fold=d.get("fold"),
                    over=(d.get("overshoot") == "True"),
                    rew=fl("reward"),
                    dgw=g(di,0), dcath=g(di,1), igw=g(ins,0), icath=g(ins,1),
                    arot=g(ca,0), ains_gw=g(ca,2), ains_cath=g(ca,3),
                    st=fl("step_time".replace("step_time","step_time")) if False else float("nan"),
                    ts=ts,
                ))
            elif body.startswith("EPISODE_OUTCOME"):
                d = parse_kv(body)
                key = (w, d.get("pid"), d.get("ep"), block)
                if key in eps:
                    eps[key]["outcome"] = d
                else:
                    # deferred flush: belongs to an earlier block
                    for b in (1,2,3):
                        k2 = (w, d.get("pid"), d.get("ep"), b)
                        if k2 in eps and eps[k2]["outcome"] is None:
                            eps[k2]["outcome"] = d
                            break

with open(OUT, "wb") as f:
    pickle.dump({"eps": eps, "order": order}, f)

cnt = defaultdict(int); nout = defaultdict(int); nstep = defaultdict(int)
for k, e in eps.items():
    cnt[e["block"]] += 1
    if e["outcome"]: nout[e["block"]] += 1
    nstep[e["block"]] += len(e["steps"])
for b in sorted(cnt):
    print("block", b, "episodes", cnt[b], "with_outcome", nout[b], "steps", nstep[b])
