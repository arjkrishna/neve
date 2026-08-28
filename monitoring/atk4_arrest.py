import re, os, sys, json, glob, collections

def parse_kv(line):
    d = {}
    for part in line.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1); d[k.strip()] = v.strip()
    return d

def run(logdir, offset, out):
    eps = {}; order = []; cur = {}
    for f in sorted(glob.glob(os.path.join(logdir, "worker_*.log"))):
        for line in open(f, errors="replace"):
            if "EPISODE_START" in line:
                d = parse_kv(line)
                key = (f, d.get("pid"), d.get("seed"), d.get("ep"))
                cur[(f, d.get("pid"))] = key
                eps[key] = dict(seed=int(d["seed"]), anat=d.get("anatomy"), mesh=d.get("mesh_fp"),
                                target=d.get("target"), maxs=-1e9, path_len=None, n=0,
                                fold_max=0, buck_max=0.0, last=None, hist=[])
                order.append(key)
            elif "STEP |" in line:
                d = parse_kv(line)
                key = cur.get((f, d.get("pid")))
                if key is None: continue
                e = eps[key]; e["n"] += 1
                if e["path_len"] is None and d.get("path_len"):
                    e["path_len"] = float(d["path_len"])
                try: e["maxs"] = max(e["maxs"], float(d["proj_s"]))
                except: pass
                try:
                    fo = d.get("fold", "0/0").split("/")[0]; e["fold_max"] = max(e["fold_max"], int(fo))
                except: pass
                try: e["buck_max"] = max(e["buck_max"], abs(float(d["buckle_phi"])))
                except: pass
                e["last"] = d
                if e["n"] % 1 == 0 and len(e["hist"]) < 100000:
                    pass
    rows = []
    for k in order:
        e = eps[k]; L = e["last"] or {}
        if e["path_len"] is None: continue
        rows.append(dict(seed=e["seed"], anat=e["anat"], target=e["target"],
                         path_len=e["path_len"], s=e["path_len"] - offset,
                         max_proj_s=e["maxs"], max_s_reached=e["maxs"] - offset,
                         steps=int(L.get("ep_step", 0)), success=L.get("term") == "True",
                         d_tgt=float(L.get("d_tgt", "nan")), fold_max=e["fold_max"],
                         buck_max=e["buck_max"], final_proj_s=float(L.get("proj_s", "nan")) - offset,
                         xt=float(L.get("xt_true", "nan")), local_r=float(L.get("local_r", "nan")),
                         tol=float(L.get("tol", "nan")), curbr=L.get("cur_branch"),
                         tip=L.get("tip3d"), nearest=L.get("nearest_named"),
                         inserted=L.get("inserted"), slack=L.get("cath_slack"),
                         arc_past=L.get("arc_past")))
    json.dump(rows, open(out, "w")); print(out, len(rows))

if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]), sys.argv[3])
