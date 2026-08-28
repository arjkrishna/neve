import re, os, sys, json, glob, collections

def parse_kv(line):
    d = {}
    for part in line.split(" | "):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def parse_dir(logdir):
    eps = {}          # (pid, seed) -> dict
    order = []
    cur = {}          # pid -> key
    for f in sorted(glob.glob(os.path.join(logdir, "worker_*.log"))):
        for line in open(f, errors="replace"):
            if "EPISODE_START" in line:
                d = parse_kv(line)
                pid = d.get("pid"); seed = d.get("seed")
                key = (os.path.basename(f), pid, seed, d.get("ep"))
                cur[(f, pid)] = key
                eps[key] = dict(seed=int(seed), mesh=d.get("mesh_fp"), anat=d.get("anatomy"),
                                target=d.get("target"), branch=d.get("target_branch"),
                                worker=os.path.basename(f), pid=pid, n=0, path_len=None,
                                variant=d.get("phase_c_variant"))
                order.append(key)
            elif "STEP |" in line:
                d = parse_kv(line)
                pid = d.get("pid")
                key = cur.get((f, pid))
                if key is None: continue
                e = eps[key]
                e["n"] += 1
                if e["path_len"] is None and d.get("path_len"):
                    try: e["path_len"] = float(d["path_len"])
                    except: pass
                e["last"] = d
    return [eps[k] for k in order]

def sect(pl):
    return "CCA" if pl < 146 else ("ICA-mid" if pl < 210 else "siphon")

if __name__ == "__main__":
    logdir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    eps = parse_dir(logdir)
    rows = []
    for e in eps:
        L = e.get("last", {})
        if e["path_len"] is None: continue
        succ = L.get("term") == "True"
        rows.append(dict(seed=e["seed"], mesh=e["mesh"], anat=e["anat"], target=e["target"],
                         path_len=e["path_len"], steps=int(L.get("ep_step", 0)),
                         term=L.get("term"), trunc=L.get("trunc"),
                         d_tgt=float(L.get("d_tgt", "nan")),
                         proj_s=float(L.get("proj_s", "nan")),
                         xt=L.get("xt_true"), local_r=L.get("local_r"), tol=L.get("tol"),
                         cum=float(L.get("cum_reward", "nan")),
                         tip=L.get("tip3d"), curbr=L.get("cur_branch"),
                         nearest=L.get("nearest_named"),
                         section=sect(e["path_len"]), success=succ))
    print("n_episodes", len(rows))
    if out:
        json.dump(rows, open(out, "w"))
        print("wrote", out)
