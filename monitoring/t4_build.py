import json, pickle, numpy as np, collections, os
OFF = 33.31
GRAFT_PL = 166.91
R = "D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints"

def official(p):
    d = {}
    for line in open(p):
        j = json.loads(line)
        d[j["seed"]] = j
    return d

OT = official(f"{R}/eval_anatomies_checkpoint2002292/episodes_official_20260828_053306.jsonl")
OH = official(f"{R}/eval_anatomies_checkpoint0/episodes_official_20260828_062606.jsonl")

def build(pkl, off_map):
    eps = pickle.load(open(pkl, "rb"))
    rows = []
    for e in eps:
        ps = np.array(e["projs"], float)
        if len(ps) == 0 or e["path_len"] is None: continue
        fo = np.array(e["fold"], int); sl = np.array(e["slack"], float)
        d1 = np.array(e["dins1"], float); d0 = np.array(e["dins0"], float)
        n = len(ps)
        pl = e["path_len"]
        mx = float(np.nanmax(ps))
        imx = int(np.nanargmax(ps))
        o = off_map.get(e["seed"], {})
        # stall window: last 100 steps (or last 25% if short)
        w = min(100, n)
        tail = ps[-w:]
        rows.append(dict(
            seed=e["seed"], mesh=e["mesh"], anat=e["mesh"].replace("topcow","mr_") if e["mesh"] else None,
            path_len=pl, tgt_s=pl-OFF, grafted=pl > GRAFT_PL,
            n=n, cap=(n >= 600),
            succ=bool(o.get("success", False)), grader=bool(o.get("grader_success", False)),
            rew=o.get("reward"), off_steps=o.get("steps"),
            max_projs=mx, max_s=mx-OFF, i_max=imx, frac_at_max=imx/max(n-1,1),
            shortfall=pl-mx,
            final_projs=float(ps[-1]), final_s=float(ps[-1])-OFF,
            tail_med_s=float(np.nanmedian(tail))-OFF, tail_p90_s=float(np.nanpercentile(tail,90))-OFF,
            tail_p10_s=float(np.nanpercentile(tail,10))-OFF,
            fold_max=int(fo.max()), fold_frac_ge20=float((fo>=20).mean()),
            slack_max=float(np.nanmax(sl)), slack_final=float(sl[-1]),
            push_frac=float((d1>1e-6).mean()), pull_frac=float((d1<-1e-6).mean()),
            hold_frac=float((np.abs(d1)<=1e-6).mean()),
            net_ins=float(np.nansum(d1)), abs_ins=float(np.nansum(np.abs(d1))),
            c_push=float((d0>1e-6).mean()), c_pull=float((d0<-1e-6).mean()),
            term=e["last"].get("term"), trunc=e["last"].get("trunc"),
            d_tgt_final=float(e["last"].get("d_tgt","nan")),
            nearest=e["last"].get("nearest_named"), curbr=e["last"].get("cur_branch"),
        ))
    return rows

T = build("_t4_teacher.pkl", OT)
H = build("_t4_heur.pkl", OH)
pickle.dump(dict(T=T,H=H), open("_t4_rows.pkl","wb"))
print(len(T), len(H))
# sanity: match seeds/targets
mt = {r["seed"]:r for r in T}; mh = {r["seed"]:r for r in H}
same = sum(1 for s in mt if s in mh and mt[s]["mesh"]==mh[s]["mesh"] and abs(mt[s]["path_len"]-mh[s]["path_len"])<1e-6)
print("matched seed+mesh+path_len:", same)
gt = [r for r in T if r["grafted"]]
print("grafted teacher:", len(gt), "succ", sum(r["succ"] for r in gt))
gh = [r for r in H if r["grafted"]]
print("grafted heur:", len(gh), "succ", sum(r["succ"] for r in gh))
print("all teacher succ", sum(r["succ"] for r in T), "all heur succ", sum(r["succ"] for r in H))
# check proj_s frame vs path_len on successes
sg=[r for r in gt if r["succ"]]
print("succ shortfall (pl-maxprojs) med", np.median([r["shortfall"] for r in sg]).round(2),
      "p05", np.percentile([r["shortfall"] for r in sg],5).round(2),
      "p95", np.percentile([r["shortfall"] for r in sg],95).round(2))
