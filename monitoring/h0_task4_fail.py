import os, re, glob, json, collections, statistics

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
                cur = dict(kind="eval" if "seed" in d else "heatup", worker=os.path.basename(f),
                           ep=int(d.get("ep", -1)), pid=d.get("pid"), seed=d.get("seed"),
                           target=d.get("target"), mesh=d.get("mesh_fp"), rows=[], outcome=None)
                episodes.append(cur)
            elif "STEP |" in line and cur is not None:
                cur["rows"].append(kv(line))
            elif "EPISODE_OUTCOME |" in line and cur is not None:
                cur["outcome"] = kv(line); cur = None

ev = [e for e in episodes if e["kind"] == "eval"]
OFF = 33.314
recs = []
for e in ev:
    R = e["rows"]
    pl = float(R[0]["path_len"])
    ps = [float(r["proj_s"]) for r in R]
    n = len(ps)
    maxps = max(ps)
    argmax = ps.index(maxps)
    fin = ps[-1]
    # last-100-step behaviour
    tail = ps[-100:] if n >= 100 else ps
    tail_gain = tail[-1] - tail[0]
    tail_rng = max(tail) - min(tail)
    # steps since max
    since_max = n - 1 - argmax
    # off-path evidence
    offbr = sum(1 for r in R if r.get("off_br") == "1")
    onpath0 = sum(1 for r in R if r.get("on_path") == "0")
    branches = collections.Counter(r.get("cur_branch") for r in R)
    nearest = collections.Counter(r.get("nearest_named") for r in R)
    # last 50 steps branch/nearest
    branches_tail = collections.Counter(r.get("cur_branch") for r in R[-50:])
    nearest_tail = collections.Counter(r.get("nearest_named") for r in R[-50:])
    dcorr = [float(r["d_corr_3d"]) for r in R]
    recs.append(dict(mesh=e["mesh"], ep=e["ep"], pid=e["pid"], seed=e["seed"],
        reason=e["outcome"]["reason"], final_branch=e["outcome"]["final_branch"],
        succ=int(e["outcome"]["grader_success"]), n=n, path_len=pl,
        s_rcca_tgt=round(pl-OFF,1), maxps=round(maxps,1), s_max=round(maxps-OFF,1),
        finps=round(fin,1), since_max=since_max, tail_gain=round(tail_gain,2),
        tail_rng=round(tail_rng,2), offbr=offbr, onpath0=onpath0,
        max_dcorr=round(max(dcorr),1), tail_dcorr=round(max(dcorr[-50:]),1),
        br_tail=branches_tail.most_common(2), near_tail=nearest_tail.most_common(2),
        frac_remaining=round((pl-maxps)/pl,3)))

json.dump(recs, open(r"D:\Arjun\workspace\neve\monitoring\h0_eval_recs.json","w"), indent=1)
F = [r for r in recs if r["succ"]==0]
print("failures:", len(F))
print(f"{'mesh':13s}{'ep':>4}{'pid':>5}{'pl':>7}{'sTgt':>7}{'maxps':>7}{'sMax':>7}{'finps':>7}{'sinceMx':>8}{'tailGain':>9}{'tailRng':>8}{'offbr':>6}{'onp0':>5}{'maxdc':>7}{'tdc':>6}  fin_br  br_tail")
for r in sorted(F, key=lambda x:(x["mesh"], x["path_len"])):
    print(f"{r['mesh']:13s}{r['ep']:>4}{r['pid']:>5}{r['path_len']:>7.1f}{r['s_rcca_tgt']:>7.1f}{r['maxps']:>7.1f}{r['s_max']:>7.1f}{r['finps']:>7.1f}{r['since_max']:>8}{r['tail_gain']:>9.2f}{r['tail_rng']:>8.2f}{r['offbr']:>6}{r['onpath0']:>5}{r['max_dcorr']:>7.1f}{r['tail_dcorr']:>6.1f}  {r['final_branch']:5s} {r['br_tail']} near={r['near_tail']}")
