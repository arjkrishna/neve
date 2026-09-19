#!/usr/bin/env python
"""traj_extract.py -- per-episode trajectory features + compact step series
from env5 worker STEP logs.

usage: python traj_extract.py <run_dir> <out_prefix> --tag <run_tag>

writes  <out_prefix>.features.jsonl   one JSON object per episode
        <out_prefix>.series.npz       ragged int16/int8 step series (offsets)

Every STEP field is parsed by key, so runs that lack a field (older logs)
get NaN / -1 for the features that need it.
"""
import sys, os, glob, json, math, time, argparse
import numpy as np

# Keep the live training run first in line for the CPU.
try:
    import ctypes
    _k = ctypes.windll.kernel32
    _k.SetPriorityClass(_k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
except Exception:
    pass

PHYS = {"bridge": 0, "RCCA": 1, "RVA": 2, "LCCA": 3, "LVA": 4, "other": 5}
CANON = dict(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0,
             soft_max=8.0, pass_eps=1.0)


# ----------------------------------------------------------------- detectors
def detect(P, CM, G, C):
    """Frontier stall detector, byte-for-byte the logic of
    monitoring/buckle_clear_dump_v1.py::step (CM may be abs or signed)."""
    ev = []
    maxp = -1e9; stall = 0; stuck = False; first = 0
    gw_peak = gw_min = retract = p0 = 0.0; onset = 0
    n = len(P)
    for i in range(n):
        proj = P[i]; cmd0 = CM[i]; gw = G[i]; k = i + 1
        if stuck:
            gw_min = min(gw_min, gw); retract = max(retract, gw_peak - gw_min)
            if proj > p0 + C["pass_eps"]:
                r = retract
                kind = "grind" if r < C["retract_min"] else "soft" if r <= C["soft_max"] else "hard"
                ev.append(dict(k=kind, r=round(r, 3), close=k, onset=onset, first=first, p0=p0))
                stuck = False; stall = 0
        else:
            stalled = (proj < maxp + C["stall_eps"]) and (cmd0 > C["push_min"])
            if stalled:
                if stall == 0: first = k
                stall += 1
            else:
                stall = max(0, stall - 2)
            if stall >= C["stuck_steps"]:
                stuck = True; onset = k; p0 = maxp; gw_peak = gw_min = gw; retract = 0.0
        maxp = max(maxp, proj)
    if stuck:
        ev.append(dict(k="unrec", r=round(retract, 3), close=-1, onset=onset, first=first, p0=p0))
    return ev


def runs_of(mask):
    """[(start, length)] of True runs."""
    out = []; n = len(mask); i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def longest_run(mask):
    return max([l for _, l in runs_of(mask)], default=0)


def reversals(d, thr):
    """push<->retract sign flips of an executed delta series, ignoring |d|<=thr."""
    last = 0; n = 0
    for x in d:
        s = 1 if x > thr else -1 if x < -thr else 0
        if s and last and s != last: n += 1
        if s: last = s
    return n


def profile_idx(n, m=20):
    return np.clip(np.round(np.linspace(0, 1, m + 1)[1:] * (n - 1)).astype(int), 0, n - 1)


# ------------------------------------------------------------------ features
def core_features(proj, gw, cath, cmd0, fold, pl):
    """Features computable from proj/gw/cath/|cmd|/fold only -- the subset also
    available in saved/stuck *episode*-format records."""
    P = np.asarray(proj, float); G = np.asarray(gw, float)
    Cc = np.asarray(cath, float) if cath is not None else None
    CM = np.asarray(cmd0, float); F = np.asarray(fold, int)
    n = len(P)
    pl = pl if (pl and pl > 0) else max(float(P.max()), 1.0)
    f = {}
    f["n"] = n
    f["prog_final"] = float(P[-1] / pl); f["prog_max"] = float(P.max() / pl)
    f["t_max_frac"] = float(int(P.argmax()) / max(n - 1, 1))
    idx = profile_idx(n)
    f["prof"] = np.round(P[idx] / pl, 3).tolist()
    f["gw_prof"] = np.round(G[idx] / pl, 3).tolist()
    slack = G - P
    f["slack_prof"] = np.round(slack[idx], 1).tolist()
    f["slack_max"] = float(slack.max()); f["slack_final"] = float(slack[-1])
    f["slack_mean"] = float(slack.mean())
    f["slack_t_max_frac"] = float(int(slack.argmax()) / max(n - 1, 1))
    if Cc is not None:
        lead = Cc - G
        f["cath_lead_max"] = float(lead.max()); f["cath_lead_frac50"] = float((lead > 50).mean())
        f["cath_final"] = float(Cc[-1])
    else:
        f["cath_lead_max"] = f["cath_lead_frac50"] = f["cath_final"] = float("nan")
    f["gw_final"] = float(G[-1])
    dP = np.diff(P, prepend=P[0])
    f["giveback_mm"] = float(np.clip(-dP, 0, None).sum())
    maxp_prev = np.maximum.accumulate(np.concatenate([[-1e9], P[:-1]]))
    flat = P < maxp_prev + 0.3
    f["flat_frac"] = float(flat.mean()); f["flat_run_max"] = int(longest_run(flat))
    # time to progress milestones (fraction of episode), -1 if never reached
    for q in (0.25, 0.5, 0.75, 0.9):
        hit = np.nonzero(P >= q * pl)[0]
        f["t_%d" % int(q * 100)] = float(hit[0] / max(n - 1, 1)) if len(hit) else -1.0
    f["fold_max"] = int(F.max()); f["fold_ge4"] = int((F >= 4).sum())
    f["fold_ge10"] = int((F >= 10).sum())
    f["fold_onsets"] = int(((F[1:] >= 1) & (F[:-1] == 0)).sum())
    f["cmd_mean"] = float(CM.mean()); f["cmd_push_frac"] = float((CM > 10).mean())
    dG = np.diff(G, prepend=G[0])
    ret = dG < -0.1
    f["ret_steps"] = int(ret.sum())
    f["withdrawn_mm"] = float(np.clip(-dG, 0, None).sum())
    f["fed_mm"] = float(np.clip(dG, 0, None).sum())
    bouts = runs_of(ret)
    f["ret_bouts"] = len(bouts)
    f["ret_bout_max_len"] = max([l for _, l in bouts], default=0)
    f["ret_bout_max_mm"] = max([float(-dG[s:s + l].sum()) for s, l in bouts], default=0.0)
    f["reversals"] = reversals(dG, 0.5)
    if Cc is not None:
        dC = np.diff(Cc, prepend=Cc[0])
        f["cath_withdrawn_mm"] = float(np.clip(-dC, 0, None).sum())
        f["cath_ret_bouts"] = len(runs_of(dC < -0.1))
    else:
        f["cath_withdrawn_mm"] = float("nan"); f["cath_ret_bouts"] = -1
    ev = detect(P, np.abs(CM), G, CANON)
    f["ev_n"] = len(ev)
    for kind in ("grind", "soft", "hard", "unrec"):
        f["ev_" + kind] = sum(e["k"] == kind for e in ev)
    f["ev_ret_max"] = max([e["r"] for e in ev], default=0.0)
    f["ev_first_onset_frac"] = float(ev[0]["onset"] / n) if ev else -1.0
    closes = [e["close"] for e in ev if e["close"] > 0]
    f["ev_last_close_frac"] = float(max(closes) / n) if closes else -1.0
    f["ev_stalled_steps"] = int(sum((e["close"] if e["close"] > 0 else n) - e["first"] for e in ev))
    return f, ev


def full_features(E):
    """E: dict of per-step lists collected by the parser."""
    f, ev = core_features(E["proj"], E["gw"], E["cath"], E["cmd0"], E["fold"], E["pl"])
    n = f["n"]
    CMs = np.asarray(E["cmd0"], float)
    f["cmd_retract_frac"] = float((CMs < -1).mean())
    evs = detect(np.asarray(E["proj"], float), CMs, np.asarray(E["gw"], float), CANON)
    f["evS_n"] = len(evs); f["evS_unrec"] = sum(e["k"] == "unrec" for e in evs)
    f["cmd_cath_retract_frac"] = float((np.asarray(E["cmd2"], float) < -1).mean())
    f["gw_rot_abs_mean"] = float(np.abs(np.asarray(E["cmd1"], float)).mean())
    f["cath_rot_abs_mean"] = float(np.abs(np.asarray(E["cmd3"], float)).mean())
    # on-path / branch
    if E["onpath"]:
        OP = np.asarray(E["onpath"], int)
        f["off_steps"] = int((OP == 0).sum())
        f["off_excursions"] = int(((OP[1:] == 0) & (OP[:-1] == 1)).sum())
        f["off_final"] = int(OP[-1] == 0)
        f["off_last100_frac"] = float((OP[-100:] == 0).mean())
        f["off_run_max"] = int(longest_run(OP == 0))
    else:
        for k in ("off_steps", "off_excursions", "off_final", "off_run_max"): f[k] = -1
        f["off_last100_frac"] = float("nan")
    f["off_br_max"] = int(max(E["offbr"])) if E["offbr"] else -1
    if E["phys"]:
        PH = np.asarray(E["phys"], int)
        for name, code in PHYS.items():
            f["phys_" + name] = int((PH == code).sum())
        rva = np.nonzero(PH == PHYS["RVA"])[0]
        f["phys_rva_first_frac"] = float(rva[0] / n) if len(rva) else -1.0
        f["phys_final"] = int(PH[-1])
    else:
        for name in PHYS: f["phys_" + name] = -1
        f["phys_rva_first_frac"] = -1.0; f["phys_final"] = -1
    if E["xt"]:
        XT = np.asarray(E["xt"], float)
        f["xt_max"] = float(XT.max()); f["xt_mean"] = float(XT.mean())
    else:
        f["xt_max"] = f["xt_mean"] = float("nan")
    if E["lr"]:
        f["local_r_min"] = float(min(E["lr"]))
    else:
        f["local_r_min"] = float("nan")
    if E["dtgt"]:
        D = np.asarray(E["dtgt"], float)
        f["dtgt_final"] = float(D[-1]); f["dtgt_min"] = float(D.min())
    else:
        f["dtgt_final"] = f["dtgt_min"] = float("nan")
    if E["tip"]:
        T = np.asarray(E["tip"], float)
        seg = np.linalg.norm(np.diff(T, axis=0), axis=1) if len(T) > 1 else np.zeros(0)
        path = float(seg.sum()); net = float(np.linalg.norm(T[-1] - T[0]))
        f["tip_path_mm"] = path; f["tip_tortuosity"] = path / (net + 1e-6)
        f["tip_backsteps"] = int((seg > 8.0).sum())  # jumps > 8 mm/step = tip whipping
    else:
        f["tip_path_mm"] = f["tip_tortuosity"] = float("nan"); f["tip_backsteps"] = -1
    if E["bphi"]:
        B = np.asarray(E["bphi"], float)
        f["bphi_min"] = float(B.min()); f["bphi_mean"] = float(B.mean())
    else:
        f["bphi_min"] = f["bphi_mean"] = float("nan")
    if E["cs"]:
        CS = np.asarray(E["cs"], float)
        f["cath_slack_max"] = float(CS.max()); f["cath_slack_final"] = float(CS[-1])
        f["cath_slack_ge50"] = int((CS > 50).sum())
    else:
        f["cath_slack_max"] = f["cath_slack_final"] = float("nan"); f["cath_slack_ge50"] = -1
    if E["herr"]:
        f["herr_abs_mean"] = float(np.abs(np.asarray(E["herr"], float)).mean())
    else:
        f["herr_abs_mean"] = float("nan")
    R = np.asarray(E["rew"], float)
    f["rew_sum"] = float(R.sum()); f["rew_neg_steps"] = int((R < -0.005).sum())
    f["rew_min"] = float(R.min())
    f["daughters_max"] = int(max(E["dau"])) if E["dau"] else -1
    f["entries_max"] = int(max(E["ent"])) if E["ent"] else -1
    f["overshoot_any"] = int(any(E["ovs"])) if E["ovs"] else -1
    f["wall_dur_s"] = float(E["wt"][-1] - E["wt"][0]) if len(E["wt"]) > 1 else 0.0
    return f, ev


# -------------------------------------------------------------------- parser
def new_ep(meta):
    E = dict(meta=meta, pl=None, reason=None, final_branch=None, ret=None,
             out_steps=None, complete=False, grader_success=None)
    for k in ("proj", "gw", "cath", "cmd0", "cmd1", "cmd2", "cmd3", "fold", "onpath",
              "offbr", "phys", "xt", "lr", "dtgt", "tip", "bphi", "cs", "herr", "rew",
              "dau", "ent", "ovs", "wt", "dgw", "dcath", "term"):
        E[k] = []
    return E


def close_by_flags(E):
    """No EPISODE_OUTCOME belongs to this episode (validation episodes at the end
    of a block lose theirs: the worker resumes a suspended explore episode and
    writes THAT outcome, or is re-created for explore). The env still logged
    how the episode ended on its last STEP line: term=True & trunc=False is a
    success, trunc=True a truncation; neither means the log was cut mid-episode."""
    if E["complete"]: return
    if E["term"] and E["term"][-1] and not E.get("trunc_last"):
        E["reason"] = "success"; E["complete"] = True; E["source"] = "step_flags"
    elif E.get("trunc_last"):
        E["reason"] = "truncated"; E["complete"] = True; E["source"] = "step_flags"
    else:
        E["source"] = "cut"


def parse_kv(s):
    d = {}
    for t in s.strip().lstrip("| ").split(" | "):
        j = t.find("=")
        if j > 0:
            d[t[:j]] = t[j + 1:]
    return d


def fl(d, k, default=float("nan")):
    v = d.get(k)
    if v is None: return default
    try: return float(v)
    except ValueError: return default


def vec(s):
    return [float(x) for x in s.strip("[]() ").split(",")]


class Writer:
    def __init__(self, prefix, tag):
        self.tag = tag
        self.fo = open(prefix + ".features.jsonl", "w")
        self.series = {k: [] for k in ("proj", "gw", "cath", "fold", "onpath", "phys",
                                       "dgw", "cmd0", "xt", "dtgt", "rew")}
        self.offsets = [0]; self.keys = []; self.n = 0

    def emit(self, E):
        if not E["proj"]: return
        f, ev = full_features(E)
        m = E["meta"]
        row = dict(tag=self.tag, pid=m["pid"], ep=m["ep"], is_eval=m["seed"] is not None,
                   seed=m["seed"], anatomy=m.get("anatomy"), mesh_fp=m.get("mesh_fp"),
                   target=m.get("target"), target_branch=m.get("target_branch"),
                   wt_start=m["wt"], gsteps_start=m.get("gsteps"), worker_file=m["file"],
                   complete=E["complete"], reason=E["reason"], final_branch=E["final_branch"],
                   source=E.get("source", "outcome"),
                   success=(E["reason"] == "success") if E["complete"] else False,
                   grader_success=E["grader_success"], ret=E["ret"], out_steps=E["out_steps"],
                   pl=E["pl"], steps=len(E["proj"]), events=ev)
        row.update(f)
        self.fo.write(json.dumps(row) + "\n")
        n = len(E["proj"]); s = self.series
        s["proj"].append(np.clip(np.round(np.asarray(E["proj"]) * 10), -32000, 32000).astype(np.int16))
        s["gw"].append(np.clip(np.round(np.asarray(E["gw"]) * 10), -32000, 32000).astype(np.int16))
        s["cath"].append(np.clip(np.round(np.asarray(E["cath"]) * 10), -32000, 32000).astype(np.int16))
        s["fold"].append(np.clip(np.asarray(E["fold"]), 0, 32000).astype(np.int16))
        s["onpath"].append(np.asarray(E["onpath"] if E["onpath"] else [-1] * n, np.int8))
        s["phys"].append(np.asarray(E["phys"] if E["phys"] else [-1] * n, np.int8))
        s["dgw"].append(np.clip(np.round(np.asarray(E["dgw"]) * 100), -32000, 32000).astype(np.int16))
        s["cmd0"].append(np.clip(np.round(np.asarray(E["cmd0"]) * 100), -32000, 32000).astype(np.int16))
        s["xt"].append(np.clip(np.round(np.nan_to_num(np.asarray(E["xt"] if E["xt"] else [np.nan] * n, float), nan=-1) * 100), -32000, 32000).astype(np.int16))
        s["dtgt"].append(np.clip(np.round(np.nan_to_num(np.asarray(E["dtgt"] if E["dtgt"] else [np.nan] * n, float), nan=-1) * 10), -32000, 32000).astype(np.int16))
        s["rew"].append(np.clip(np.round(np.asarray(E["rew"]) * 1000), -32000, 32000).astype(np.int16))
        self.offsets.append(self.offsets[-1] + n)
        self.keys.append("%s|%s|%s" % (self.tag, m["pid"], m["ep"]))
        self.n += 1

    def close(self, prefix):
        self.fo.close()
        arrs = {k: (np.concatenate(v) if v else np.zeros(0, np.int16)) for k, v in self.series.items()}
        arrs["offsets"] = np.asarray(self.offsets, np.int64)
        arrs["keys"] = np.asarray(self.keys)
        np.savez_compressed(prefix + ".series.npz", **arrs)


def process_file(path, W):
    live = {}
    fname = os.path.basename(path)
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if "EPISODE_START" in line:
                d = parse_kv(line[line.find("EPISODE_START") + 14:])
                pid = d.get("pid"); ep = int(d.get("ep", -1))
                if pid in live:
                    close_by_flags(live[pid]); W.emit(live.pop(pid))
                seed = d.get("seed"); seed = int(seed) if seed is not None else None
                meta = dict(pid=pid, ep=ep, seed=seed, anatomy=d.get("anatomy"),
                            mesh_fp=d.get("mesh_fp"), target=d.get("target"),
                            target_branch=d.get("target_branch"), wt=fl(d, "wall_time"),
                            gsteps=int(fl(d, "global_steps", -1)), file=fname)
                live[pid] = new_ep(meta); continue
            i = line.find(" STEP | ")
            if i >= 0:
                d = parse_kv(line[i + 8:])
                E = live.get(d.get("pid"))
                if E is None: continue
                try:
                    ca = vec(d["cmd_action"]); ins = vec(d["inserted"]); dins = vec(d["delta_ins"])
                    pr = float(d["proj_s"])
                except (KeyError, ValueError):
                    continue
                if E["pl"] is None and "path_len" in d:
                    try: E["pl"] = float(d["path_len"])
                    except ValueError: pass
                E["proj"].append(pr); E["gw"].append(ins[0]); E["cath"].append(ins[1])
                E["dgw"].append(dins[0]); E["dcath"].append(dins[1])
                E["cmd0"].append(ca[0]); E["cmd1"].append(ca[1]); E["cmd2"].append(ca[2]); E["cmd3"].append(ca[3])
                fv = d.get("fold"); E["fold"].append(int(fv.split("/")[0]) if fv else 0)
                if "on_path" in d: E["onpath"].append(int(d["on_path"]))
                if "off_br" in d: E["offbr"].append(int(d["off_br"]))
                if "phys" in d: E["phys"].append(PHYS.get(d["phys"], 5))
                if "xt_true" in d: E["xt"].append(fl(d, "xt_true"))
                if "local_r" in d: E["lr"].append(fl(d, "local_r"))
                if "d_tgt" in d: E["dtgt"].append(fl(d, "d_tgt"))
                if "tip3d" in d:
                    try: E["tip"].append(vec(d["tip3d"]))
                    except ValueError: pass
                if "buckle_phi" in d: E["bphi"].append(fl(d, "buckle_phi"))
                if "cath_slack" in d: E["cs"].append(fl(d, "cath_slack"))
                if "heading_err" in d: E["herr"].append(fl(d, "heading_err"))
                if "daughters_passed" in d: E["dau"].append(int(fl(d, "daughters_passed", 0)))
                if "entries_passed" in d: E["ent"].append(int(fl(d, "entries_passed", 0)))
                if "overshoot" in d: E["ovs"].append(d["overshoot"] == "True")
                E["rew"].append(fl(d, "reward", 0.0)); E["wt"].append(fl(d, "wall_time"))
                E["term"].append(d.get("term") == "True"); E["trunc_last"] = d.get("trunc") == "True"
                continue
            if "EPISODE_OUTCOME" in line:
                d = parse_kv(line[line.find("EPISODE_OUTCOME") + 16:])
                E = live.get(d.get("pid"))
                if E is None: continue
                if int(fl(d, "ep", -1)) != E["meta"]["ep"]:
                    # outcome of a different episode of this pid (a suspended
                    # explore episode finishing after a validation block): close
                    # the live one from its own step flags and ignore this line
                    close_by_flags(E); W.emit(live.pop(d.get("pid"))); continue
                E["reason"] = d.get("reason"); E["final_branch"] = d.get("final_branch")
                E["ret"] = fl(d, "return"); E["out_steps"] = int(fl(d, "steps", -1))
                gs = d.get("grader_success"); E["grader_success"] = int(gs) if gs is not None else None
                E["complete"] = True
                W.emit(live.pop(d.get("pid")))
    for E in live.values():
        close_by_flags(E); W.emit(E)   # env-ended without outcome line, or log cut (complete=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out_prefix"); ap.add_argument("--tag", required=True)
    ap.add_argument("--logs_glob", default=None,
                    help="glob for worker logs relative to run_dir (default diagnostics/logs_subprocesses/worker_*.log)")
    a = ap.parse_args()
    pat = a.logs_glob or os.path.join("diagnostics", "logs_subprocesses", "worker_*.log")
    files = sorted(glob.glob(os.path.join(a.run_dir, pat), recursive=True))
    W = Writer(a.out_prefix, a.tag)
    t0 = time.time(); total = sum(os.path.getsize(f) for f in files); done = 0
    for i, p in enumerate(files):
        process_file(p, W); done += os.path.getsize(p)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            el = time.time() - t0
            sys.stderr.write("[%s] %d/%d files  %.0f/%.0f MB  eps=%d  %.0fs  (%.1f MB/s)\n" % (
                a.tag, i + 1, len(files), done / 1e6, total / 1e6, W.n, el, done / 1e6 / max(el, 1e-6)))
    W.close(a.out_prefix)
    sys.stderr.write("[%s] DONE episodes=%d in %.0fs\n" % (a.tag, W.n, time.time() - t0))


if __name__ == "__main__":
    main()
