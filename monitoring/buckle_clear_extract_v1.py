"""Buckle-clearing window extractor.

Reuses extract_stuck.py's CANON detector VERBATIM (stall_eps=0.3, push_min=2.0,
stuck_steps=12, retract_min=1.0, soft_max=8.0, pass_eps=1.0) so the event set is
byte-identical to the established taxonomy, then measures a BEFORE/DURING/AFTER
window around every SOFT and HARD event.

Reads EVAL log dirs (logs/<stamp>/worker_*.log) where EVERY episode is an eval
episode (EPISODE_START carries seed=).  Streams line by line; buffers only the
per-step scalars of the currently-live episode of each pid.
"""
import glob, json, os, re, sys

C = dict(stall_eps=0.3, push_min=2.0, stuck_steps=12,
         retract_min=1.0, soft_max=8.0, pass_eps=1.0)

PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=\[([-0-9.]+),")
INS  = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
FOLD = re.compile(r"fold=(\d+)/")
PL   = re.compile(r"path_len=([0-9.]+)")
SEED = re.compile(r"seed=(\d+)")


def new_ep(seed):
    return dict(seed=seed, succ=False, reason=None, pl=None,
                proj=[], gw=[], fold=[], cmd=[],
                maxp=-1e9, stall=0, stuck=False, first=0,
                gw_peak=0.0, gw_min=0.0, retract=0.0, p0=0.0, onset=0,
                events=[])


def step(st, proj, cmd0, gw, fd, i):
    """i is 1-based ep_step index into st['proj'] (i-1)."""
    c = C
    if st["stuck"]:
        st["gw_min"] = min(st["gw_min"], gw)
        st["retract"] = max(st["retract"], st["gw_peak"] - st["gw_min"])
        if proj > st["p0"] + c["pass_eps"]:
            r = st["retract"]
            kind = ("grind" if r < c["retract_min"]
                    else "soft" if r <= c["soft_max"] else "hard")
            st["events"].append(dict(k=kind, r=round(r, 3), close=i,
                                     onset=st["onset"], first=st["first"],
                                     p0=st["p0"]))
            st["stuck"] = False
            st["stall"] = 0
    else:
        stalled = (proj < st["maxp"] + c["stall_eps"]) and (cmd0 > c["push_min"])
        if stalled:
            if st["stall"] == 0:
                st["first"] = i          # step where this stall run began
            st["stall"] += 1
        else:
            st["stall"] = max(0, st["stall"] - 2)
        if st["stall"] >= c["stuck_steps"]:
            st["stuck"] = True
            st["onset"] = i
            st["p0"] = st["maxp"]
            st["gw_peak"] = st["gw_min"] = gw
            st["retract"] = 0.0


def close_ep(st, out, counts):
    if st["pl"] is None or not st["proj"]:
        return
    if st["stuck"]:
        st["events"].append(dict(k="unrec", r=round(st["retract"], 3), close=-1,
                                 onset=st["onset"], first=st["first"], p0=st["p0"]))
    proj, gw, fold = st["proj"], st["gw"], st["fold"]
    n = len(proj)
    slack = [gw[j] - proj[j] for j in range(n)]
    counts["ep"] += 1
    counts["succ"] += 1 if st["succ"] else 0
    counts["steps"] += n
    for kind in ("grind", "soft", "hard", "unrec"):
        counts[kind] += sum(1 for e in st["events"] if e["k"] == kind)

    ev_out = []
    for idx, e in enumerate(st["events"]):
        rec = dict(seed=st["seed"], succ=st["succ"], reason=st["reason"],
                   pl=st["pl"], ep_steps=n, k=e["k"], retract=e["r"],
                   first=e["first"], onset=e["onset"], close=e["close"],
                   p0=e["p0"])
        a = e["first"] - 1                       # 0-based start of stall run
        b = (e["close"] - 1) if e["close"] > 0 else n - 1   # 0-based close
        a = max(0, min(a, n - 1)); b = max(a, min(b, n - 1))
        seg = slice(a, b + 1)
        # pre-stall BASELINE slack: median over the <=20 steps ending just
        # before the stall run began.  Absolute slack_gw is policy-specific
        # (chord-cutting drives it negative), so buckle LOAD must be measured
        # as a rise above the episode's own local baseline.
        pre = slack[max(0, a - 20):a] or slack[:1]
        sp = sorted(pre); rec["slack_base"] = round(sp[len(sp) // 2], 3)
        rec["n_pre"] = len(pre)
        rec["slack_rise"] = round(max(slack[seg]) - rec["slack_base"], 3)
        gseg = gw[a:b + 1]
        rec["gw_fed"] = round(max(gseg) - gseg[0], 3)
        if e["close"] > 0:
            post = slack[b:min(b + 11, n)]
            rec["slack_post"] = round(sorted(post)[len(post) // 2], 3)
            rec["slack_release"] = round(max(slack[seg]) - rec["slack_post"], 3)
            rec["slack_resid"] = round(rec["slack_post"] - rec["slack_base"], 3)
        rec["fold_ge5"] = sum(1 for f in fold[a:b + 1] if f >= 5)
        rec["fold_ge10"] = sum(1 for f in fold[a:b + 1] if f >= 10)
        rec["fold_ge20"] = sum(1 for f in fold[a:b + 1] if f >= 20)
        rec["slack_first"] = round(slack[a], 3)
        rec["slack_onset"] = round(slack[min(e["onset"] - 1, n - 1)], 3) if e["onset"] else None
        rec["slack_max"]   = round(max(slack[seg]), 3)
        rec["slack_min"]   = round(min(slack[seg]), 3)
        rec["slack_drop"]  = round(max(slack[seg]) - min(slack[seg]), 3)
        rec["slack_at_close"] = round(slack[b], 3)
        rec["fold_first"]  = fold[a]
        rec["fold_max"]    = max(fold[seg])
        rec["fold_at_close"] = fold[b]
        rec["fold_reset"]  = int(any(f == 0 for f in fold[a:b + 1]))
        rec["stall_len"]   = b - a + 1
        if e["close"] > 0:
            for K in (10, 25, 50):
                j = min(b + K, n - 1)
                rec["adv_%d" % K] = round(max(proj[b:j + 1]) - e["p0"], 3)
            rec["adv_end"] = round(max(proj[b:]) - e["p0"], 3)
            rec["proj_end"] = round(proj[-1], 3)
            rec["steps_left"] = n - 1 - b
            # re-stall at same station: next event whose stall run begins
            # within 20 steps of close and whose arrest proj_s is within 2 mm
            rec["restall_dp"] = None
            rec["restall_gap"] = None
            for f in st["events"][idx + 1:]:
                gap = f["first"] - e["close"]
                if 0 <= gap <= 20:
                    rec["restall_dp"] = round(abs(f["p0"] - e["p0"]), 3)
                    rec["restall_gap"] = gap
                break
        ev_out.append(rec)
    for r in ev_out:
        out.write(json.dumps(r) + "\n")


def run(log_dir, out_path):
    counts = dict(ep=0, succ=0, steps=0, grind=0, soft=0, hard=0, unrec=0)
    out = open(out_path, "w")
    for path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        live = {}
        with open(path, errors="replace") as fh:
            for line in fh:
                if "EPISODE_START" in line:
                    p = line.find("pid="); pid = line[p + 4:].split(" ")[0].strip()
                    if pid in live:
                        close_ep(live.pop(pid), out, counts)
                    m = SEED.search(line)
                    live[pid] = new_ep(int(m.group(1)) if m else None)
                    continue
                if " STEP |" not in line and "EPISODE_OUTCOME" not in line:
                    continue
                p = line.find("pid="); pid = line[p + 4:].split(" ")[0].strip()
                st = live.get(pid)
                if st is None:
                    continue
                if "EPISODE_OUTCOME" in line:
                    i = line.find("reason=")
                    if i >= 0:
                        st["reason"] = line[i + 7:].split(" ")[0].strip()
                        if st["reason"] == "success":
                            st["succ"] = True
                    close_ep(st, out, counts); live.pop(pid, None)
                    continue
                mp, mc, mi = PROJ.search(line), CMD.search(line), INS.search(line)
                if not (mp and mc and mi):
                    continue
                proj = float(mp.group(1)); cmd0 = abs(float(mc.group(1)))
                gw = float(mi.group(1))
                mf = FOLD.search(line); fd = int(mf.group(1)) if mf else 0
                if st["pl"] is None:
                    m = PL.search(line)
                    if m: st["pl"] = float(m.group(1))
                st["proj"].append(proj); st["gw"].append(gw); st["fold"].append(fd)
                st["cmd"].append(cmd0)
                i = len(st["proj"])
                step(st, proj, cmd0, gw, fd, i)
                st["maxp"] = max(st["maxp"], proj)
                if "term=True" in line and "trunc=False" in line:
                    st["succ"] = True
        for st in list(live.values()):
            close_ep(st, out, counts)
        live.clear()
    out.close()
    sys.stderr.write(json.dumps(counts) + "\n")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
