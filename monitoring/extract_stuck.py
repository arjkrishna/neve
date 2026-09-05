"""Canonical stuck/recovery extractor — EXPLORE-ONLY, per-episode records.

WHY A DEDICATED EXTRACTOR
- episode_summary.jsonl logs only ~36% of explore episodes => rates OK, counts NOT.
  Worker-log EPISODE_START/STEP/EPISODE_OUTCOME lines are the complete ledger.
- Eval episodes must be excluded. main.log's "evaluation : <dur>s" lines give the EXACT
  window [end-dur-margin, end+margin]; the timestamp prefix of a log line equals its
  wall_time epoch when parsed as UTC (verified per run below), so the windows are exact.

DETECTOR (and why)
A "stuck event" opens when the wire is being PUSHED but path progress stalls:
    proj_s < running_max_proj_s + stall_eps   AND   cmd_action[gw_trans] > push_min
for `stuck_steps` consecutive steps (a decay counter, -2 per non-stalled step, tolerates
brief command sign-flips inside an otherwise-stuck push phase). It CLOSES as a recovery
when progress passes the pre-stuck frontier: proj_s > proj_s_at_onset + pass_eps.
Classification is by how much guidewire was withdrawn before passing:
    grind : retract <  retract_min      brute-forced through, barely pulled back
    soft  : retract_min..soft_max       the wanted move — ease slack, then re-advance
    hard  : retract >  soft_max         escaped only after a large pullback
    unrec : never passed before the episode ended
Retraction is measured as (peak gw insertion at/after onset) - (deepest gw insertion during
the stall), i.e. actual withdrawn length, not commanded.

THRESHOLD SENSITIVITY: every episode is scored under THREE configs simultaneously so the
conclusions can be checked against detector choice (the absolute rates move; the trends
should not).

Emits one JSON object per EXPLORE episode.
"""
import calendar
import glob
import json
import os
import re
import sys
import time

CONFIGS = {
    "canon":  dict(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0, soft_max=8.0, pass_eps=1.0),
    "sens":   dict(stall_eps=0.3, push_min=1.0, stuck_steps=8,  retract_min=0.5, soft_max=8.0, pass_eps=1.0),
    "strict": dict(stall_eps=0.5, push_min=4.0, stuck_steps=16, retract_min=1.0, soft_max=8.0, pass_eps=1.0),
}

WT   = re.compile(r"wall_time=([0-9.]+)")
PL   = re.compile(r"path_len=([0-9.]+)")
PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=\[([-0-9.]+),")
INS  = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
DTG  = re.compile(r"d_tgt=([0-9.]+)")
LOCR = re.compile(r"local_r=([0-9.]+)")
TS   = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
EVDUR= re.compile(r"evaluation :\s*([0-9.]+)s")   # \s*: the trainer pads durations <1000 s with two spaces; the old single-space form matched only the ~1356 s H0 eval


def eval_windows(run_dir, margin=90.0):
    """[(start_epoch, end_epoch)] for every eval block, from main.log."""
    wins = []
    mp = os.path.join(run_dir, "main.log")
    if not os.path.exists(mp):
        return wins
    with open(mp, errors="replace") as fh:
        for line in fh:
            m = EVDUR.search(line)
            if not m:
                continue
            mt = TS.match(line)
            if not mt:
                continue
            end = calendar.timegm(time.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S"))
            wins.append((end - float(m.group(1)) - margin, end + margin))
    return wins


class Det:
    """One threshold config's state machine for a single episode."""
    __slots__ = ("c", "stall", "stuck", "gw_peak", "gw_min", "retract", "p0", "events")

    def __init__(self, c):
        self.c = c
        self.stall = 0
        self.stuck = False
        self.gw_peak = self.gw_min = self.retract = 0.0
        self.p0 = 0.0
        self.events = []

    def step(self, proj, maxp, cmd0, gw, ep_step):
        c = self.c
        if self.stuck:
            self.gw_min = min(self.gw_min, gw)
            self.retract = max(self.retract, self.gw_peak - self.gw_min)
            if proj > self.p0 + c["pass_eps"]:
                r = self.retract
                kind = ("grind" if r < c["retract_min"]
                        else "soft" if r <= c["soft_max"] else "hard")
                self.events.append({"k": kind, "r": round(r, 2), "s": ep_step})
                self.stuck = False
                self.stall = 0
        else:
            stalled = (proj < maxp + c["stall_eps"]) and (cmd0 > c["push_min"])
            self.stall = self.stall + 1 if stalled else max(0, self.stall - 2)
            if self.stall >= c["stuck_steps"]:
                self.stuck = True
                self.p0 = maxp
                self.gw_peak = self.gw_min = gw
                self.retract = 0.0

    def close(self):
        if self.stuck:
            self.events.append({"k": "unrec", "r": round(self.retract, 2), "s": -1})
        return self.events


def run(run_dir, out_path):
    wins = eval_windows(run_dir)
    sys.stderr.write(f"eval windows: {len(wins)}\n")

    def in_eval(t):
        for a, b in wins:
            if a <= t <= b:
                return True
        return False

    logs = sorted(glob.glob(os.path.join(run_dir, "diagnostics/logs_subprocesses/worker_*.log")))
    n_ep = n_eval = 0
    out = open(out_path, "w")
    for path in logs:
        live = {}   # pid -> episode state
        with open(path, errors="replace") as fh:
            for line in fh:
                if "EPISODE_START" in line:
                    mp_ = line.find("pid=")
                    pid = line[mp_ + 4:].split(" ")[0].strip() if mp_ >= 0 else "?"
                    prev = live.pop(pid, None)
                    if prev:
                        n_ep, n_eval = _emit(prev, out, n_ep, n_eval)
                    m = WT.search(line)
                    t0 = float(m.group(1)) if m else 0.0
                    # EXACT eval discriminator: runner.eval() resets with an
                    # explicit seed, so eval EPISODE_START lines carry `seed=`
                    # while explore resets (no seed) do not. Verified on v1bp:
                    # 980 seeded == 10 evals x 98 exactly, 8040 unseeded.
                    # (The main.log time-window method leaks — it caught only
                    # 567 of those 980, contaminating explore with ~5% eval.)
                    is_eval = " seed=" in line
                    live[pid] = {
                        "t": t0, "pid": pid, "ev": is_eval,
                        "pl": None, "steps": 0, "succ": False, "reason": None,
                        "maxp": -1e9, "dets": {k: Det(v) for k, v in CONFIGS.items()},
                        "gw_max": 0.0, "cath_lead": 0, "n": 0, "last_dtgt": None,
                        "minr": None,
                    }
                    continue
                if " STEP |" not in line and "EPISODE_OUTCOME" not in line:
                    continue
                mp_ = line.find("pid=")
                pid = line[mp_ + 4:].split(" ")[0].strip() if mp_ >= 0 else "?"
                st = live.get(pid)
                if st is None:
                    continue
                if "EPISODE_OUTCOME" in line:
                    i = line.find("reason=")
                    if i >= 0:
                        st["reason"] = line[i + 7:].split(" ")[0].strip()
                        if st["reason"] == "success":
                            st["succ"] = True
                    n_ep, n_eval = _emit(st, out, n_ep, n_eval)
                    live.pop(pid, None)
                    continue
                # --- STEP line ---
                mproj = PROJ.search(line); mcmd = CMD.search(line); mins = INS.search(line)
                if not (mproj and mcmd and mins):
                    continue
                proj = float(mproj.group(1)); cmd0 = abs(float(mcmd.group(1)))
                gw = float(mins.group(1)); cath = float(mins.group(2))
                st["steps"] += 1
                if st["pl"] is None:
                    m = PL.search(line)
                    if m:
                        st["pl"] = float(m.group(1))
                md = DTG.search(line)
                if md:
                    st["last_dtgt"] = float(md.group(1))
                mr = LOCR.search(line)
                if mr:
                    v = float(mr.group(1))
                    st["minr"] = v if st["minr"] is None else min(st["minr"], v)
                st["gw_max"] = max(st["gw_max"], gw)
                if cath > gw + 50:
                    st["cath_lead"] += 1
                st["n"] += 1
                for d in st["dets"].values():
                    d.step(proj, st["maxp"], cmd0, gw, st["steps"])
                st["maxp"] = max(st["maxp"], proj)
                if "term=True" in line and "trunc=False" in line:
                    st["succ"] = True
        for st in live.values():
            n_ep, n_eval = _emit(st, out, n_ep, n_eval)
    out.close()
    sys.stderr.write(f"explore episodes emitted: {n_ep}   eval episodes skipped: {n_eval}\n")


def _emit(st, out, n_ep, n_eval):
    if st["pl"] is None or st["steps"] == 0:
        return n_ep, n_eval
    if st["ev"]:
        return n_ep, n_eval + 1
    rec = {
        "t": st["t"], "pid": st["pid"], "pl": st["pl"], "steps": st["steps"],
        "succ": bool(st["succ"]), "reason": st["reason"],
        "gw_max": round(st["gw_max"], 1),
        "cath_lead_frac": round(st["cath_lead"] / max(1, st["n"]), 3),
        "d_tgt": st["last_dtgt"], "min_r": st["minr"],
        "ev": {k: d.close() for k, d in st["dets"].items()},
    }
    out.write(json.dumps(rec) + "\n")
    return n_ep + 1, n_eval


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
