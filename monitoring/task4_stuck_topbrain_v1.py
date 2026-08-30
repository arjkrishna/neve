"""TASK4: stuck/recovery taxonomy for 2026-08-28_075919_rcca_topbrain_v1.
Reuses monitoring/extract_stuck.py detector logic verbatim; adds
 - eval/explore split via " seed=" on EPISODE_START
 - eval-block attribution via main.log evaluation windows
 - stuck_steps sensitivity sweep {4,6,8,12}
 - stall-onset proj_s recorded for spatial distribution
 - EXECUTED-advance (delta_ins) detector variant alongside the proj_s one
Emits one JSON per episode (eval AND explore).
"""
import calendar, glob, json, os, re, sys, time

RUN = sys.argv[1]
OUT = sys.argv[2]

CONFIGS = {
    "canon":  dict(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0, soft_max=8.0, pass_eps=1.0),
    "sens":   dict(stall_eps=0.3, push_min=1.0, stuck_steps=8,  retract_min=0.5, soft_max=8.0, pass_eps=1.0),
    "strict": dict(stall_eps=0.5, push_min=4.0, stuck_steps=16, retract_min=1.0, soft_max=8.0, pass_eps=1.0),
    "ss4":    dict(stall_eps=0.3, push_min=2.0, stuck_steps=4,  retract_min=1.0, soft_max=8.0, pass_eps=1.0),
    "ss6":    dict(stall_eps=0.3, push_min=2.0, stuck_steps=6,  retract_min=1.0, soft_max=8.0, pass_eps=1.0),
    "ss8":    dict(stall_eps=0.3, push_min=2.0, stuck_steps=8,  retract_min=1.0, soft_max=8.0, pass_eps=1.0),
}

WT   = re.compile(r"wall_time=([0-9.]+)")
GS   = re.compile(r"global_steps=([0-9]+)")
PL   = re.compile(r"path_len=([0-9.]+)")
PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=\[([-0-9.]+),")
INS  = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
DINS = re.compile(r"delta_ins=\[([-0-9.]+),([-0-9.]+)\]")
DTG  = re.compile(r"d_tgt=([0-9.]+)")
MESH = re.compile(r"mesh_fp=([A-Za-z0-9_]+)")
LOG  = re.compile(r"step_logger_(train|eval)_(\d+)")
OSTEP= re.compile(r"steps=(\d+)")
TS   = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
EVDUR= re.compile(r"evaluation :\s+([0-9.]+)s")


def eval_windows(run_dir, margin=120.0):
    wins = []
    with open(os.path.join(run_dir, "main.log"), errors="replace") as fh:
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
    __slots__ = ("c","stall","stuck","gw_peak","gw_min","retract","p0","events","onset_s","onset_step")
    def __init__(self, c):
        self.c = c; self.stall = 0; self.stuck = False
        self.gw_peak = self.gw_min = self.retract = 0.0
        self.p0 = 0.0; self.events = []; self.onset_s = 0.0; self.onset_step = 0

    def _close_ev(self, proj, ep_step):
        c = self.c
        if proj > self.p0 + c["pass_eps"]:
            r = self.retract
            kind = ("grind" if r < c["retract_min"] else "soft" if r <= c["soft_max"] else "hard")
            self.events.append({"k":kind,"r":round(r,2),"s":ep_step,
                                "s0":round(self.onset_s,1),"st0":self.onset_step,
                                "dur":ep_step-self.onset_step})
            self.stuck = False; self.stall = 0
            return True
        return False

    def step(self, proj, maxp, cmd0, gw, ep_step):
        c = self.c
        if self.stuck:
            self.gw_min = min(self.gw_min, gw)
            self.retract = max(self.retract, self.gw_peak - self.gw_min)
            self._close_ev(proj, ep_step)
        else:
            stalled = (proj < maxp + c["stall_eps"]) and (cmd0 > c["push_min"])
            self.stall = self.stall + 1 if stalled else max(0, self.stall - 2)
            if self.stall >= c["stuck_steps"]:
                self.stuck = True; self.p0 = maxp
                self.gw_peak = self.gw_min = gw; self.retract = 0.0
                self.onset_s = maxp; self.onset_step = ep_step

    def close(self):
        if self.stuck:
            self.events.append({"k":"unrec","r":round(self.retract,2),"s":-1,
                                "s0":round(self.onset_s,1),"st0":self.onset_step,"dur":-1})
            self.stuck = False
        return self.events


class DetX(Det):
    """Same machine, but the stall test uses EXECUTED gw advance (delta_ins)."""
    __slots__ = ()
    def stepx(self, proj, maxp, cmd0, dgw, gw, ep_step):
        c = self.c
        if self.stuck:
            self.gw_min = min(self.gw_min, gw)
            self.retract = max(self.retract, self.gw_peak - self.gw_min)
            self._close_ev(proj, ep_step)
        else:
            stalled = (dgw < c["stall_eps"]) and (cmd0 > c["push_min"])
            self.stall = self.stall + 1 if stalled else max(0, self.stall - 2)
            if self.stall >= c["stuck_steps"]:
                self.stuck = True; self.p0 = maxp
                self.gw_peak = self.gw_min = gw; self.retract = 0.0
                self.onset_s = maxp; self.onset_step = ep_step


wins = eval_windows(RUN)
sys.stderr.write("eval windows: %d %s\n" % (len(wins), wins))


def evblock(t):
    for i, (a, b) in enumerate(wins):
        if a <= t <= b:
            return i + 1
    return 0


logs = sorted(glob.glob(os.path.join(RUN, "logs_subprocesses/worker_*.log")))
out = open(OUT, "w")
n_start = {"eval":0, "explore":0}
n_out   = {"eval":0, "explore":0}
n_emit  = [0]


def emit(st):
    if st["steps"] == 0:
        return
    rec = {"t":st["t"],"gs":st["gs"],"pid":st["pid"],"w":st["w"],"stream":st["stream"],
           "evblk":st["evblk"],"mesh":st["mesh"],"pl":st["pl"],"steps":st["steps"],
           "succ":bool(st["succ"]),"reason":st["reason"],"maxp":round(st["maxp"],1),
           "gw_max":round(st["gw_max"],1),"d_tgt":st["last_dtgt"],
           "closed":st["closed"],"osteps":st["osteps"],
           "maxrun":st["maxrun"],"maxrunx":st["maxrunx"],
           "n_seam":st["n_seam"],"n_167":st["n_167"],
           "ev":{k:d.close() for k,d in st["dets"].items()},
           "evx":{k:d.close() for k,d in st["detx"].items()}}
    out.write(json.dumps(rec)+"\n")
    n_emit[0] += 1


for path in logs:
    w = int(re.search(r"worker_(\d+)\.log", path).group(1))
    live = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            ml = LOG.search(line)
            if ml is None:
                continue
            lg = ml.group(1) + "_" + ml.group(2)
            if "EPISODE_START" in line:
                i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip()
                key = (lg, pid)
                prev = live.pop(key, None)
                if prev:
                    emit(prev)
                m = WT.search(line); t0 = float(m.group(1)) if m else 0.0
                mg = GS.search(line); gs = int(mg.group(1)) if mg else -1
                mm = MESH.search(line); mesh = mm.group(1) if mm else "?"
                is_eval = (ml.group(1) == "eval")
                assert is_eval == (" seed=" in line), line
                n_start["eval" if is_eval else "explore"] += 1
                live[key] = {"t":t0,"gs":gs,"pid":pid,"w":w,
                             "stream":"eval" if is_eval else "explore",
                             "evblk":evblock(t0) if is_eval else 0,
                             "mesh":mesh,"pl":None,"steps":0,"succ":False,"reason":None,
                             "maxp":-1e9,"gw_max":0.0,"last_dtgt":None,"closed":False,"osteps":-1,
                             "maxrun":0,"maxrunx":0,"n_seam":0,"n_167":0,"dtgt0":None,
                             "dets":{k:Det(v) for k,v in CONFIGS.items()},
                             "detx":{k:DetX(v) for k,v in CONFIGS.items()}}
                continue
            if " STEP |" not in line and "EPISODE_OUTCOME" not in line:
                continue
            i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip()
            key = (lg, pid)
            st = live.get(key)
            if st is None:
                continue
            if "EPISODE_OUTCOME" in line:
                n_out[st["stream"]] += 1
                j = line.find("reason=")
                if j >= 0:
                    st["reason"] = line[j+7:].split(" ")[0].strip()
                    if st["reason"] == "success":
                        st["succ"] = True
                st["closed"] = True
                mo = OSTEP.search(line)
                st["osteps"] = int(mo.group(1)) if mo else -1
                emit(st); live.pop(key, None); continue
            mproj = PROJ.search(line); mcmd = CMD.search(line); mins = INS.search(line)
            md_ = DINS.search(line)
            if not (mproj and mcmd and mins and md_):
                continue
            proj = float(mproj.group(1)); cmd0 = abs(float(mcmd.group(1)))
            gw = float(mins.group(1)); dgw = float(md_.group(1))
            st["steps"] += 1
            if st["pl"] is None:
                m = PL.search(line)
                if m:
                    st["pl"] = float(m.group(1))
            mdt = DTG.search(line)
            if mdt:
                st["last_dtgt"] = float(mdt.group(1))
            st["gw_max"] = max(st["gw_max"], gw)
            if st["dtgt0"] is None and mdt:
                st["dtgt0"] = float(mdt.group(1))
            sr = proj - 33.31
            if sr >= 133.6:
                st["n_seam"] += 1
            if 167.0 <= sr <= 200.0:
                st["n_167"] += 1
            for d in st["dets"].values():
                d.step(proj, st["maxp"], cmd0, gw, st["steps"])
            for d in st["detx"].values():
                d.stepx(proj, st["maxp"], cmd0, dgw, gw, st["steps"])
            st["maxrun"]  = max(st["maxrun"],  st["dets"]["canon"].stall)
            st["maxrunx"] = max(st["maxrunx"], st["detx"]["canon"].stall)
            st["maxp"] = max(st["maxp"], proj)
    for stv in live.values():
        emit(stv)

out.close()
sys.stderr.write("STARTS %s  OUTCOMES %s  emitted %d\n" % (n_start, n_out, n_emit[0]))
