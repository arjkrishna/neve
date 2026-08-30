"""CELL 2 extractor: family-A (v1bp) on TOPBRAIN + matched checkpoint0 control.

Reuses monitoring/extract_stuck.py detector logic verbatim (Det state machine,
canonical thresholds) but runs it over EVAL worker logs (logs/<ts>/worker_*.log)
and sweeps stuck_steps in {4,6,8,12}.
"""
import glob, json, os, re, sys, statistics as stx

BS = chr(92)
PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=" + BS + r"[([-0-9.]+),")
INS  = re.compile(r"inserted=" + BS + r"[([-0-9.]+),([-0-9.]+)" + BS + r"]")
DINS = re.compile(r"delta_ins=" + BS + r"[([-0-9.]+),([-0-9.]+)" + BS + r"]")
PL   = re.compile(r"path_len=([0-9.]+)")
SEED = re.compile(r"seed=(" + BS + r"d+)")
ANAT = re.compile(r"anatomy=(" + BS + r"S+)")
MFP  = re.compile(r"mesh_fp=(" + BS + r"S+)")

BASE = dict(stall_eps=0.3, push_min=2.0, retract_min=1.0, soft_max=8.0, pass_eps=1.0)
SWEEP = [4, 6, 8, 12]

class Det:
    __slots__ = ("c","stall","stuck","gw_peak","gw_min","retract","p0","events","exec_mode","onset_step")
    def __init__(self, c, exec_mode=False):
        self.c = c; self.stall = 0; self.stuck = False
        self.gw_peak = self.gw_min = self.retract = 0.0
        self.p0 = 0.0; self.events = []; self.exec_mode = exec_mode; self.onset_step = -1
    def step(self, proj, maxp, push, gw, ep_step):
        c = self.c
        if self.stuck:
            self.gw_min = min(self.gw_min, gw)
            self.retract = max(self.retract, self.gw_peak - self.gw_min)
            if proj > self.p0 + c["pass_eps"]:
                r = self.retract
                kind = ("grind" if r < c["retract_min"] else "soft" if r <= c["soft_max"] else "hard")
                self.events.append({"k": kind, "r": round(r,2), "s": ep_step, "on": round(self.p0,2)})
                self.stuck = False; self.stall = 0
        else:
            stalled = (proj < maxp + c["stall_eps"]) and push
            self.stall = self.stall + 1 if stalled else max(0, self.stall - 2)
            if self.stall >= c["stuck_steps"]:
                self.stuck = True; self.p0 = maxp
                self.gw_peak = self.gw_min = gw; self.retract = 0.0
                self.onset_step = ep_step
    def close(self):
        if self.stuck:
            self.events.append({"k":"unrec","r":round(self.retract,2),"s":-1,"on":round(self.p0,2)})
        return self.events


def new_ep(seed, anat, mfp):
    dets = {}
    for ss in SWEEP:
        c = dict(BASE); c["stuck_steps"] = ss
        dets[("cmd", ss)] = Det(c)
    c = dict(BASE); c["stuck_steps"] = 12
    dets[("exec", 12)] = Det(c, exec_mode=True)
    return {"seed": seed, "anat": anat, "mfp": mfp, "pl": None, "steps": 0,
            "maxp": -1e9, "dets": dets, "outcome": None, "reason": None,
            "run": 0, "runmax": 0, "onsets": [], "final_proj": None,
            "gw_max": 0.0, "maxslack": 0.0}


def parse(run_dir):
    eps = []
    for path in sorted(glob.glob(os.path.join(run_dir, "worker_*.log"))):
        live = {}
        with open(path, errors="replace") as fh:
            for line in fh:
                mp_ = line.find("pid=")
                pid = line[mp_+4:].split(" ")[0].strip() if mp_ >= 0 else "?"
                if "EPISODE_START" in line:
                    prev = live.pop(pid, None)
                    if prev and prev["steps"]: eps.append(fin(prev))
                    ms = SEED.search(line); ma = ANAT.search(line); mf = MFP.search(line)
                    live[pid] = new_ep(int(ms.group(1)) if ms else None,
                                       ma.group(1) if ma else None,
                                       mf.group(1) if mf else None)
                    continue
                st = live.get(pid)
                if st is None: continue
                if "EPISODE_OUTCOME" in line:
                    i = line.find("reason=")
                    st["reason"] = line[i+7:].split(" ")[0].strip() if i >= 0 else None
                    st["outcome"] = True
                    if st["steps"]: eps.append(fin(st))
                    live.pop(pid, None); continue
                if " STEP |" not in line: continue
                mproj = PROJ.search(line); mcmd = CMD.search(line); mins = INS.search(line)
                if not (mproj and mcmd and mins): continue
                mdin = DINS.search(line)
                proj = float(mproj.group(1)); cmd0 = float(mcmd.group(1))
                gw = float(mins.group(1))
                dgw = float(mdin.group(1)) if mdin else 0.0
                st["steps"] += 1
                if st["pl"] is None:
                    m = PL.search(line)
                    if m: st["pl"] = float(m.group(1))
                st["gw_max"] = max(st["gw_max"], gw)
                st["final_proj"] = proj
                st["maxslack"] = max(st["maxslack"], gw - proj)
                push_cmd = abs(cmd0) > BASE["push_min"]
                push_exe = dgw > BASE["push_min"] * 0.132
                # raw longest low-advance-while-pushing run (no decay)
                if (proj < st["maxp"] + BASE["stall_eps"]) and push_cmd:
                    st["run"] += 1; st["runmax"] = max(st["runmax"], st["run"])
                else:
                    st["run"] = 0
                for (mode, ss), d in st["dets"].items():
                    d.step(proj, st["maxp"], push_exe if mode == "exec" else push_cmd,
                           gw, st["steps"])
                st["maxp"] = max(st["maxp"], proj)
        for st in live.values():
            if st["steps"]: eps.append(fin(st))
    return eps


def fin(st):
    out = {k: st[k] for k in ("seed","anat","mfp","pl","steps","runmax","reason",
                              "outcome","final_proj","gw_max","maxslack")}
    out["maxp"] = round(st["maxp"], 2)
    out["ev"] = {"%s%d" % (m, s): d.close() for (m, s), d in st["dets"].items()}
    return out


if __name__ == "__main__":
    eps = parse(sys.argv[1])
    with open(sys.argv[2], "w") as fh:
        for e in eps: fh.write(json.dumps(e) + BS.replace(BS, "") + "\n")
    sys.stderr.write("episodes=%d outcomes=%d\n" % (len(eps), sum(1 for e in eps if e["outcome"])))
