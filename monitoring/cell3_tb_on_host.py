"""CELL 3: TopBrain-trained family on the HOST anatomy. Canonical extract_stuck logic,
applied to EVAL logs (logs/<ts>/worker_*.log), keyed by pid."""
import glob, json, os, re, sys, statistics as S

CANON = dict(stall_eps=0.3, push_min=2.0, retract_min=1.0, soft_max=8.0, pass_eps=1.0)
SWEEP = [4, 6, 8, 12]

PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=\[([-0-9.]+),")
INS  = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
DIN  = re.compile(r"delta_ins=\[([-0-9.]+),([-0-9.]+)\]")
PL   = re.compile(r"path_len=([0-9.]+)")
SEED = re.compile(r"seed=([0-9]+)")
EP   = re.compile(r"\| ep=([0-9]+) ")
DTG  = re.compile(r"d_tgt=([0-9.]+)")
CB   = re.compile(r"cur_branch=([^|]*)\|")
REA  = re.compile(r"reason=(\S+)")
GS   = re.compile(r"grader_success=([01])")
STP  = re.compile(r"steps=([0-9]+)")


class Det:
    __slots__ = ("ss","stall","stuck","gw_peak","gw_min","retract","p0","s0","events")
    def __init__(self, ss):
        self.ss = ss; self.stall = 0; self.stuck = False
        self.gw_peak = self.gw_min = self.retract = 0.0
        self.p0 = 0.0; self.s0 = 0; self.events = []
    def step(self, proj, maxp, cmd0, gw, ep_step):
        c = CANON
        if self.stuck:
            self.gw_min = min(self.gw_min, gw)
            self.retract = max(self.retract, self.gw_peak - self.gw_min)
            if proj > self.p0 + c["pass_eps"]:
                r = self.retract
                k = "grind" if r < c["retract_min"] else "soft" if r <= c["soft_max"] else "hard"
                self.events.append({"k":k,"r":round(r,2),"onset_s":self.s0,"onset_proj":round(self.p0,1),"end":ep_step})
                self.stuck = False; self.stall = 0
        else:
            stalled = (proj < maxp + c["stall_eps"]) and (cmd0 > c["push_min"])
            self.stall = self.stall + 1 if stalled else max(0, self.stall - 2)
            if self.stall >= self.ss:
                self.stuck = True; self.p0 = maxp; self.s0 = ep_step
                self.gw_peak = self.gw_min = gw; self.retract = 0.0
    def close(self):
        if self.stuck:
            self.events.append({"k":"unrec","r":round(self.retract,2),"onset_s":self.s0,
                                "onset_proj":round(self.p0,1),"end":-1})
        return self.events


def new_ep(seed, ep, pid):
    return {"seed":seed,"ep":ep,"pid":pid,"pl":None,"steps":0,"maxp":-1e9,
            "dets":{ss:Det(ss) for ss in SWEEP},"run":0,"maxrun":0,
            "reason":None,"gs":None,"osteo":None,"lastproj":0.0,"maxproj":0.0,
            "gw_max":0.0,"ratio_n":0,"ratio_sum":0.0,"last_dtgt":None,"nstart":1}


def parse(logdir):
    eps = []
    n_start = n_out = 0
    for path in sorted(glob.glob(os.path.join(logdir, "worker_*.log"))):
        live = {}
        with open(path, errors="replace") as fh:
            for line in fh:
                if "EPISODE_START" in line:
                    n_start += 1
                    i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip()
                    prev = live.pop(pid, None)
                    if prev is not None and prev["steps"]:
                        prev["ev"] = {k:d.close() for k,d in prev["dets"].items()}; eps.append(prev)
                    ms = SEED.search(line); me = EP.search(line)
                    live[pid] = new_ep(int(ms.group(1)) if ms else -1,
                                       int(me.group(1)) if me else -1, pid)
                    continue
                if " STEP |" not in line and "EPISODE_OUTCOME" not in line:
                    continue
                i = line.find("pid="); pid = line[i+4:].split(" ")[0].strip()
                st = live.get(pid)
                if st is None: continue
                if "EPISODE_OUTCOME" in line:
                    n_out += 1
                    m = REA.search(line); st["reason"] = m.group(1) if m else None
                    m = GS.search(line); st["gs"] = int(m.group(1)) if m else None
                    m = STP.search(line)
                    st["out_steps"] = int(m.group(1)) if m else None
                    if st["steps"]:
                        st["ev"] = {k:d.close() for k,d in st["dets"].items()}; eps.append(st)
                    live.pop(pid, None)
                    continue
                mp = PROJ.search(line); mc = CMD.search(line); mi = INS.search(line)
                if not (mp and mc and mi): continue
                proj = float(mp.group(1)); cmd0 = abs(float(mc.group(1))); gw = float(mi.group(1))
                st["steps"] += 1
                if st["pl"] is None:
                    m = PL.search(line)
                    if m: st["pl"] = float(m.group(1))
                md = DIN.search(line)
                if md:
                    dg = abs(float(md.group(1)))
                    if cmd0 > 1e-6:
                        st["ratio_n"] += 1; st["ratio_sum"] += dg / (cmd0*0.132)
                mb = CB.search(line)
                if mb and st["osteo"] is None and "RCCA" in mb.group(1):
                    st["osteo"] = proj
                mt = DTG.search(line)
                if mt: st["last_dtgt"] = float(mt.group(1))
                st["gw_max"] = max(st["gw_max"], gw)
                stalled = (proj < st["maxp"] + CANON["stall_eps"]) and (cmd0 > CANON["push_min"])
                st["run"] = st["run"] + 1 if stalled else 0
                st["maxrun"] = max(st["maxrun"], st["run"])
                for d in st["dets"].values():
                    d.step(proj, st["maxp"], cmd0, gw, st["steps"])
                st["maxp"] = max(st["maxp"], proj); st["maxproj"] = st["maxp"]
                st["lastproj"] = proj
        for st in live.values():
            if st["steps"]:
                st["ev"] = {k:d.close() for k,d in st["dets"].items()}; eps.append(st)
    return eps, n_start, n_out


def section(pl):
    if pl is None: return "?"
    return "CCA" if pl < 146.2 else "ICA-mid" if pl < 210.4 else "siphon"


def load_jsonl(p):
    out = {}
    for L in open(p):
        d = json.loads(L)
        out[d["seed"]] = d
    return out


if __name__ == "__main__":
    logdir, jsonlp, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    eps, ns, no = parse(logdir)
    jl = load_jsonl(jsonlp)
    for e in eps:
        j = jl.get(e["seed"])
        e["succ"] = bool(j["grader_success"]) if j else None
        e["sec"] = section(e["pl"])
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell3_%s.jsonl" % tag)
    with open(outp, "w") as fh:
        for e in eps:
            fh.write(json.dumps({k:v for k,v in e.items() if k not in ("dets",)})+chr(10))
    sys.stderr.write("%s: EPISODE_START=%d EPISODE_OUTCOME=%d episodes_emitted=%d jsonl=%d matched=%d\n"
                     % (tag, ns, no, len(eps), len(jl), sum(1 for e in eps if e["succ"] is not None)))
