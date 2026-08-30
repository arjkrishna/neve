"""Cell-4 stall + coil extractor. Reuses monitoring/extract_stuck.py Det logic verbatim
for the canonical config; adds stuck_steps sweep, raw longest-stall-run, slack/coil stats.
Read-only. Usage: python cell4_stall_coil.py <spec.json> <out.jsonl>
"""
import json, re, os, sys, glob

CANON = dict(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0, soft_max=8.0, pass_eps=1.0)
SWEEP = [4, 6, 8, 12]

PROJ = re.compile(r"proj_s=([-0-9.]+)")
CMD  = re.compile(r"cmd_action=\[([-0-9.]+),")
INS  = re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
DINS = re.compile(r"delta_ins=\[([-0-9.]+),([-0-9.]+)\]")
PL   = re.compile(r"path_len=([0-9.]+)")
CSL  = re.compile(r"cath_slack=([-+0-9.]+)")
WT   = re.compile(r"wall_time=([0-9.]+)")
SEED = re.compile(r" seed=(\S+)")
GS   = re.compile(r"grader_success=(\d)")
FOLD = re.compile(r"fold=(\d+)/")
BUCK = re.compile(r"buckle_phi=([-+0-9.]+)")
RSN  = re.compile(r"reason=(\S+)")


class Det:
    __slots__ = ("c","stall","stuck","gw_peak","gw_min","retract","p0","events","onsets","onset_steps")
    def __init__(self, c):
        self.c=c; self.stall=0; self.stuck=False
        self.gw_peak=self.gw_min=self.retract=0.0; self.p0=0.0
        self.events=[]; self.onsets=[]; self.onset_steps=[]
    def step(self, proj, maxp, cmd0, gw, ep_step):
        c=self.c
        if self.stuck:
            self.gw_min=min(self.gw_min,gw)
            self.retract=max(self.retract,self.gw_peak-self.gw_min)
            if proj > self.p0 + c["pass_eps"]:
                r=self.retract
                kind=("grind" if r < c["retract_min"] else "soft" if r <= c["soft_max"] else "hard")
                self.events.append({"k":kind,"r":round(r,2),"s":ep_step})
                self.stuck=False; self.stall=0
        else:
            stalled=(proj < maxp + c["stall_eps"]) and (cmd0 > c["push_min"])
            self.stall = self.stall+1 if stalled else max(0,self.stall-2)
            if self.stall >= c["stuck_steps"]:
                self.stuck=True; self.p0=maxp
                self.gw_peak=self.gw_min=gw; self.retract=0.0
                self.onsets.append(round(maxp,2))
                self.onset_steps.append(ep_step)
    def close(self):
        if self.stuck:
            self.events.append({"k":"unrec","r":round(self.retract,2),"s":-1})
        return self.events


def new_ep(pid, t0, seed):
    d = {"pid":pid,"t":t0,"seed":seed,"steps":0,"pl":None,
         "succ":None,"reason":None,
         "gwslack_max":-1e9,"cathslack_max":-1e9,
         "gw_max":0.0,"proj_max":-1e9,"dins_sum":0.0,"dins_n":0,
         "raw_run":0,"raw_run_max":0,"ost_s":None,
         "sl_onpath":-1e9,"sl_offbr":-1e9,"n_off":0,"fold_max":0,"buck_max":0.0,
         "cath50_step":None,"onset_steps":[],
         "maxp":-1e9,
         "dets":{k:Det(dict(CANON, stuck_steps=k)) for k in SWEEP}}
    return d


def process(files, out, label, mode, win=None):
    """mode: 'all' | 'seeded' | 'unseeded'; win=(t0,t1) extra filter on wall_time."""
    n_start=n_out=n_emit=0
    for path in sorted(files):
        live={}
        for line in open(path, errors="replace"):
            if "EPISODE_START" in line:
                i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip() if i>=0 else "?"
                prev=live.pop(pid,None)
                if prev is not None: n_emit+=emit(prev,out,label)
                m=WT.search(line); t0=float(m.group(1)) if m else 0.0
                ms=SEED.search(line); seed=ms.group(1) if ms else None
                n_start+=1
                keep = (mode=="all") or (mode=="seeded" and seed is not None) or (mode=="unseeded" and seed is None)
                if keep and win is not None:
                    keep = win[0] <= t0 <= win[1]
                live[pid]= new_ep(pid,t0,seed) if keep else None
                continue
            isstep = " STEP |" in line
            if not isstep and "EPISODE_OUTCOME" not in line: continue
            i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip() if i>=0 else "?"
            if pid not in live: continue
            st=live[pid]
            if st is None:
                if not isstep: live.pop(pid,None)
                continue
            if not isstep:
                n_out+=1
                m=GS.search(line)
                if m: st["succ"]= (m.group(1)=="1")
                m=RSN.search(line)
                if m: st["reason"]=m.group(1)
                n_emit+=emit(st,out,label); live.pop(pid,None); continue
            mp=PROJ.search(line); mc=CMD.search(line); mi=INS.search(line)
            if not (mp and mc and mi): continue
            proj=float(mp.group(1)); cmd0=abs(float(mc.group(1)))
            gw=float(mi.group(1))
            st["steps"]+=1
            if st["pl"] is None:
                m=PL.search(line)
                if m: st["pl"]=float(m.group(1))
            md=DINS.search(line)
            if md:
                st["dins_sum"]+=float(md.group(1)); st["dins_n"]+=1
            mcs=CSL.search(line)
            if mcs: st["cathslack_max"]=max(st["cathslack_max"],float(mcs.group(1)))
            sl = gw - proj
            if sl > st["gwslack_max"]: st["gwslack_max"]=sl
            onp = "| on_path=1 |" in line
            offb = "| off_br=1 |" in line
            if onp and not offb:
                if sl > st["sl_onpath"]: st["sl_onpath"]=sl
            if offb:
                st["n_off"]+=1
                if sl > st["sl_offbr"]: st["sl_offbr"]=sl
            mf=FOLD.search(line)
            if mf: st["fold_max"]=max(st["fold_max"],int(mf.group(1)))
            mb=BUCK.search(line)
            if mb: st["buck_max"]=max(st["buck_max"],abs(float(mb.group(1))))
            if mcs and st["cath50_step"] is None and float(mcs.group(1))>50: st["cath50_step"]=st["steps"]
            st["gw_max"]=max(st["gw_max"],gw); st["proj_max"]=max(st["proj_max"],proj)
            if st["ost_s"] is None and "cur_branch=Centerline curve - RCCA" in line:
                st["ost_s"]=proj
            raw = (proj < st["maxp"] + CANON["stall_eps"]) and (cmd0 > CANON["push_min"])
            if raw:
                st["raw_run"]+=1
                if st["raw_run"]>st["raw_run_max"]: st["raw_run_max"]=st["raw_run"]
            else:
                st["raw_run"]=0
            for d in st["dets"].values():
                d.step(proj, st["maxp"], cmd0, gw, st["steps"])
            st["maxp"]=max(st["maxp"],proj)
        for st in live.values():
            if st is not None: n_emit+=emit(st,out,label)
    sys.stderr.write("%s starts=%d outcomes=%d emitted=%d\n"%(label,n_start,n_out,n_emit))


def emit(st,out,label):
    if st["steps"]==0: return 0
    rec={"run":label,"pid":st["pid"],"t":st["t"],"seed":st["seed"],
         "steps":st["steps"],"pl":st["pl"],"succ":st["succ"],"reason":st["reason"],
         "gwslack_max":round(st["gwslack_max"],2),
         "cathslack_max":round(st["cathslack_max"],2),
         "gw_max":round(st["gw_max"],2),"proj_max":round(st["proj_max"],2),
         "dins_mean":round(st["dins_sum"]/max(1,st["dins_n"]),4),
         "raw_run_max":st["raw_run_max"],"ost_s":st["ost_s"],
         "ev":{str(k):d.close() for k,d in st["dets"].items()},
         "onsets":st["dets"][12].onsets,"onset_steps":st["dets"][12].onset_steps,
         "sl_onpath":round(st["sl_onpath"],2),"sl_offbr":round(st["sl_offbr"],2),
         "frac_off":round(st["n_off"]/max(1,st["steps"]),3),"fold_max":st["fold_max"],
         "buck_max":round(st["buck_max"],3),"cath50_step":st["cath50_step"]}
    out.write(json.dumps(rec)+"\n")
    return 1


if __name__=="__main__":
    spec=json.load(open(sys.argv[1]))
    out=open(sys.argv[2],"w")
    for s in spec:
        files=[]
        for g in s["globs"]: files+=glob.glob(g)
        process(files,out,s["label"],s.get("mode","all"),s.get("win"))
    out.close()
