"""Dump compact per-episode step traces + canon stall events for offline analysis."""
import glob, json, os, re, sys
C = dict(stall_eps=0.3, push_min=2.0, stuck_steps=12, retract_min=1.0, soft_max=8.0, pass_eps=1.0)
PROJ=re.compile(r"proj_s=([-0-9.]+)"); CMD=re.compile(r"cmd_action=\[([-0-9.]+),")
INS=re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]"); FOLD=re.compile(r"fold=(\d+)/")
PL=re.compile(r"path_len=([0-9.]+)"); SEED=re.compile(r"seed=(\d+)")
def new(seed): return dict(seed=seed,succ=False,reason=None,pl=None,proj=[],gw=[],fold=[],cmd=[],
    maxp=-1e9,stall=0,stuck=False,first=0,gw_peak=0.,gw_min=0.,retract=0.,p0=0.,onset=0,events=[])
def step(st,proj,cmd0,gw,i):
    if st["stuck"]:
        st["gw_min"]=min(st["gw_min"],gw); st["retract"]=max(st["retract"],st["gw_peak"]-st["gw_min"])
        if proj>st["p0"]+C["pass_eps"]:
            r=st["retract"]; k="grind" if r<C["retract_min"] else "soft" if r<=C["soft_max"] else "hard"
            st["events"].append(dict(k=k,r=round(r,3),close=i,onset=st["onset"],first=st["first"],p0=st["p0"]))
            st["stuck"]=False; st["stall"]=0
    else:
        stalled=(proj<st["maxp"]+C["stall_eps"]) and (cmd0>C["push_min"])
        if stalled:
            if st["stall"]==0: st["first"]=i
            st["stall"]+=1
        else: st["stall"]=max(0,st["stall"]-2)
        if st["stall"]>=C["stuck_steps"]:
            st["stuck"]=True; st["onset"]=i; st["p0"]=st["maxp"]; st["gw_peak"]=st["gw_min"]=gw; st["retract"]=0.
def emit(st,out):
    if st["pl"] is None or not st["proj"]: return 0
    if st["stuck"]: st["events"].append(dict(k="unrec",r=round(st["retract"],3),close=-1,onset=st["onset"],first=st["first"],p0=st["p0"]))
    out.write(json.dumps(dict(seed=st["seed"],succ=st["succ"],reason=st["reason"],pl=st["pl"],
        proj=[round(x,2) for x in st["proj"]],gw=[round(x,2) for x in st["gw"]],
        fold=st["fold"],cmd=[round(x,2) for x in st["cmd"]],events=st["events"]))+"\n")
    return 1
def run(d,o):
    out=open(o,"w"); n=0
    for path in sorted(glob.glob(os.path.join(d,"worker_*.log"))):
        live={}
        for line in open(path,errors="replace"):
            if "EPISODE_START" in line:
                p=line.find("pid="); pid=line[p+4:].split(" ")[0].strip()
                if pid in live: n+=emit(live.pop(pid),out)
                m=SEED.search(line); live[pid]=new(int(m.group(1)) if m else None); continue
            if " STEP |" not in line and "EPISODE_OUTCOME" not in line: continue
            p=line.find("pid="); pid=line[p+4:].split(" ")[0].strip(); st=live.get(pid)
            if st is None: continue
            if "EPISODE_OUTCOME" in line:
                i=line.find("reason=")
                if i>=0:
                    st["reason"]=line[i+7:].split(" ")[0].strip()
                    if st["reason"]=="success": st["succ"]=True
                n+=emit(st,out); live.pop(pid,None); continue
            mp,mc,mi=PROJ.search(line),CMD.search(line),INS.search(line)
            if not(mp and mc and mi): continue
            pr=float(mp.group(1)); c0=abs(float(mc.group(1))); g=float(mi.group(1))
            mf=FOLD.search(line)
            if st["pl"] is None:
                m=PL.search(line)
                if m: st["pl"]=float(m.group(1))
            st["proj"].append(pr); st["gw"].append(g); st["fold"].append(int(mf.group(1)) if mf else 0); st["cmd"].append(c0)
            step(st,pr,c0,g,len(st["proj"])); st["maxp"]=max(st["maxp"],pr)
            if "term=True" in line and "trunc=False" in line: st["succ"]=True
        for st in list(live.values()): n+=emit(st,out)
    out.close(); sys.stderr.write("episodes=%d\n"%n)
if __name__=="__main__": run(sys.argv[1],sys.argv[2])
