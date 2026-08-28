import os, re, glob, json, statistics as st
from collections import Counter, defaultdict
V1BP = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
H0   = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
HOLD={"topcowmr004","topcowmr008","topcowmr017","topcowmr023"}
def kv(s):
    d={}
    for p in s.split(" | "):
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
ACT=re.compile(r"cmd_action=\[([^\]]*)\]")

def scan(files, tmin=None, tmax=None):
    """episode-aware scan using ep + ep_step continuity per (pid, env-slot)."""
    episodes=[]
    for f in files:
        slots=defaultdict(dict)  # pid -> {slotid: ep dict}
        for line in open(f, errors="replace"):
            ts=line[:23]
            if tmin and (ts<tmin or ts>tmax): continue
            if "EPISODE_START" in line:
                d=kv(line[line.index("EPISODE_START"):]); pid=d.get("pid")
                e={"mesh":d.get("mesh_fp"),"anat":d.get("anatomy"),"seed":d.get("seed"),
                   "ep":d.get("ep"),"target":d.get("target"),"pid":pid,"file":f,
                   "acts":[],"nsteps":0,"last":None,"maxstep":0}
                slots[pid][("ep",d.get("ep"),len(slots[pid]))]=e
                # keep list of open eps for pid
                slots[pid].setdefault("_open",[]).append(e)
                episodes.append(e)
            elif "STEP |" in line:
                d=kv(line[line.index("STEP |"):]); pid=d.get("pid")
                op=[e for e in slots[pid].get("_open",[]) if e["ep"]==d.get("ep")]
                if len(op)>1:
                    op=[e for e in op if e["maxstep"]==int(d["ep_step"])-1] or op
                if not op: continue
                e=op[0]
                m=ACT.search(line)
                if m:
                    try: e["acts"].append([float(x) for x in m.group(1).split(",")])
                    except: pass
                e["nsteps"]+=1; e["maxstep"]=int(d["ep_step"]); e["last"]=d
            elif "EPISODE_END" in line:
                d=kv(line[line.index("EPISODE_END"):]); pid=d.get("pid")
                op=[e for e in slots[pid].get("_open",[]) if e["ep"]==d.get("ep")]
                if op:
                    e=op[0]; e["end"]=d; slots[pid]["_open"].remove(e)
    return episodes

def summ(vals,name,f="{:.3f}"):
    if not vals: return name+": n=0"
    v=sorted(vals); n=len(v); q=lambda p: v[min(n-1,int(p*n))]
    return (name+f": n={n} mean="+f+" sd="+f+" min="+f+" p10="+f+" p50="+f+" p90="+f+" max="+f).format(
        sum(v)/n, st.pstdev(v) if n>1 else 0.0, v[0], q(.1), q(.5), q(.9), v[-1])

vb=scan(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
h0=scan(sorted(glob.glob(os.path.join(H0,"diagnostics","logs_subprocesses","worker_*.log"))),
        "2026-08-28 03:51:29","2026-08-28 04:15:20")
print("V1BP eps:",len(vb)," H0-window eps:",len(h0))
print("V1BP meshes:",Counter(e["mesh"] for e in vb))
print("H0 meshes:",Counter(e["mesh"] for e in h0))
print("H0 non-holdout in window:",Counter(e["mesh"] for e in h0 if e["mesh"] not in HOLD))
h0=[e for e in h0 if e["mesh"] in HOLD]
print("H0 hold eps:",len(h0))

for tag,S in (("V1BP",vb),("H0",h0)):
    A=[a for e in S for a in e["acts"]]
    print(f"\n--- {tag}: total logged steps={len(A)} episodes={len(S)}")
    for i,nm in enumerate(["gw_trans","gw_rot","cath_trans","cath_rot"]):
        col=[a[i] for a in A if len(a)==4]
        print("  ",summ(col,nm))
    # saturation at limits
    lim=[9.0,0.45,9.0,0.45]
    print("   NOTE raw units differ; fraction |v|>9.0 in col0:",
          sum(1 for a in A if abs(a[0])>9.0)/max(1,len(A)))
import pickle
pickle.dump({"vb":vb,"h0":h0}, open(r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad\a3.pkl","wb"))
