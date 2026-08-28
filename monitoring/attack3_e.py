import os,re,glob,csv,json,statistics as st
from collections import Counter,defaultdict
V1BP=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
H0=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
HOLD={"topcowmr004","topcowmr008","topcowmr017","topcowmr023"}
ES=re.compile(r"EPISODE_START \| ep=(\d+).*?pid=(\d+) \| target=\(([-\d\.]+),([-\d\.]+),([-\d\.]+)\)")
def kv(s):
    d={}
    for p in s.split(" | "):
        if "=" in p: k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def collect(files,tmin=None,tmax=None):
    out=[]
    for f in files:
        live={}
        for line in open(f,errors="replace"):
            ts=line[:23]
            if tmin and not (tmin<=ts<=tmax): continue
            if "EPISODE_START" in line:
                d=kv(line[line.index("EPISODE_START"):])
                live[d["pid"]]={"mesh":d.get("mesh_fp"),"anat":d.get("anatomy"),
                    "seed":d.get("seed"),"tgt":d.get("target"),"pl":None,"steps":0,
                    "succ":False,"first":None,"last":None,"outcome":None,"pid":d["pid"]}
            elif " STEP |" in line:
                d=kv(line[line.index("STEP |"):]); s=live.get(d["pid"])
                if s is None: continue
                s["steps"]+=1
                if s["pl"] is None and d.get("path_len","?") not in ("?",None):
                    try: s["pl"]=float(d["path_len"])
                    except: pass
                if d.get("term")=="True" and d.get("trunc")=="False": s["succ"]=True
                if s["first"] is None: s["first"]=d
                s["last"]=d
            elif "EPISODE_OUTCOME" in line:
                d=kv(line[line.index("EPISODE_OUTCOME"):]); s=live.pop(d["pid"],None)
                if s is None: continue
                s["outcome"]=d.get("reason"); s["gs"]=d.get("grader_success")
                if d.get("reason")=="success": s["succ"]=True
                out.append(s)
        out.extend(live.values())
    return out
def sec(pl): return "CCA" if pl<146 else ("ICA-mid" if pl<210 else "siphon")
def rate(rows):
    d=defaultdict(lambda:[0,0])
    for r in rows: d[r["mesh"]][0]+=1; d[r["mesh"]][1]+=r["succ"]
    return d

vb=collect(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
h0=collect(sorted(glob.glob(os.path.join(H0,"diagnostics","logs_subprocesses","worker_*.log"))),
           "2026-08-28 03:51:29","2026-08-28 04:15:20")
h0=[r for r in h0 if r["mesh"] in HOLD]
print("V1BP n=%d k=%d  H0 n=%d k=%d"%(len(vb),sum(r["succ"] for r in vb),len(h0),sum(r["succ"] for r in h0)))
print("V1BP outcomes:",Counter(r["outcome"] for r in vb))
print("H0   outcomes:",Counter(r["outcome"] for r in h0))
# cross-check vs jsonl
jl={j["seed"]:j for j in map(json.loads,open(os.path.join(V1BP,"episodes_official_20260828_045651.jsonl")))}
bad=0
for r in vb:
    j=jl[int(r["seed"])]
    if bool(j["success"])!=bool(r["succ"]): bad+=1;print("MISMATCH seed",r["seed"],j["success"],r["succ"],r["outcome"])
print("log-vs-official success mismatches:",bad)
# final d_tgt
fd=[float(r["last"]["d_tgt"]) for r in vb if r["succ"] and r["last"] and r["last"].get("d_tgt") not in (None,"?")]
fdf=[float(r["last"]["d_tgt"]) for r in vb if not r["succ"] and r["last"] and r["last"].get("d_tgt") not in (None,"?")]
print("final d_tgt SUCCESS: n=%d min=%.2f max=%.2f  (threshold 5.0 mm)"%(len(fd),min(fd),max(fd)))
print("  count final d_tgt>=5.0 among successes:",sum(1 for x in fd if x>=5.0))
print("final d_tgt FAIL: n=%d sorted=%s"%(len(fdf),[round(x,1) for x in sorted(fdf)]))
# initial d_tgt / path_len relation -> offset derivation
print("\ninitial d_tgt (step1) vs path_len:")
for r in sorted(vb,key=lambda r:r["pl"])[:6]:
    print("  mesh=%s pl=%.1f d_tgt1=%s steps=%d succ=%d tgt=%s"%(r["mesh"],r["pl"],r["last"] and r["first"]["d_tgt"],r["steps"],r["succ"],r["tgt"]))
# per-anatomy both runs
for tag,rows in (("V1BP",vb),("H0",h0)):
    d=rate(rows); print("\n%s per-anatomy:"%tag)
    for m in sorted(d): print("   %s %d/%d = %.4f"%(m,d[m][1],d[m][0],d[m][1]/d[m][0]))
    print("   pooled %.4f  unweighted %.4f"%(sum(v[1] for v in d.values())/len(rows), sum(v[1]/v[0] for v in d.values())/len(d)))
# reweight cross
dv=rate(vb); dh=rate(h0)
alloc_h={m:dh[m][0] for m in dh}; alloc_v={m:dv[m][0] for m in dv}
print("\nV1BP scored under H0's allocation: %.4f"%(sum(alloc_h[m]*dv[m][1]/dv[m][0] for m in dv)/98))
print("H0   scored under V1BP allocation: %.4f"%(sum(alloc_v[m]*dh[m][1]/dh[m][0] for m in dh)/98))
# sections
for tag,rows in (("V1BP",vb),("H0",h0)):
    s=defaultdict(lambda:[0,0])
    for r in rows:
        if r["pl"] is None: continue
        s[sec(r["pl"])][0]+=1; s[sec(r["pl"])][1]+=r["succ"]
    print("%s sections: %s"%(tag,{k:(v[1],v[0],round(v[1]/v[0],3)) for k,v in sorted(s.items())}))
    pl=[r["pl"] for r in rows if r["pl"]]
    print("   path_len min=%.1f p50=%.1f max=%.1f"%(min(pl),sorted(pl)[len(pl)//2],max(pl)))
# H0 steps
hs=[r["steps"] for r in h0 if r["succ"]]
print("\nH0 success steps: n=%d min=%d p50=%d p90=%d max=%d"%(len(hs),min(hs),sorted(hs)[len(hs)//2],sorted(hs)[int(.9*len(hs))],max(hs)))
for cap in (200,300,400,500,600):
    print("   H0 succ<=%d: %d (%.4f)"%(cap,sum(1 for s in hs if s<=cap),sum(1 for s in hs if s<=cap)/98))
# per-anatomy path_len ranges (offset derivation pooled over both runs)
allr=vb+h0
pa=defaultdict(list)
for r in allr:
    if r["pl"]: pa[r["mesh"]].append(r["pl"])
print("\npooled(196) per-anatomy path_len: ", {m:(round(min(v),1),round(max(v),1),len(v)) for m,v in sorted(pa.items())})
