import os,re,glob,collections,statistics,json
BASE=r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
def kv(s):
    d={}
    for p in s.split(" | ")[1:]:
        if "=" in p:
            k,v=p.split("=",1); d[k.strip()]=v.strip()
    return d
def parse(tag):
    LOG=os.path.join(BASE,"logs",tag)
    eps={}; steps=collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(LOG,"*"))):
        if not os.path.isfile(f): continue
        for line in open(f,errors="replace"):
            if "EPISODE_START" in line:
                d=kv(line[line.index("EPISODE_START"):])
                if "pid" in d and "ep" in d: eps[(int(d["pid"]),int(d["ep"]))]=dict(start=d,outcome=None)
            elif "EPISODE_OUTCOME" in line:
                d=kv(line[line.index("EPISODE_OUTCOME"):])
                k=(int(d["pid"]),int(d["ep"]))
                if k in eps: eps[k]["outcome"]=d
            elif " - STEP | " in line:
                d=kv(line[line.index("STEP |"):])
                steps[(int(d["pid"]),int(d["ep"]))].append(d)
    return eps,steps
for tag,jl in (("20260828_045651","episodes_official_20260828_045651.jsonl"),
               ("20260828_053306","episodes_official_20260828_053306.jsonl")):
    eps,steps=parse(tag)
    js=[json.loads(l) for l in open(os.path.join(BASE,jl))]
    print("===",tag,"starts",len(eps),"outcomes",sum(1 for v in eps.values() if v['outcome']),"jsonl",len(js),"jsonl_succ",sum(1 for r in js if r['success']))
    # map by seed
    byseed=collections.defaultdict(list)
    for k,v in eps.items():
        s=v["start"].get("seed")
        byseed[s].append(k)
    dup=[s for s,v in byseed.items() if len(v)>1]
    print("  unique seeds:",len(byseed),"dups:",len(dup))
    rows=[]
    for r in js:
        s=str(r["seed"])
        ks=byseed.get(s,[])
        if len(ks)!=1:
            rows.append(dict(seed=r["seed"],succ=int(r["success"]),mesh=None,path_len=None,maxps=None)); continue
        k=ks[0]; ss=steps[k]
        rows.append(dict(seed=r["seed"],succ=int(r["success"]),pid=k[0],ep=k[1],
                         mesh=eps[k]["start"].get("mesh_fp"),
                         path_len=float(ss[0]["path_len"]) if ss else None,
                         maxps=max(float(x["proj_s"]) for x in ss) if ss else None,
                         finps=float(ss[-1]["proj_s"]) if ss else None,
                         n=len(ss),
                         fb=eps[k]["outcome"]["final_branch"] if eps[k]["outcome"] else None))
    print("  joined w/ mesh:",sum(1 for r in rows if r["mesh"]))
    print("  mesh counts:",collections.Counter(r["mesh"] for r in rows))
    json.dump(rows,open(os.path.join(r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad","teach_%s.json"%tag),"w"))
