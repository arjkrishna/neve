import glob,os,re,collections,sys
R="saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
def parse(stamp):
    eps={}
    for f in glob.glob(os.path.join(R,"logs",stamp,"*.log")):
        for ln in open(f,errors="ignore"):
            if " STEP | " not in ln: continue
            d=dict(kv.split("=",1) for kv in (x.strip() for x in ln.split(" | ")[1:]) if "=" in kv)
            try: key=(d["pid"],int(d["ep"]))
            except KeyError: continue
            e=eps.setdefault(key,{"pl":None,"n":0,"term":"False","trunc":"False","maxs":0.0,"reward":0.0})
            e["n"]=max(e["n"],int(d["ep_step"]))
            if d.get("path_len") not in (None,"inf"): e["pl"]=float(d["path_len"])
            e["term"]=d.get("term",e["term"]); e["trunc"]=d.get("trunc",e["trunc"])
            try: e["maxs"]=max(e["maxs"],float(d.get("proj_s","0")))
            except ValueError: pass
            try: e["reward"]=float(d.get("cum_reward",0))
            except ValueError: pass
    return eps
def sec(pl): return "CCA" if pl<146 else ("ICA-mid" if pl<210 else "siphon")
for stamp,lab in (("20260826_180252","HOST real patient"),("20260828_045651","TopBrain 4 holdout")):
    eps=parse(stamp)
    tot=collections.defaultdict(lambda:[0,0])
    fails=[]
    for k,e in eps.items():
        if e["pl"] is None: continue
        s=sec(e["pl"]); ok = e["term"]=="True"
        tot[s][1]+=1; tot[s][0]+=ok
        if not ok: fails.append((e["pl"],e["maxs"],e["n"]))
    n=sum(v[1] for v in tot.values()); g=sum(v[0] for v in tot.values())
    print("%-22s episodes=%d  term(success)=%d  %.1f%%"%(lab,n,g,100*g/max(n,1)))
    for s in ("CCA","ICA-mid","siphon"):
        v=tot[s]; print("   %-8s %3d/%-3d = %5.1f%%"%(s,v[0],v[1],100*v[0]/max(v[1],1)))
    fails.sort()
    print("   failures (path_len, max proj_s reached, steps):")
    for pl,ms,nn in fails: print("      %6.1f  %6.1f  %4d"%(pl,ms,nn))
    print()
