import glob,os,collections
R="saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
def parse(stamp):
    eps={}
    for f in glob.glob(os.path.join(R,"logs",stamp,"*.log")):
        for ln in open(f,errors="ignore"):
            if " STEP | " not in ln: continue
            d=dict(kv.split("=",1) for kv in (x.strip() for x in ln.split(" | ")[1:]) if "=" in kv)
            if "pid" not in d or "ep" not in d: continue
            e=eps.setdefault((d["pid"],int(d["ep"])),{})
            if d.get("path_len") not in (None,"inf"): e["pl"]=float(d["path_len"])
            e["term"]=d.get("term","False"); e["tgt"]=d.get("tgt")
    return eps
H=parse("20260826_180252"); T=parse("20260828_045651")
ht={}; tt={}
for e in H.values(): ht.setdefault(e.get("tgt"),[]).append((e.get("pl"),e["term"]=="True"))
for e in T.values(): tt.setdefault(e.get("tgt"),[]).append((e.get("pl"),e["term"]=="True"))
common=set(ht)&set(tt)-{None}
print("distinct targets: HOST=%d  TB4=%d  identical 3D target points in BOTH runs=%d"%(len(ht),len(tt),len(common)))
if common:
    hs=sum(s for c in common for _,s in ht[c]); hn=sum(len(ht[c]) for c in common)
    ts=sum(s for c in common for _,s in tt[c]); tn=sum(len(tt[c]) for c in common)
    print("  on those shared targets: HOST %d/%d = %.1f%%   TB4 %d/%d = %.1f%%"%(hs,hn,100*hs/hn,ts,tn,100*ts/tn))
    pls=sorted(ht[c][0][0] for c in common)
    print("  their path_len range: %.1f .. %.1f"%(pls[0],pls[-1]))
    print("  per-target: pl, host succ/n, tb succ/n")
    for c in sorted(common,key=lambda c: ht[c][0][0]):
        print("   %6.1f  H %d/%d   T %d/%d   %s"%(ht[c][0][0],sum(s for _,s in ht[c]),len(ht[c]),
              sum(s for _,s in tt[c]),len(tt[c]),c))
