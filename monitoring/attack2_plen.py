import glob,os,collections
import numpy as np
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
for stamp,lab in (("20260826_180252","HOST"),("20260828_045651","TB4")):
    eps=parse(stamp); pl=np.array([e["pl"] for e in eps.values() if "pl" in e])
    print("%-5s n=%d  path_len min=%.1f p25=%.1f med=%.1f p75=%.1f max=%.1f  -> RCCA arclen %.1f..%.1f"%(
      lab,len(pl),pl.min(),np.percentile(pl,25),np.median(pl),np.percentile(pl,75),pl.max(),pl.min()-33.5,pl.max()-33.5))
    print("      distinct targets:",len({e.get("tgt") for e in eps.values()}))
    h,_=np.histogram(pl,bins=[70,110,146,180,210,240,275])
    print("      hist [70,110,146,180,210,240,275):",h.tolist())
