import glob,json,re,sys
BS=chr(92)
PR=re.compile(r"proj_s=([-0-9.]+)")
DI=re.compile(r"delta_ins="+BS+r"[([-0-9.]+),([-0-9.]+)"+BS+r"]")
CM=re.compile(r"cmd_action="+BS+r"[([-0-9.]+),([-0-9.]+),([-0-9.]+),([-0-9.]+)"+BS+r"]")
IN=re.compile(r"inserted="+BS+r"[([-0-9.]+),([-0-9.]+)"+BS+r"]")
SD=re.compile(r"seed=(\d+)"); FO=re.compile(r"fold=(\d+)/"); SL=re.compile(r"cath_slack=([-+0-9.]+)")
OB=re.compile(r"off_br=(\d+)"); OP=re.compile(r"on_path=(\d+)")
out={}
for p in sorted(glob.glob(sys.argv[1]+"/worker_*.log")):
    live={}
    for line in open(p,errors="replace"):
        i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip() if i>=0 else "?"
        if "EPISODE_START" in line:
            pv=live.pop(pid,None)
            if pv and pv["n"]: out[pv["seed"]]=pv
            m=SD.search(line); live[pid]={"seed":int(m.group(1)) if m else None,
              "n":0,"push":0,"pull":0,"idle":0,"absd":0.0,"net":0.0,"rotmag":0.0,
              "maxfold":0,"offbr":0,"offpath":0,"maxslack":0.0,"finalproj":0.0,"maxproj":-1e9,
              "cmdpos":0,"cmdneg":0}
            continue
        st=live.get(pid)
        if st is None or " STEP |" not in line: continue
        md=DI.search(line); mc=CM.search(line); mp=PR.search(line)
        if not(md and mc and mp): continue
        d=float(md.group(1)); st["n"]+=1
        if d>0.264: st["push"]+=1
        elif d<-0.264: st["pull"]+=1
        else: st["idle"]+=1
        st["absd"]+=abs(d); st["net"]+=d
        c0=float(mc.group(1))
        if c0>0: st["cmdpos"]+=1
        else: st["cmdneg"]+=1
        st["rotmag"]+=abs(float(mc.group(2)))
        v=float(mp.group(1)); st["finalproj"]=v; st["maxproj"]=max(st["maxproj"],v)
        mf=FO.search(line)
        if mf: st["maxfold"]=max(st["maxfold"],int(mf.group(1)))
        mo=OB.search(line)
        if mo and mo.group(1)!="0": st["offbr"]+=1
        mq=OP.search(line)
        if mq and mq.group(1)=="0": st["offpath"]+=1
        ms=SL.search(line)
        if ms: st["maxslack"]=max(st["maxslack"],float(ms.group(1)))
    for st in live.values():
        if st["n"]: out[st["seed"]]=st
    # also flush on EPISODE_START handled by overwrite; capture all
json.dump(out, open(sys.argv[2],"w"))
print("eps=%d"%len(out))
