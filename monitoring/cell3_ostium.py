import glob,os,re,sys,json
PROJ=re.compile(r"proj_s=([-0-9.]+)"); CB=re.compile(r"cur_branch=([^|]*)\|")
lo=[];hi=[]
for logdir,tag in [(sys.argv[1],'a')]:
    for p in sorted(glob.glob(os.path.join(logdir,"worker_*.log"))):
        live={}
        for line in open(p,errors="replace"):
            if "EPISODE_START" in line:
                i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip()
                pv=live.pop(pid,None)
                if pv:
                    if pv[0] is not None: lo.append(pv[0])
                    if pv[1] is not None: hi.append(pv[1])
                live[pid]=[None,None]
                continue
            if " STEP |" not in line: continue
            i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip()
            st=live.get(pid)
            if st is None or st[1] is not None: continue
            mp=PROJ.search(line); mb=CB.search(line)
            if not(mp and mb): continue
            v=float(mp.group(1)); b=mb.group(1)
            if "RCCA" in b: st[1]=v
            else: st[0]=v
        for st in live.values():
            if st[0] is not None: lo.append(st[0])
            if st[1] is not None: hi.append(st[1])
lo.sort(); hi.sort()
print("last-parent proj_s : n=%d max=%.2f p90=%.2f median=%.2f"%(len(lo),lo[-1],lo[int(.9*len(lo))],lo[len(lo)//2]))
print("first-RCCA proj_s  : n=%d min=%.2f p10=%.2f median=%.2f"%(len(hi),hi[0],hi[int(.1*len(hi))],hi[len(hi)//2]))
print("=> OFF_host bracket [%.2f, %.2f]  midpoint %.2f"%(lo[-1],hi[0],(lo[-1]+hi[0])/2))
