import glob,os,re,sys,json,statistics as S
D=sys.argv[1]
PROJ=re.compile(r"proj_s=([-0-9.]+)"); DTG=re.compile(r"d_tgt=([0-9.]+)")
SLK=re.compile(r"cath_slack=([+-][0-9.]+)"); BUC=re.compile(r"buckle_phi=([+-][0-9.]+)")
FLD=re.compile(r"fold=([0-9]+)/"); ONP=re.compile(r"on_path=([01])"); OFB=re.compile(r"off_br=([01])")
XT=re.compile(r"xt_true=([0-9.]+)"); CMD=re.compile(r"cmd_action=\[([-0-9.]+),([-0-9.]+),([-0-9.]+),")
INS=re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
W=141.2
rows={'wall':[], 'other':[]}
# per-episode: while proj_s pinned at 141.2 for >=12 consecutive steps, track d_tgt drift
pin={}
for p in sorted(glob.glob(os.path.join(D,"worker_*.log"))):
    live={}
    for line in open(p,errors="replace"):
        if "EPISODE_START" in line:
            i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip(); live[pid]={'run':0,'d0':None,'dlast':None,'best':None,'g0':None,'glast':None}
            continue
        if " STEP |" not in line: continue
        i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip(); st=live.get(pid)
        if st is None: continue
        mp=PROJ.search(line)
        if not mp: continue
        pr=float(mp.group(1))
        md=DTG.search(line); mi=INS.search(line); mc=CMD.search(line)
        if not(md and mi and mc): continue
        d=float(md.group(1)); gw=float(mi.group(1)); c0=float(mc.group(1))
        key='wall' if abs(pr-W)<0.05 else 'other'
        ms=SLK.search(line); mb=BUC.search(line); mf=FLD.search(line); mx=XT.search(line); mo=OFB.search(line)
        rows[key].append((float(ms.group(1)) if ms else 0.0, abs(float(mb.group(1))) if mb else 0.0,
                          int(mf.group(1)) if mf else 0, float(mx.group(1)) if mx else 0.0,
                          int(mo.group(1)) if mo else 0, c0, d, gw))
        if key=='wall' and c0>2.0:
            st['run']+=1
            if st['run']==1: st['d0']=d; st['g0']=gw
            st['dlast']=d; st['glast']=gw
            st['best']=d if st['best'] is None else min(st['best'],d)
        else:
            if st['run']>=12:
                pin.setdefault('runs',[]).append((st['run'],st['d0'],st['dlast'],st['best'],st['g0'],st['glast']))
            st['run']=0; st['d0']=None; st['best']=None
    for st in live.values():
        if st['run']>=12: pin.setdefault('runs',[]).append((st['run'],st['d0'],st['dlast'],st['best'],st['g0'],st['glast']))
def st_(v): 
    v=sorted(v); return (v[len(v)//2], v[int(.9*(len(v)-1))])
for k in ['wall','other']:
    R=rows[k]
    if not R: continue
    print("%-6s n_steps=%6d  slack med %.1f p90 %.1f | |buckle| med %.3f p90 %.3f | fold med %d p90 %d | xt_true med %.2f p90 %.2f | off_br frac %.3f | cmd_gw med %.1f"%(
        k,len(R),*st_([r[0] for r in R]),*st_([r[1] for r in R]),*st_([r[2] for r in R]),*st_([r[3] for r in R]),
        sum(r[4] for r in R)/len(R), st_([r[5] for r in R])[0]))
rr=pin.get('runs',[])
print("PINNED-AT-141.2 push runs >=12 steps: n=%d"%len(rr))
if rr:
    dd=[r[1]-r[3] for r in rr]   # d_tgt improvement from run start to best during pin
    gg=[r[5]-r[4] for r in rr]
    print("  run length: median %d p90 %d max %d"%(*st_([r[0] for r in rr]),max(r[0] for r in rr)))
    print("  d_tgt(start)-d_tgt(best) during pin: median %.2f mm  p90 %.2f  max %.2f  (>2mm in %d/%d)"%(
        *st_(dd),max(dd),sum(1 for v in dd if v>2),len(dd)))
    print("  gw inserted net change over pin: median %.1f mm p90 %.1f"%st_(gg))
