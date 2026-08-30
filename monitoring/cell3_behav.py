import glob,os,re,sys
D=sys.argv[1]
CMD=re.compile(r"cmd_action=\[([-0-9.]+),([-0-9.]+),([-0-9.]+),")
INS=re.compile(r"inserted=\[([-0-9.]+),([-0-9.]+)\]")
SLK=re.compile(r"cath_slack=([+-][0-9.]+)"); BUC=re.compile(r"buckle_phi=([+-][0-9.]+)")
PROJ=re.compile(r"proj_s=([-0-9.]+)"); FLD=re.compile(r"fold=([0-9]+)/")
n=0; push=0; pull=0; cg=[]; slk=[]; buc=[]; fld=0; gwtot=0.0; gwback=0.0; prev={}
for p in sorted(glob.glob(os.path.join(D,"worker_*.log"))):
    for line in open(p,errors="replace"):
        if " STEP |" not in line: continue
        i=line.find("pid="); pid=line[i+4:].split(" ")[0].strip()
        mc=CMD.search(line); mi=INS.search(line)
        if not(mc and mi): continue
        c0=float(mc.group(1)); gw=float(mi.group(1))
        n+=1; cg.append(c0)
        if c0>2.0: push+=1
        if c0<-2.0: pull+=1
        ms=SLK.search(line); mb=BUC.search(line); mf=FLD.search(line)
        if ms: slk.append(float(ms.group(1)))
        if mb: buc.append(abs(float(mb.group(1))))
        if mf and int(mf.group(1))>0: fld+=1
        pv=prev.get(pid)
        if pv is not None:
            d=gw-pv
            if d>0: gwtot+=d
            else: gwback+= -d
        prev[pid]=gw
        if "EPISODE_OUTCOME" in line: prev.pop(pid,None)
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*(len(v)-1)))]
print("  steps=%d  push_duty(cmd_gw>2)=%.1f%%  pull_duty(cmd_gw<-2)=%.1f%%  cmd_gw median %.2f p10 %.2f p90 %.2f"%(
    n,100*push/n,100*pull/n,q(cg,.5),q(cg,.1),q(cg,.9)))
print("  cath_slack median %.1f p90 %.1f max %.1f | |buckle_phi| median %.3f p90 %.3f | fold>0 in %.1f%% steps"%(
    q(slk,.5),q(slk,.9),max(slk),q(buc,.5),q(buc,.9),100*fld/n))
print("  gw advanced total %.0f mm, withdrawn total %.0f mm, withdraw/advance = %.3f"%(gwtot,gwback,gwback/gwtot))
