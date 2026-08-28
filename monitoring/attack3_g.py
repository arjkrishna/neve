import os,re,glob,json,statistics as st
from collections import Counter,defaultdict
exec(open(r"d:/Arjun/workspace/neve/monitoring/attack3_e.py").read().split("vb=collect")[0])
V1BP=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
H0=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
vb=collect(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
h0=[r for r in collect(sorted(glob.glob(os.path.join(H0,"diagnostics","logs_subprocesses","worker_*.log"))),
    "2026-08-28 03:51:29","2026-08-28 04:15:20") if r["mesh"] in HOLD]
for r in h0: r["succ"]=(r["outcome"]=="success")
# --- targets: uniqueness / co-location across anatomies
tg=defaultdict(set)
for r in vb+h0: tg[r["mesh"]].add(r["tgt"])
print("distinct target coords per anatomy (196 eps):",{m:len(v) for m,v in sorted(tg.items())})
alltg={}
for r in vb+h0: alltg.setdefault(r["tgt"],set()).add(r["mesh"])
shared=[t for t,ms in alltg.items() if len(ms)>1]
print("target coords appearing in >1 anatomy: %d of %d distinct"%(len(shared),len(alltg)))
plof={}
for r in vb+h0: plof[r["tgt"]]=r["pl"]
sp=[plof[t] for t in shared]
if sp: print("   their path_len: min=%.1f max=%.1f  (seam at path_len 163.4)"%(min(sp),max(sp)))
# --- per-section cap sensitivity (V1BP)
print("\nV1BP cap sensitivity by section (n per section fixed):")
bys=defaultdict(list)
for r in vb: bys[sec(r["pl"])].append(r)
for s in ("CCA","ICA-mid","siphon"):
    rows=bys[s]; n=len(rows)
    line=[]
    for cap in (150,200,300,400,500,600):
        k=sum(1 for r in rows if r["succ"] and r["steps"]<=cap)
        line.append("%d:%d/%d=%.2f"%(cap,k,n,k/n))
    print("  %-8s %s"%(s," ".join(line)))
print("  OVERALL  "+" ".join("%d:%d/98=%.3f"%(c,sum(1 for r in vb if r["succ"] and r["steps"]<=c),
      sum(1 for r in vb if r["succ"] and r["steps"]<=c)/98) for c in (150,200,300,400,500,600)))
# --- action saturation
def acts(rows,fdir,tmin=None,tmax=None): pass
ACT=re.compile(r"cmd_action=\[([^\]]*)\]")
def pull(files,tmin=None,tmax=None,meshfilter=False):
    A=[]
    for f in files:
        keep=True
        for line in open(f,errors="replace"):
            ts=line[:23]
            if tmin and not(tmin<=ts<=tmax): continue
            if "EPISODE_START" in line and meshfilter:
                keep = ("mesh_fp=topcowmr004" in line or "mesh_fp=topcowmr008" in line
                        or "mesh_fp=topcowmr017" in line or "mesh_fp=topcowmr023" in line)
            if " STEP |" in line and keep:
                m=ACT.search(line)
                if m:
                    try: A.append([float(x) for x in m.group(1).split(",")])
                    except: pass
    return A
Av=pull(sorted(glob.glob(os.path.join(V1BP,"logs","20260828_045651","worker_*.log"))))
Ah=pull(sorted(glob.glob(os.path.join(H0,"diagnostics","logs_subprocesses","worker_*.log"))),
        "2026-08-28 03:51:29","2026-08-28 04:15:20",meshfilter=True)
print("\nsteps: V1BP=%d H0=%d"%(len(Av),len(Ah)))
names=["gw_trans","gw_rot","cath_trans","cath_rot"]
lim=[30.0,1.5,30.0,1.5]
for i,nm in enumerate(names):
    cv=[a[i] for a in Av]; ch=[a[i] for a in Ah]
    print("  %-11s H0 range[%.3f,%.3f] sd=%.4f | V1BP range[%.3f,%.3f] sd=%.4f | V1BP frac |x|>max|H0|=%.3f | V1BP frac at clip(|x|>=%.2f)=%.3f"%(
        nm,min(ch),max(ch),st.pstdev(ch),min(cv),max(cv),st.pstdev(cv),
        sum(1 for x in cv if abs(x)>max(abs(min(ch)),abs(max(ch))))/len(cv),
        lim[i]*0.999, sum(1 for x in cv if abs(x)>=lim[i]*0.999)/len(cv)))
# residual lower bound on cath_rot (heuristic inert there)
cr=[a[3] for a in Av]
print("\nresidual[cath_rot] (heuristic contribution bounded by |a_h|<=0.013 from H0):")
print("  mean=%.4f sd=%.4f  |res|>0.1 in %.3f of steps  |res|>0.5 in %.3f  at clip 1.5 in %.4f"%(
    st.mean(cr),st.pstdev(cr),sum(1 for x in cr if abs(x)>0.1)/len(cr),
    sum(1 for x in cr if abs(x)>0.5)/len(cr),sum(1 for x in cr if abs(x)>=1.499)/len(cr)))
# per-episode: any episode with near-zero residual (i.e. policy dead)?
per=[]
for r in vb: pass
