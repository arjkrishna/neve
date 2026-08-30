"""Pass 5: eval stats with the extreme-coil episodes (max GW slack > 100mm) excluded,
to separate 'typical gait' from 'rare catastrophic coil'."""
import os, glob
from collections import defaultdict

LOGDIR = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"

def tsec(l):
    try: return int(l[11:13])*3600+int(l[14:16])*60+int(l[17:19])+int(l[20:23])/1000.0
    except Exception: return None
def hms(h,m,s=0): return h*3600+m*60+s
WIN=[("EVAL1",hms(8,0),hms(9,0)),("EVAL2",hms(12,0),hms(13,0)),("EVAL3",hms(16,0),hms(17,0))]

rows=defaultdict(lambda: defaultdict(list))  # group -> epkey -> list of (phi, slack, fold, cslack)
for fp in sorted(glob.glob(os.path.join(LOGDIR,"worker_*.log"))):
    fn=os.path.basename(fp)
    with open(fp,'r',errors='replace') as fh:
        for line in fh:
            if 'step_logger_eval' not in line or ' - INFO - ' not in line: continue
            body=line.split(' - INFO - ',1)[1]
            if not body.startswith('STEP |'): continue
            t=tsec(line); g=None
            for nm,a,b in WIN:
                if t is not None and a<=t<b: g=nm
            if g is None: continue
            d={}
            for p in body.rstrip("\n").split(' | ')[1:]:
                if '=' in p:
                    k,v=p.split('=',1); d[k]=v
            try:
                phi=float(d['buckle_phi']); ins=float(d['inserted'].strip('[]').split(',')[0])
                ps=float(d['proj_s']); cs=float(d['cath_slack'])
                fo=int(d.get('fold','0/20').split('/')[0])
            except Exception:
                continue
            rows[g][(fn,d.get('pid'),d['ep'])].append((phi, ins-ps, fo, cs))

def pct(a,q):
    if not a: return float('nan')
    s=sorted(a); return s[min(len(s)-1,int(q*(len(s)-1)+0.5))]
def mean(a): return sum(a)/len(a) if a else float('nan')

for g in ["EVAL1","EVAL2","EVAL3"]:
    eps=rows[g]
    for label, keep in (("ALL", lambda st: True),
                        ("TRIMMED (drop eps with max slack>100mm)",
                         lambda st: max(x[1] for x in st) <= 100.0)):
        sub={k:v for k,v in eps.items() if keep(v)}
        allst=[x for v in sub.values() for x in v]
        n=len(allst)
        if not n: continue
        phi=[x[0] for x in allst]; sl=[x[1] for x in allst]
        fo=[x[2] for x in allst]; cs=[x[3] for x in allst]
        mf=[max(x[2] for x in v) for v in sub.values()]
        ms=[max(x[1] for x in v) for v in sub.values()]
        mc=[max(x[3] for x in v) for v in sub.values()]
        foldsteps_per_ep=[sum(1 for x in v if x[2]>0)/len(v) for v in sub.values()]
        print("%-6s %-40s eps=%3d steps=%6d" % (g, label, len(sub), n))
        print("        mean phi=%+.5f | frac steps phi<=-0.10 = %.4f | frac phi<=-0.50 = %.4f"
              % (mean(phi), sum(1 for x in phi if x<=-0.10)/n, sum(1 for x in phi if x<=-0.50)/n))
        print("        GW slack p50=%+.2f p90=%+.2f p99=%+.2f max=%+.1f | frac steps >20mm=%.4f >40mm=%.4f"
              % (pct(sl,.5),pct(sl,.9),pct(sl,.99),max(sl),
                 sum(1 for x in sl if x>20)/n, sum(1 for x in sl if x>40)/n))
        print("        fold: frac steps n>0=%.4f | per-ep frac-of-own-steps with n>0: p50=%.3f p90=%.3f | per-ep maxfold p50=%d p90=%d max=%d | eps>=10: %d/%d"
              % (sum(1 for x in fo if x>0)/n, pct(foldsteps_per_ep,.5), pct(foldsteps_per_ep,.9),
                 pct(mf,.5),pct(mf,.9),max(mf), sum(1 for x in mf if x>=10), len(mf)))
        print("        cath_slack p50=%+.2f p90=%+.2f max=%+.1f | per-ep max p50=%+.1f p90=%+.1f | eps>50mm: %d/%d"
              % (pct(cs,.5),pct(cs,.9),max(cs),pct(mc,.5),pct(mc,.9),
                 sum(1 for x in mc if x>50), len(mc)))
    print("-"*88)
