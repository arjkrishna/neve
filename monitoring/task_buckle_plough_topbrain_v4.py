"""Pass 4: lumen-tracking quality — true cross-track error vs local tolerance."""
import os, glob, math
from collections import defaultdict

LOGDIR = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"

def tsec(l):
    try:
        return int(l[11:13])*3600+int(l[14:16])*60+int(l[17:19])+int(l[20:23])/1000.0
    except Exception:
        return None

def hms(h,m,s=0): return h*3600+m*60+s
WIN = [("EVAL1",hms(8,0),hms(9,0)),("EVAL2",hms(12,0),hms(13,0)),("EVAL3",hms(16,0),hms(17,0))]

G = defaultdict(lambda: {'xt':[], 'xtr':[], 'onpath':0, 'offbr':0, 'n':0,
                         'xt_siphon':[], 'n_siphon':0, 'onpath_siphon':0,
                         'ep_maxxt':defaultdict(float)})

for fp in sorted(glob.glob(os.path.join(LOGDIR,"worker_*.log"))):
    fn = os.path.basename(fp)
    with open(fp,'r',errors='replace') as fh:
        for line in fh:
            if 'step_logger_eval' not in line or ' - INFO - ' not in line:
                continue
            body = line.split(' - INFO - ',1)[1]
            if not body.startswith('STEP |'):
                continue
            t = tsec(line); g = None
            for nm,a,b in WIN:
                if t is not None and a <= t < b: g = nm
            if g is None: continue
            d = {}
            for p in body.rstrip("\n").split(' | ')[1:]:
                if '=' in p:
                    k,v = p.split('=',1); d[k]=v
            try:
                xt = abs(float(d['xt_true'])); tol = float(d['tol']); ps = float(d['proj_s'])
            except Exception:
                continue
            S = G[g]
            S['n'] += 1
            S['xt'].append(xt)
            if tol > 0: S['xtr'].append(xt/tol)
            if d.get('on_path') == '1': S['onpath'] += 1
            if d.get('off_br') == '1': S['offbr'] += 1
            key = (fn, d.get('pid'), d.get('ep'))
            if xt > S['ep_maxxt'][key]: S['ep_maxxt'][key] = xt
            # s_RCCA = path_len - 33.31 ; graft seam at s_RCCA=133.6 -> proj_s ~166.9
            if ps >= 166.9:
                S['n_siphon'] += 1
                S['xt_siphon'].append(xt)
                if d.get('on_path') == '1': S['onpath_siphon'] += 1

def pct(a,q):
    if not a: return float('nan')
    s=sorted(a); return s[min(len(s)-1,int(q*(len(s)-1)+0.5))]
def mean(a): return sum(a)/len(a) if a else float('nan')

for g in ["EVAL1","EVAL2","EVAL3"]:
    S = G[g]
    n = S['n']
    if not n: continue
    print("="*80)
    print("%s steps=%d" % (g, n))
    print("  |xt_true| mm (true planned-path cross-track): p50=%.2f p90=%.2f p99=%.2f max=%.2f mean=%.2f"
          % (pct(S['xt'],.5), pct(S['xt'],.9), pct(S['xt'],.99), max(S['xt']), mean(S['xt'])))
    print("  |xt|/tol ratio: p50=%.3f p90=%.3f p99=%.3f ; frac steps |xt|>tol = %.4f"
          % (pct(S['xtr'],.5), pct(S['xtr'],.9), pct(S['xtr'],.99),
             sum(1 for x in S['xtr'] if x > 1.0)/max(1,len(S['xtr']))))
    print("  on_path frac=%.4f ; off_br frac=%.4f" % (S['onpath']/n, S['offbr']/n))
    mx = list(S['ep_maxxt'].values())
    print("  per-episode max |xt|: p50=%.2f p90=%.2f max=%.2f (eps=%d)"
          % (pct(mx,.5), pct(mx,.9), max(mx), len(mx)))
    if S['n_siphon']:
        print("  PAST-SEAM (proj_s>=166.9mm, real patient siphon): steps=%d (%.3f) |xt| p50=%.2f p90=%.2f max=%.2f ; on_path frac=%.4f"
              % (S['n_siphon'], S['n_siphon']/n, pct(S['xt_siphon'],.5), pct(S['xt_siphon'],.9),
                 max(S['xt_siphon']), S['onpath_siphon']/S['n_siphon']))
