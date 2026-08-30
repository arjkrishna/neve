"""Pass 3: per-episode detail for the three evals — where the buckling mass sits."""
import os, glob, math
from collections import defaultdict

LOGDIR = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"

def tsec(line):
    try:
        return int(line[11:13])*3600 + int(line[14:16])*60 + int(line[17:19]) + int(line[20:23])/1000.0
    except Exception:
        return None

def hms(h, m, s=0):
    return h*3600 + m*60 + s

EVAL_WIN = [("EVAL1", hms(8,0), hms(9,0)), ("EVAL2", hms(12,0), hms(13,0)), ("EVAL3", hms(16,0), hms(17,0))]

class E:
    def __init__(self):
        self.n=0; self.phisum=0.0; self.maxfold=0; self.maxslack=-1e9
        self.s_at_max=None; self.br_at_max=None; self.mesh=None
        self.steps_sl20=0; self.steps_sl40=0; self.steps_fold10=0
        self.reason=None; self.succ=None; self.ret=None; self.maxcath=-1e9
        self.last_projs=None; self.pathlen=None; self.maxins=0.0

EPS = defaultdict(lambda: defaultdict(E))

for fp in sorted(glob.glob(os.path.join(LOGDIR, "worker_*.log"))):
    fname = os.path.basename(fp)
    with open(fp, 'r', errors='replace') as fh:
        for line in fh:
            if 'step_logger_eval' not in line or ' - INFO - ' not in line:
                continue
            t = tsec(line)
            g = None
            for nm, a, b in EVAL_WIN:
                if t is not None and a <= t < b:
                    g = nm
            if g is None:
                continue
            body = line.split(' - INFO - ', 1)[1]
            isstep = body.startswith('STEP |')
            isout = body.startswith('EPISODE_OUTCOME')
            if not (isstep or isout):
                continue
            d = {}
            for p in body.rstrip("\n").split(' | ')[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    d[k] = v
            if 'ep' not in d:
                continue
            key = (fname, d.get('pid','x'), d['ep'])
            e = EPS[g][key]
            if isout:
                e.reason = d.get('reason','?')
                e.succ = (d.get('grader_success') == '1')
                try: e.ret = float(d.get('return'))
                except Exception: pass
                continue
            try:
                phi = float(d['buckle_phi'])
                ins = float(d['inserted'].strip('[]').split(',')[0])
                ps = float(d['proj_s'])
            except Exception:
                continue
            sl = ins - ps
            e.n += 1
            e.phisum += phi
            e.last_projs = ps
            e.pathlen = d.get('path_len')
            if ins > e.maxins: e.maxins = ins
            if sl > e.maxslack:
                e.maxslack = sl; e.s_at_max = ps; e.br_at_max = d.get('nearest_named')
            if sl > 20: e.steps_sl20 += 1
            if sl > 40: e.steps_sl40 += 1
            try: fn = int(d.get('fold','0/20').split('/')[0])
            except Exception: fn = 0
            if fn > e.maxfold: e.maxfold = fn
            if fn >= 10: e.steps_fold10 += 1
            try:
                c = float(d['cath_slack'])
                if c > e.maxcath: e.maxcath = c
            except Exception:
                pass
            e.mesh = d.get('nearest_named')

def pct(a,q):
    if not a: return float('nan')
    s=sorted(a); return s[min(len(s)-1,int(q*(len(s)-1)+0.5))]
def mean(a):
    return sum(a)/len(a) if a else float('nan')

for g in ["EVAL1","EVAL2","EVAL3"]:
    eps = [(k,e) for k,e in EPS[g].items() if e.n > 0]
    print("="*88)
    print("%s : %d episodes with steps, %d total steps" % (g, len(eps), sum(e.n for _,e in eps)))
    tot = sum(e.n for _,e in eps)
    tot_sl40 = sum(e.steps_sl40 for _,e in eps)
    srt = sorted(eps, key=lambda kv: -kv[1].maxslack)
    print("  TOP-12 episodes by max GW slack:")
    print("   %-4s %-6s %5s %5s %8s %8s %7s %7s %8s %8s %s" % ("ep","succ","stps","fold","maxslk","s@max","stp>20","stp>40","maxcath","return","reason"))
    for k,e in srt[:12]:
        print("   %-4s %-6s %5d %5d %8.1f %8.1f %7d %7d %8.1f %8s %s"
              % (k[2], str(e.succ), e.n, e.maxfold, e.maxslack, (e.s_at_max or -1),
                 e.steps_sl20, e.steps_sl40, e.maxcath,
                 ("%.2f"%e.ret) if e.ret is not None else "NA", e.reason))
    print("  slack>40mm step mass: total=%d (%.4f of steps); held by episodes with succ=False/None: %d"
          % (tot_sl40, tot_sl40/tot, sum(e.steps_sl40 for _,e in eps if e.succ is not True)))
    for label, sub in (("ALL", eps),
                       ("SUCCESS", [(k,e) for k,e in eps if e.succ is True]),
                       ("NOT-SUCCESS(incl no-outcome)", [(k,e) for k,e in eps if e.succ is not True])):
        if not sub: continue
        ms = [e.maxslack for _,e in sub]
        mf = [e.maxfold for _,e in sub]
        n = sum(e.n for _,e in sub)
        print("  %-30s eps=%3d steps=%6d | maxslack p50=%+.1f p90=%+.1f max=%+.1f ; eps maxslack>20mm=%d ; >100mm=%d"
              % (label, len(sub), n, pct(ms,.5), pct(ms,.9), max(ms),
                 sum(1 for x in ms if x>20), sum(1 for x in ms if x>100)))
        print("  %-30s | maxfold p50=%d p90=%d max=%d ; eps fold>=10=%d ; eps fold>=20=%d ; mean phi=%+.5f"
              % ("", pct(mf,.5), pct(mf,.9), max(mf),
                 sum(1 for x in mf if x>=10), sum(1 for x in mf if x>=20),
                 sum(e.phisum for _,e in sub)/max(1,n)))
