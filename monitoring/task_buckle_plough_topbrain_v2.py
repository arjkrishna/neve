"""Pass 2: decompose buckle_phi into its slack and contact channels, and
check whether the eval-2/3 buckling mass is concentrated in a few episodes.

phi = -(0.5*slack_ex/40 + 0.5*contact/2) = -(slack_ex/80 + contact/4)
  slack_ex = clip(inserted_gw - proj_s - 5, 0, 40)   [both logged]
  => contact_mm = -4*phi - slack_ex/20   (exact while contact < 2mm cap;
     where phi <= -(slack_ex/80 + 0.5) the contact channel is SATURATED
     and the recovered value is a lower bound of 2.0)
"""
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
EXP_WIN = [("EXPLORE_preEVAL2", hms(12,0), hms(12,33)), ("EXPLORE_preEVAL3", hms(16,0), hms(16,31))]

def group_of(kind, t):
    if t is None:
        return None
    if kind == 'eval':
        for nm, a, b in EVAL_WIN:
            if a <= t < b:
                return nm
        return None
    for nm, a, b in EXP_WIN:
        if a <= t < b:
            return nm
    return None

class E:
    __slots__ = ('n','phi','slack','contact','fold','cs','sat','reason','succ','ret','maxfold','maxslack','maxcs','mincontact')
    def __init__(self):
        self.n=0; self.phi=0.0; self.slack=[]; self.contact=[]; self.fold=0; self.cs=[]
        self.sat=0; self.reason=None; self.succ=None; self.ret=None
        self.maxfold=0; self.maxslack=-1e9; self.maxcs=-1e9

EPS = defaultdict(lambda: defaultdict(E))
STEPS = defaultdict(lambda: {'slack':[], 'contact':[], 'contact_sat':0, 'n':0})

for fp in sorted(glob.glob(os.path.join(LOGDIR, "worker_*.log"))):
    fname = os.path.basename(fp)
    with open(fp, 'r', errors='replace') as fh:
        for line in fh:
            if ' - INFO - ' not in line:
                continue
            if 'step_logger_eval' in line:
                kind = 'eval'
            elif 'step_logger_train' in line:
                kind = 'train'
            else:
                continue
            body = line.split(' - INFO - ', 1)[1]
            isstep = body.startswith('STEP |')
            isout = body.startswith('EPISODE_OUTCOME')
            if not (isstep or isout):
                continue
            g = group_of(kind, tsec(line))
            if g is None:
                continue
            d = {}
            for p in body.rstrip("\n").split(' | ')[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    d[k] = v
            if 'ep' not in d:
                continue
            key = (fname, kind, d.get('pid','x'), d['ep'])
            e = EPS[g][key]
            if isout:
                e.reason = d.get('reason','?')
                e.succ = (d.get('grader_success') == '1')
                try:
                    e.ret = float(d.get('return'))
                except Exception:
                    pass
                continue
            try:
                phi = float(d['buckle_phi'])
            except Exception:
                continue
            try:
                ins_gw = float(d['inserted'].strip('[]').split(',')[0])
                proj_s = float(d['proj_s'])
                slack = ins_gw - proj_s
            except Exception:
                slack = float('nan')
            if math.isnan(slack):
                continue
            slack_ex = min(max(slack - 5.0, 0.0), 40.0)
            contact = -4.0*phi - slack_ex/20.0
            saturated = contact >= 1.995
            if saturated:
                contact = 2.0
            e.n += 1
            e.phi += phi
            e.slack.append(slack)
            e.contact.append(contact)
            if saturated:
                e.sat += 1
            if slack > e.maxslack:
                e.maxslack = slack
            try:
                fn = int(d.get('fold','0/20').split('/')[0])
            except Exception:
                fn = 0
            if fn > e.maxfold:
                e.maxfold = fn
            try:
                c = float(d['cath_slack'])
                e.cs.append(c)
                if c > e.maxcs:
                    e.maxcs = c
            except Exception:
                pass
            S = STEPS[g]
            S['n'] += 1
            S['slack'].append(slack)
            S['contact'].append(contact)
            if saturated:
                S['contact_sat'] += 1

def pct(a, q):
    if not a:
        return float('nan')
    s = sorted(a)
    return s[min(len(s)-1, int(q*(len(s)-1)+0.5))]

def mean(a):
    return sum(a)/len(a) if a else float('nan')

for g in ["EVAL1","EVAL2","EVAL3","EXPLORE_preEVAL2","EXPLORE_preEVAL3"]:
    if g not in STEPS:
        continue
    S = STEPS[g]
    print("="*80)
    print("GROUP %s  steps=%d episodes=%d" % (g, S['n'], len(EPS[g])))
    sl = S['slack']; co = S['contact']
    print("  GW SLACK mm (inserted_gw - proj_s): p50=%+.2f p90=%+.2f p99=%+.2f max=%+.2f mean=%+.2f"
          % (pct(sl,.5), pct(sl,.9), pct(sl,.99), max(sl), mean(sl)))
    for thr in (5, 10, 20, 40):
        c = sum(1 for x in sl if x > thr)
        print("    frac slack>%2dmm : %.4f (%d)" % (thr, c/S['n'], c))
    print("  CONTACT mm (recovered, cap 2.0): p50=%.3f p90=%.3f p99=%.3f mean=%.3f | frac SATURATED(>=2mm)=%.4f (%d)"
          % (pct(co,.5), pct(co,.9), pct(co,.99), mean(co), S['contact_sat']/S['n'], S['contact_sat']))
    for thr in (0.1, 0.5, 1.0, 1.5):
        c = sum(1 for x in co if x >= thr)
        print("    frac contact>=%.1fmm : %.4f (%d)" % (thr, c/S['n'], c))
    # per-episode concentration
    eps = list(EPS[g].items())
    eps_sat = sorted(eps, key=lambda kv: -kv[1].sat)
    tot_sat = S['contact_sat']
    top = eps_sat[:5]
    print("  contact-saturation concentration: top-5 episodes hold %d/%d (%.3f) of saturated steps"
          % (sum(e.sat for _, e in top), max(1,tot_sat), sum(e.sat for _, e in top)/max(1,tot_sat)))
    for k, e in top:
        print("      ep=%s steps=%d sat=%d meanphi=%+.4f maxfold=%d maxslack=%+.1f maxcath=%+.1f reason=%s"
              % (k[3], e.n, e.sat, e.phi/max(1,e.n), e.maxfold, e.maxslack, e.maxcs, e.reason))
    neps = len(eps)
    print("  per-episode: eps with any saturated-contact step: %d/%d ; eps with >10%% sat steps: %d ; eps with maxslack>10mm: %d ; >20mm: %d"
          % (sum(1 for _, e in eps if e.sat > 0), neps,
             sum(1 for _, e in eps if e.n and e.sat/e.n > 0.10),
             sum(1 for _, e in eps if e.maxslack > 10),
             sum(1 for _, e in eps if e.maxslack > 20)))
    mp = [e.phi/max(1,e.n) for _, e in eps]
    print("  per-episode mean phi: p50=%+.4f p90=%+.4f worst=%+.4f" % (pct(mp,.5), pct(mp,.10), min(mp)))
    ms = [e.maxslack for _, e in eps]
    print("  per-episode max GW slack: p50=%+.1f p90=%+.1f max=%+.1f" % (pct(ms,.5), pct(ms,.9), max(ms)))
    # success-only restriction
    succ = [(k,e) for k,e in eps if e.succ is True]
    fail = [(k,e) for k,e in eps if e.succ is False]
    unk  = [(k,e) for k,e in eps if e.succ is None]
    print("  outcome join: success=%d fail=%d NO-OUTCOME-LINE=%d" % (len(succ), len(fail), len(unk)))
    for label, sub in (("SUCCESS-only", succ), ("FAIL-only", fail), ("no-outcome", unk)):
        if not sub:
            continue
        n = sum(e.n for _, e in sub)
        sat = sum(e.sat for _, e in sub)
        mphi = sum(e.phi for _, e in sub)/max(1,n)
        mfold = [e.maxfold for _, e in sub]
        msl = [e.maxslack for _, e in sub]
        print("    %-12s eps=%3d steps=%6d meanphi=%+.4f fracsat=%.4f  per-ep maxfold p50=%d p90=%d max=%d  maxslack p90=%+.1f max=%+.1f"
              % (label, len(sub), n, mphi, sat/max(1,n), pct(mfold,.5), pct(mfold,.9), max(mfold), pct(msl,.9), max(msl)))
