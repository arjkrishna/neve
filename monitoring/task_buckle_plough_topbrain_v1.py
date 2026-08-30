import os, sys, glob, math
from collections import defaultdict

LOGDIR = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/logs_subprocesses"
DT = 1.0 / 7.5

def tsec(line):
    try:
        h = int(line[11:13]); m = int(line[14:16]); s = int(line[17:19]); ms = int(line[20:23])
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception:
        return None

def hms(h, m, s=0):
    return h * 3600 + m * 60 + s

EVAL_WIN = [("EVAL1", hms(8, 0), hms(9, 0)),
            ("EVAL2", hms(12, 0), hms(13, 0)),
            ("EVAL3", hms(16, 0), hms(17, 0))]
EXP_WIN = [("EXPLORE_heatup_preEVAL1", hms(8, 0), hms(8, 11, 52)),
           ("EXPLORE_preEVAL2", hms(12, 0), hms(12, 33)),
           ("EXPLORE_preEVAL3", hms(16, 0), hms(16, 31))]

def group_of(kind, t):
    if t is None:
        return None
    if kind == 'eval':
        for name, a, b in EVAL_WIN:
            if a <= t < b:
                return name
        return "EVAL_OTHER"
    for name, a, b in EXP_WIN:
        if a <= t < b:
            return name
    return None

class Acc:
    def __init__(self):
        self.phi = []; self.fold = []; self.cslack = []
        self.cmd_gw = []; self.ach_gw = []; self.cmd_ct = []; self.ach_ct = []
        self.nsteps = 0
        self.ep_first_phi = {}; self.ep_last_phi = {}
        self.ep_gross_buckle = defaultdict(float)
        self.ep_pos_buckle = defaultdict(float)
        self.ep_first_cs = {}; self.ep_last_cs = {}
        self.ep_gross_cath = defaultdict(float)
        self.ep_maxfold = defaultdict(int)
        self.ep_maxcs = defaultdict(lambda: -1e9)
        self.ep_minphi = defaultdict(float)
        self.ep_ret = {}
        self.ep_steps = defaultdict(int)
        self.outcomes = defaultdict(int)
        self.n_start = 0; self.n_out = 0; self.succ = 0

ACC = defaultdict(Acc)
prev_phi = {}
prev_cs_phi = {}

def cath_phi(slack):
    ex = min(max(slack - 15.0, 0.0), 150.0)
    return -(ex / 150.0)

files = sorted(glob.glob(os.path.join(LOGDIR, "worker_*.log")))
for fp in files:
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
            if body.startswith('STEP |'):
                t = tsec(line); g = group_of(kind, t)
                if g is None:
                    continue
                parts = body.rstrip("\n").split(' | ')
                d = {}
                for p in parts[1:]:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        d[k] = v
                if 'ep' not in d:
                    continue
                key = (fname, kind, d.get('pid', 'x'), d['ep'])
                A = ACC[g]
                A.nsteps += 1
                A.ep_steps[key] += 1
                try:
                    phi = float(d['buckle_phi'])
                except Exception:
                    phi = 0.0
                A.phi.append(phi)
                if key not in A.ep_first_phi:
                    A.ep_first_phi[key] = phi
                A.ep_last_phi[key] = phi
                if phi < A.ep_minphi[key]:
                    A.ep_minphi[key] = phi
                pk = (g, key)
                if pk in prev_phi:
                    dphi = phi - prev_phi[pk]
                    if dphi < 0:
                        A.ep_gross_buckle[key] += 0.5 * dphi
                    else:
                        A.ep_pos_buckle[key] += 0.5 * dphi
                prev_phi[pk] = phi
                try:
                    cs = float(d['cath_slack'])
                except Exception:
                    cs = 0.0
                A.cslack.append(cs)
                if key not in A.ep_first_cs:
                    A.ep_first_cs[key] = cs
                A.ep_last_cs[key] = cs
                if cs > A.ep_maxcs[key]:
                    A.ep_maxcs[key] = cs
                cp = cath_phi(cs)
                if pk in prev_cs_phi:
                    dc = cp - prev_cs_phi[pk]
                    if dc < 0:
                        A.ep_gross_cath[key] += 0.5 * dc
                prev_cs_phi[pk] = cp
                try:
                    fn = int(d.get('fold', '0/20').split('/')[0])
                except Exception:
                    fn = 0
                A.fold.append(fn)
                if fn > A.ep_maxfold[key]:
                    A.ep_maxfold[key] = fn
                try:
                    ca = d.get('cmd_action', '').strip('[]').split(',')
                    cgw = float(ca[0]); cct = float(ca[2])
                except Exception:
                    cgw = cct = float('nan')
                try:
                    di = d.get('delta_ins', '').strip('[]').split(',')
                    agw = float(di[0]); act = float(di[1])
                except Exception:
                    agw = act = float('nan')
                if not (math.isnan(cgw) or math.isnan(agw)):
                    A.cmd_gw.append(cgw); A.ach_gw.append(agw)
                    A.cmd_ct.append(cct); A.ach_ct.append(act)
                try:
                    A.ep_ret[key] = float(d['cum_reward'])
                except Exception:
                    pass
            elif body.startswith('EPISODE_START'):
                t = tsec(line); g = group_of(kind, t)
                if g is None:
                    continue
                ACC[g].n_start += 1
            elif body.startswith('EPISODE_OUTCOME'):
                t = tsec(line); g = group_of(kind, t)
                if g is None:
                    continue
                parts = body.rstrip("\n").split(' | ')
                d = {}
                for p in parts[1:]:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        d[k] = v
                A = ACC[g]
                A.n_out += 1
                A.outcomes[d.get('reason', 'unknown')] += 1
                if d.get('grader_success') == '1':
                    A.succ += 1

def pct(a, q):
    if not a:
        return float('nan')
    s = sorted(a)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]

def mean(a):
    return sum(a) / len(a) if a else float('nan')

order = ["EVAL1", "EVAL2", "EVAL3", "EXPLORE_heatup_preEVAL1",
         "EXPLORE_preEVAL2", "EXPLORE_preEVAL3", "EVAL_OTHER"]
for g in order:
    if g not in ACC:
        continue
    A = ACC[g]
    n = A.nsteps
    if n == 0:
        continue
    print("=" * 78)
    print("GROUP %s: ep_starts=%d outcomes=%d steps=%d eps_with_steps=%d"
          % (g, A.n_start, A.n_out, n, len(A.ep_steps)))
    print("  outcomes: %s | grader_success=%d rate=%.3f"
          % (dict(A.outcomes), A.succ, A.succ / max(1, A.n_out)))
    print("  mean ep steps=%.1f  mean final cum_reward=%.3f"
          % (mean(list(A.ep_steps.values())), mean(list(A.ep_ret.values()))))
    ph = A.phi
    print("  buckle_phi: p50=%+.4f p90=%+.4f p99=%+.4f worst=%+.4f mean=%+.4f"
          % (pct(ph, .5), pct(ph, .10), pct(ph, .01), min(ph), mean(ph)))
    for thr in (-0.001, -0.01, -0.05, -0.10, -0.25, -0.50):
        c = sum(1 for x in ph if x <= thr)
        print("    frac phi<=%+.3f : %.4f (%d)" % (thr, c / n, c))
    fo = A.fold
    c0 = sum(1 for x in fo if x > 0)
    print("  fold n: frac n>0=%.4f (%d) mean=%.3f p90=%d p99=%d max=%d"
          % (c0 / n, c0, mean(fo), pct(fo, .9), pct(fo, .99), max(fo)))
    for thr in (1, 5, 10, 20):
        c = sum(1 for x in fo if x >= thr)
        print("    frac n>=%d : %.4f (%d)" % (thr, c / n, c))
    mf = list(A.ep_maxfold.values())
    print("  per-ep max fold: p50=%d p90=%d max=%d; eps>=10: %d/%d; eps>=20: %d/%d"
          % (pct(mf, .5), pct(mf, .9), max(mf), sum(1 for x in mf if x >= 10), len(mf),
             sum(1 for x in mf if x >= 20), len(mf)))
    cs = A.cslack
    print("  cath_slack mm: p50=%+.2f p90=%+.2f p99=%+.2f max=%+.2f mean=%+.2f"
          % (pct(cs, .5), pct(cs, .9), pct(cs, .99), max(cs), mean(cs)))
    for thr in (5, 15, 50, 100):
        c = sum(1 for x in cs if x > thr)
        print("    frac cath_slack>%dmm : %.4f (%d)" % (thr, c / n, c))
    mc = list(A.ep_maxcs.values())
    print("  per-ep max cath_slack: p50=%+.1f p90=%+.1f max=%+.1f; eps>50mm: %d/%d"
          % (pct(mc, .5), pct(mc, .9), max(mc), sum(1 for x in mc if x > 50), len(mc)))
    cg = A.cmd_gw; ag = A.ach_gw; cc = A.cmd_ct; ac = A.ach_ct
    cgmm = [x * DT for x in cg]
    ccmm = [x * DT for x in cc]
    print("  GW cmd mm/s: mean=%+.3f p90=%+.2f max=%+.2f | cmd mm/step mean=%+.4f | ach mm/step mean=%+.4f"
          % (mean(cg), pct(cg, .9), max(cg), mean(cgmm), mean(ag)))
    fwd = [(a, c) for a, c in zip(ag, cgmm) if c > 0.05]
    if fwd:
        mc_ = mean([c for a, c in fwd]); ma_ = mean([a for a, c in fwd])
        r = [a / c for a, c in fwd]
        print("    fwd-cmd steps n=%d (%.3f) mean_cmd=%.4f mean_ach=%.4f ratio=%.4f"
              % (len(fwd), len(fwd) / n, mc_, ma_, ma_ / mc_))
        print("    per-step ach/cmd: p10=%.3f p50=%.3f p90=%.3f" % (pct(r, .1), pct(r, .5), pct(r, .9)))
    print("  CATH cmd mm/s: mean=%+.3f p90=%+.2f max=%+.2f | ach mm/step mean=%+.4f"
          % (mean(cc), pct(cc, .9), max(cc), mean(ac)))
    fwdc = [(a, c) for a, c in zip(ac, ccmm) if c > 0.05]
    if fwdc:
        mc_ = mean([c for a, c in fwdc]); ma_ = mean([a for a, c in fwdc])
        r = [a / c for a, c in fwdc]
        print("    fwd-cmd steps n=%d (%.3f) mean_cmd=%.4f mean_ach=%.4f ratio=%.4f"
              % (len(fwdc), len(fwdc) / n, mc_, ma_, ma_ / mc_))
        print("    per-step ach/cmd: p10=%.3f p50=%.3f p90=%.3f" % (pct(r, .1), pct(r, .5), pct(r, .9)))
    sat = sum(1 for x in cg if x >= 27.0)
    satc = sum(1 for x in cc if x >= 27.0)
    print("  frac cmd_gw>=27mm/s: %.4f ; cmd_cath>=27mm/s: %.4f" % (sat / max(1, len(cg)), satc / max(1, len(cc))))
    aspd = [abs(x) / DT for x in ag]
    aspdc = [abs(x) / DT for x in ac]
    print("  achieved GW |speed| mm/s: mean=%.2f p50=%.2f p90=%.2f max=%.2f"
          % (mean(aspd), pct(aspd, .5), pct(aspd, .9), max(aspd)))
    print("  achieved CATH |speed| mm/s: mean=%.2f p50=%.2f p90=%.2f max=%.2f"
          % (mean(aspdc), pct(aspdc, .5), pct(aspdc, .9), max(aspdc)))
    tele = [0.5 * (A.ep_last_phi[k] - A.ep_first_phi[k]) for k in A.ep_last_phi]
    gross = [A.ep_gross_buckle[k] for k in A.ep_last_phi]
    posb = [A.ep_pos_buckle[k] for k in A.ep_last_phi]
    telec = [0.5 * (cath_phi(A.ep_last_cs[k]) - cath_phi(A.ep_first_cs[k])) for k in A.ep_last_cs]
    grossc = [A.ep_gross_cath[k] for k in A.ep_last_cs]
    print("  BUCKLE channel/ep: telescoped mean=%+.4f p10=%+.4f min=%+.4f | gross_neg mean=%+.4f p10=%+.4f min=%+.4f | gross_pos mean=%+.4f"
          % (mean(tele), pct(tele, .1), min(tele), mean(gross), pct(gross, .1), min(gross), mean(posb)))
    print("    eps with any neg buckle delta: %d/%d ; eps gross_neg<-0.10: %d"
          % (sum(1 for x in gross if x < -1e-9), len(gross), sum(1 for x in gross if x < -0.10)))
    print("  CATH-SLACK channel/ep: telescoped mean=%+.4f min=%+.4f | gross_neg mean=%+.4f min=%+.4f"
          % (mean(telec), min(telec), mean(grossc), min(grossc)))
