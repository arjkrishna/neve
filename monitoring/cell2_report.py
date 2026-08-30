import json, sys, statistics as stx
from collections import Counter, defaultdict

S = "C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
R = "D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/"
V = R + "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/"

RUNS = {
 "A_holdout4": (S+"A_holdout4.jsonl", V+"eval_anatomies_checkpoint2002292/episodes_official_20260828_045651.jsonl"),
 "A_all22":    (S+"A_all22.jsonl",    V+"eval_anatomies_checkpoint2002292/episodes_official_20260828_053306.jsonl"),
 "H_all22":    (S+"H_all22.jsonl",    V+"eval_anatomies_checkpoint0/episodes_official_20260828_062606.jsonl"),
}

def load(name):
    lp, jp = RUNS[name]
    succ = {}
    for ln in open(jp):
        d = json.loads(ln); succ[d["seed"]] = bool(d["success"])
    eps = [json.loads(l) for l in open(lp)]
    for e in eps:
        e["succ"] = succ.get(e["seed"])
    return eps, succ

def pct(a, b): return "%.1f%%" % (100.0*a/b) if b else "n/a"
def q(v, p):
    if not v: return float('nan')
    v = sorted(v); import math
    i = min(len(v)-1, int(math.ceil(p*len(v)))-1); return v[max(0,i)]

def block(name, eps, key="cmd12", tag=""):
    n = len(eps); steps = [e["steps"] for e in eps]
    ev = [e["ev"][key] for e in eps]
    tot = sum(len(x) for x in ev)
    nst = sum(1 for x in ev if x)
    kinds = Counter(k["k"] for x in ev for k in x)
    res = tot - kinds["unrec"]
    rs = [k["r"] for x in ev for k in x if k["k"] != "unrec"]
    # success conditionals
    ns = [e for e, x in zip(eps, ev) if not x]
    srec = [e for e, x in zip(eps, ev) if x and not any(k["k"]=="unrec" for k in x)]
    sunr = [e for e, x in zip(eps, ev) if x and any(k["k"]=="unrec" for k in x)]
    def ps(g):
        g = [e for e in g if e["succ"] is not None]
        return "%s (%d/%d)" % (pct(sum(e["succ"] for e in g), len(g)), sum(e["succ"] for e in g), len(g))
    rm = [e["runmax"] for e in eps]
    out = []
    out.append("### %s%s  n_ep=%d  n_steps=%d  median_len=%.0f  mean_len=%.1f" % (name, tag, n, sum(steps), stx.median(steps), stx.mean(steps)))
    out.append("  stalls=%d  /ep=%.3f  /1000steps=%.2f  %%eps>=1stall=%s" % (tot, tot/n, 1000.0*tot/max(1,sum(steps)), pct(nst, n)))
    out.append("  mix grind/soft/hard/unrec = %d/%d/%d/%d  = %s / %s / %s / %s" % (
        kinds["grind"],kinds["soft"],kinds["hard"],kinds["unrec"],
        pct(kinds["grind"],tot),pct(kinds["soft"],tot),pct(kinds["hard"],tot),pct(kinds["unrec"],tot)))
    out.append("  resolved fraction = %s" % pct(res, tot))
    out.append("  P(succ|no stall)=%s   P(succ|stalled+all recovered)=%s   P(succ|stalled w/ unrec)=%s" % (ps(ns), ps(srec), ps(sunr)))
    if rs:
        out.append("  retract of resolved (mm): med=%.2f p90=%.2f max=%.2f" % (stx.median(rs), q(rs,.9), max(rs)))
    else:
        out.append("  retract of resolved: none")
    sw = []
    for ss in (4,6,8,12):
        t = sum(len(e["ev"]["cmd%d"%ss]) for e in eps)
        sw.append("ss=%d:%.3f/ep (%d tot, %.2f/1k, %s eps)" % (ss, t/n, t, 1000.0*t/max(1,sum(steps)), pct(sum(1 for e in eps if e["ev"]["cmd%d"%ss]), n)))
    out.append("  stuck_steps sweep: " + "  ".join(sw))
    te = sum(len(e["ev"]["exec12"]) for e in eps)
    out.append("  exec-push variant (delta_ins>0.264, ss=12): %d stalls, %.3f/ep" % (te, te/n))
    out.append("  longest low-adv-while-pushing run: med=%.0f p90=%.0f max=%d  %%eps>=4=%s  %%eps>=12=%s" % (
        stx.median(rm), q(rm,.9), max(rm), pct(sum(1 for v in rm if v>=4), n), pct(sum(1 for v in rm if v>=12), n)))
    ons = [k["on"] for x in ev for k in x]
    if ons:
        b = Counter(int(o//20)*20 for o in ons)
        out.append("  onset proj_s med=%.1f p90=%.1f ; bands(20mm): %s" % (stx.median(ons), q(ons,.9),
            " ".join("%d-%d:%d" % (a,a+20,b[a]) for a in sorted(b))))
        srcca = []
        for e, x in zip(eps, ev):
            for k in x: srcca.append(k["on"] - (e["pl"] - 33.31))
        b2 = Counter(int(v//10)*10 for v in srcca)
        out.append("  onset s_RCCA (=proj_s-(path_len-33.31)) med=%.1f p90=%.1f ; bands(10mm): %s" % (
            stx.median(srcca), q(srcca,.9), " ".join("%d:%d" % (a,b2[a]) for a in sorted(b2))))
    out.append("  path_len: med=%.1f  shared(<=166.91)=%d  grafted=%d" % (
        stx.median([e["pl"] for e in eps]), sum(1 for e in eps if e["pl"]<=166.91), sum(1 for e in eps if e["pl"]>166.91)))
    out.append("  success (jsonl): %s" % pct(sum(1 for e in eps if e["succ"]), n))
    return "\n".join(out)

if __name__ == "__main__":
    for nm in ("A_holdout4","A_all22","H_all22"):
        eps, _ = load(nm)
        print(block(nm, eps)); print()
        sh = [e for e in eps if e["pl"] <= 166.91]; gr = [e for e in eps if e["pl"] > 166.91]
        if sh: print(block(nm, sh, tag="  [SHARED host course]")); print()
        if gr: print(block(nm, gr, tag="  [GRAFTED]")); print()
