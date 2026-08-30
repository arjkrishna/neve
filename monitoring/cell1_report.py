"""CELL 1 report: family A (v1bp, procedural-trained) on the HOST anatomy.

Stall records come from monitoring/cell2_topbrain_stalls.py, which reuses the
extract_stuck.py Det state machine and canonical thresholds verbatim.
Bands are the project-canonical path_len sections (report_single.py sect()).
HOST offset OFF=33.80 mm derived independently in cell1_host_off_geom.py.
"""
import json, math, statistics as stx
from collections import Counter

S = "C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
R = "D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/"
V = R + "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/"
OFF = 33.80

RUNS = {
 "A2002292_HOST": (S + "C1_A2002292_host.jsonl", V + "eval_anatomies_checkpoint2002292/episodes_official_20260729_085006.jsonl"),
 "A514264_HOST":  (S + "C1_A514264_host.jsonl",  V + "eval_anatomies_checkpoint514264/episodes_official_20260729_070938.jsonl"),
 "H0_CONTROL":    (S + "C1_H0_ctrl.jsonl",       V + "eval_anatomies_checkpoint0/episodes_official_20260728_045004.jsonl"),
}


def sect(pl):
    return "CCA" if pl < 146 else ("ICA-mid" if pl < 210 else "siphon")


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, int(math.ceil(p * len(v))) - 1)
    return v[max(0, i)]


def load(name):
    lp, jp = RUNS[name]
    succ = {}
    for ln in open(jp):
        d = json.loads(ln)
        succ[d["seed"]] = bool(d["success"])
    eps = [json.loads(l) for l in open(lp)]
    for e in eps:
        e["succ"] = succ.get(e["seed"])
        e["sect"] = sect(e["pl"])
    return eps


def block(name, eps, key="cmd12", tag=""):
    n = len(eps)
    steps = [e["steps"] for e in eps]
    tsteps = sum(steps)
    ev = [e["ev"][key] for e in eps]
    tot = sum(len(x) for x in ev)
    nst = sum(1 for x in ev if x)
    kinds = Counter(k["k"] for x in ev for k in x)
    res = tot - kinds["unrec"]
    rs = [k["r"] for x in ev for k in x if k["k"] != "unrec"]
    ns = [e for e, x in zip(eps, ev) if not x]
    srec = [e for e, x in zip(eps, ev) if x and not any(k["k"] == "unrec" for k in x)]
    sunr = [e for e, x in zip(eps, ev) if x and any(k["k"] == "unrec" for k in x)]

    def ps(g):
        g = [e for e in g if e["succ"] is not None]
        return "%s (%d/%d)" % (pct(sum(e["succ"] for e in g), len(g)), sum(e["succ"] for e in g), len(g))

    rm = [e["runmax"] for e in eps]
    o = []
    o.append("### %s%s  n_ep=%d  n_steps=%d  median_len=%.0f  mean_len=%.1f  success=%s" % (
        name, tag, n, tsteps, stx.median(steps), stx.mean(steps),
        pct(sum(1 for e in eps if e["succ"]), sum(1 for e in eps if e["succ"] is not None))))
    o.append("  stalls=%d  per_ep=%.3f  per_1000steps=%.2f  pct_eps_with_stall=%s" % (
        tot, tot / n, 1000.0 * tot / max(1, tsteps), pct(nst, n)))
    o.append("  mix grind/soft/hard/unrec = %d/%d/%d/%d = %s / %s / %s / %s" % (
        kinds["grind"], kinds["soft"], kinds["hard"], kinds["unrec"],
        pct(kinds["grind"], tot), pct(kinds["soft"], tot), pct(kinds["hard"], tot), pct(kinds["unrec"], tot)))
    o.append("  resolved fraction = %s" % pct(res, tot))
    o.append("  P(succ|no stall)=%s  P(succ|stalled+all recovered)=%s  P(succ|stalled w/ unrec)=%s" % (
        ps(ns), ps(srec), ps(sunr)))
    if rs:
        o.append("  retract of resolved (mm): med=%.2f p90=%.2f max=%.2f" % (stx.median(rs), q(rs, .9), max(rs)))
    else:
        o.append("  retract of resolved: none")
    sw = []
    for ss in (4, 6, 8, 12):
        t = sum(len(e["ev"]["cmd%d" % ss]) for e in eps)
        sw.append("ss=%d:%.3f/ep(%d tot,%.2f/1k,%s eps)" % (
            ss, t / n, t, 1000.0 * t / max(1, tsteps), pct(sum(1 for e in eps if e["ev"]["cmd%d" % ss]), n)))
    o.append("  stuck_steps sweep: " + "  ".join(sw))
    te = sum(len(e["ev"]["exec12"]) for e in eps)
    tk = Counter(k["k"] for e in eps for k in e["ev"]["exec12"])
    o.append("  EXEC-advance variant (delta_ins>0.264mm, ss=12): %d stalls, %.3f/ep, %.2f/1k, mix g/s/h/u=%d/%d/%d/%d" % (
        te, te / n, 1000.0 * te / max(1, tsteps), tk["grind"], tk["soft"], tk["hard"], tk["unrec"]))
    o.append("  longest low-adv-while-pushing run: med=%.0f p90=%.0f max=%d  pct_eps>=4=%s  pct_eps>=12=%s" % (
        stx.median(rm), q(rm, .9), max(rm), pct(sum(1 for v in rm if v >= 4), n), pct(sum(1 for v in rm if v >= 12), n)))
    ons = [k["on"] for x in ev for k in x]
    if ons:
        b = Counter(int(v // 20) * 20 for v in ons)
        o.append("  onset proj_s: med=%.1f p90=%.1f ; bands20 %s" % (
            stx.median(ons), q(ons, .9), " ".join("%d-%d:%d" % (a, a + 20, b[a]) for a in sorted(b))))
        sr = [x - OFF for x in ons]
        b2 = Counter(int(v // 20) * 20 for v in sr)
        md = max(b2.items(), key=lambda kv: kv[1])
        o.append("  onset s_RCCA (proj_s-33.80): med=%.1f p90=%.1f ; bands20 %s ; MODAL %d-%d mm (%d, %s)" % (
            stx.median(sr), q(sr, .9), " ".join("%d:%d" % (a, b2[a]) for a in sorted(b2)),
            md[0], md[0] + 20, md[1], pct(md[1], len(sr))))
        rel = []
        for e, x in zip(eps, ev):
            for k in x:
                rel.append(k["on"] - e["pl"])
        o.append("  onset relative to target (proj_s-path_len): med=%.1f p10=%.1f p90=%.1f" % (
            stx.median(rel), q(rel, .1), q(rel, .9)))
    o.append("  path_len med=%.1f ; sections CCA/ICA-mid/siphon = %d/%d/%d" % (
        stx.median([e["pl"] for e in eps]),
        sum(1 for e in eps if e["sect"] == "CCA"),
        sum(1 for e in eps if e["sect"] == "ICA-mid"),
        sum(1 for e in eps if e["sect"] == "siphon")))
    return "\n".join(o)


def bandtab(name, eps, key="cmd12"):
    o = ["--- %s : per-section decomposition (canon cmd12) ---" % name]
    o.append("  sect       n  succ         stallEP%  /ep    /1k    unrecEP%  P(s|clean)   P(s|rec)     P(s|unrec)   medlen")
    for s in ("CCA", "ICA-mid", "siphon"):
        g = [e for e in eps if e["sect"] == s]
        if not g:
            continue
        n = len(g)
        st = sum(e["steps"] for e in g)
        ev = [e["ev"][key] for e in g]
        tot = sum(len(x) for x in ev)
        clean = [e for e, x in zip(g, ev) if not x]
        rec = [e for e, x in zip(g, ev) if x and not any(k["k"] == "unrec" for k in x)]
        unr = [e for e, x in zip(g, ev) if x and any(k["k"] == "unrec" for k in x)]

        def ps(gg):
            gg = [e for e in gg if e["succ"] is not None]
            return "%s(%d/%d)" % (pct(sum(e["succ"] for e in gg), len(gg)), sum(e["succ"] for e in gg), len(gg))

        o.append("  %-8s %3d  %-11s %7s %6.2f %6.2f %8s   %-12s %-12s %-12s %5.0f" % (
            s, n, "%s(%d/%d)" % (pct(sum(1 for e in g if e["succ"]), n), sum(1 for e in g if e["succ"]), n),
            pct(sum(1 for x in ev if x), n), tot / n, 1000.0 * tot / max(1, st), pct(len(unr), n),
            ps(clean), ps(rec), ps(unr), stx.median([e["steps"] for e in g])))
    return "\n".join(o)


def failtax(name, eps, key="cmd12"):
    o = ["--- %s : FAILURE taxonomy (canon cmd12) ---" % name]
    o.append("  sect      nFail  fail_noStall        fail_stall_allRecovered  fail_stall_unrecovered")
    for s in ("CCA", "ICA-mid", "siphon", "ALL"):
        g = [e for e in eps if (e["sect"] == s or s == "ALL") and e["succ"] is False]
        if not g:
            continue
        a = sum(1 for e in g if not e["ev"][key])
        c = sum(1 for e in g if e["ev"][key] and any(k["k"] == "unrec" for k in e["ev"][key]))
        b = len(g) - a - c
        o.append("  %-8s %5d  %-19s %-24s %s" % (
            s, len(g), "%2d (%s)" % (a, pct(a, len(g))), "%2d (%s)" % (b, pct(b, len(g))),
            "%2d (%s)" % (c, pct(c, len(g)))))
    return "\n".join(o)


def reach(name, eps):
    """How far did failures get, and did they arrest short of target?"""
    o = ["--- %s : reach of failures (maxp vs path_len) ---" % name]
    for s in ("CCA", "ICA-mid", "siphon"):
        g = [e for e in eps if e["sect"] == s and e["succ"] is False]
        if not g:
            continue
        d = [e["pl"] - e["maxp"] for e in g]
        far = sum(1 for v in d if v <= 5.0)
        o.append("  %-8s nFail=%2d  shortfall(path_len-maxp) med=%.1f p10=%.1f p90=%.1f mm ; reached within 5mm of target: %d (%s)" % (
            s, len(g), stx.median(d), q(d, .1), q(d, .9), far, pct(far, len(g))))
        sl = [e["maxslack"] for e in g]
        o.append("           max slack (gw_ins - proj_s) med=%.1f p90=%.1f mm ; steps med=%.0f" % (
            stx.median(sl), q(sl, .9), stx.median([e["steps"] for e in g])))
    for s in ("siphon",):
        g = [e for e in eps if e["sect"] == s and e["succ"] is True]
        if g:
            sl = [e["maxslack"] for e in g]
            o.append("  siphon SUCCESSES n=%d  max slack med=%.1f p90=%.1f ; steps med=%.0f" % (
                len(g), stx.median(sl), q(sl, .9), stx.median([e["steps"] for e in g])))
    return "\n".join(o)


if __name__ == "__main__":
    for nm in RUNS:
        eps = load(nm)
        print(block(nm, eps))
        print()
        print(bandtab(nm, eps))
        print()
        print(failtax(nm, eps))
        print()
        print(reach(nm, eps))
        print()
        for s in ("CCA", "ICA-mid", "siphon"):
            g = [e for e in eps if e["sect"] == s]
            if g:
                print(block(nm, g, tag="  [%s]" % s))
                print()
        print("=" * 110)
        print()
