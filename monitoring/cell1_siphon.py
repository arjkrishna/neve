"""CELL 1 addendum: where do HOST episodes ARREST, and what is the siphon ceiling made of?
Reads the same stall records as cell1_report.py plus episodes_official jsonl
(seed, success, steps, final_branch_short).  OFF=33.80 (cell1_host_off_geom.py).
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


def q(v, p):
    v = sorted(v)
    return v[max(0, min(len(v) - 1, int(math.ceil(p * len(v))) - 1))]


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


for nm, (lp, jp) in RUNS.items():
    J = {}
    for ln in open(jp):
        d = json.loads(ln)
        J[d["seed"]] = d
    eps = [json.loads(l) for l in open(lp)]
    for e in eps:
        d = J[e["seed"]]
        e["succ"] = bool(d["success"])
        e["jsteps"] = d["steps"]
        e["fb"] = d.get("final_branch_short")
        e["sect"] = sect(e["pl"])
        e["srmax"] = e["maxp"] - OFF
        e["srtgt"] = e["pl"] - OFF
    print("=" * 100)
    print(nm)
    ds = [e["steps"] - e["jsteps"] for e in eps]
    print("  log-steps vs jsonl-steps agreement: identical=%d  meddiff=%d ; EPISODE_OUTCOME lines present for 82/98 (16 short) -> success taken from jsonl" % (
        sum(1 for v in ds if v == 0), int(stx.median(ds))))
    print("  truncated at 600 steps: %d/%d (%s)" % (sum(1 for e in eps if e["jsteps"] >= 600), len(eps), pct(sum(1 for e in eps if e["jsteps"] >= 600), len(eps))))
    print("  final_branch_short x success: %s" % Counter((e["fb"], e["succ"]) for e in eps).most_common(8))
    print("  ARREST (max reach) in s_RCCA mm, by section and outcome:")
    for s in ("CCA", "ICA-mid", "siphon"):
        for tag, want in (("succ", True), ("FAIL", False)):
            g = [e for e in eps if e["sect"] == s and e["succ"] is want]
            if not g:
                continue
            m = [e["srmax"] for e in g]
            t = [e["srtgt"] for e in g]
            print("    %-8s %-4s n=%2d  max reach s_RCCA med=%6.1f p10=%6.1f p90=%6.1f  | target s_RCCA med=%6.1f  | shortfall med=%6.1f" % (
                s, tag, len(g), stx.median(m), q(m, .1), q(m, .9), stx.median(t), stx.median([e["srtgt"] - e["srmax"] for e in g])))
    # the wall: histogram of arrest s_RCCA for all failures
    f = [e["srmax"] for e in eps if not e["succ"]]
    if f:
        b = Counter(int(v // 10) * 10 for v in f)
        print("  ALL FAILURES arrest s_RCCA hist(10mm): %s" % " ".join("%d:%d" % (a, b[a]) for a in sorted(b)))
        print("  ALL FAILURES arrest s_RCCA: med=%.1f p10=%.1f p90=%.1f max=%.1f" % (stx.median(f), q(f, .1), q(f, .9), max(f)))
    sc = [e["srmax"] for e in eps if e["succ"]]
    if sc:
        print("  ALL SUCCESSES arrest s_RCCA: med=%.1f p90=%.1f max=%.1f" % (stx.median(sc), q(sc, .9), max(sc)))
    # per-episode stall burden vs success, siphon only
    for s in ("siphon",):
        g = [e for e in eps if e["sect"] == s]
        if not g:
            continue
        for tag, want in (("succ", True), ("FAIL", False)):
            h = [e for e in g if e["succ"] is want]
            if not h:
                continue
            ns = [len(e["ev"]["cmd12"]) for e in h]
            print("  %s %s: n=%d  stalls/ep med=%.1f max=%d ; steps med=%.0f ; runmax med=%.0f ; maxslack med=%.1f" % (
                s, tag, len(h), stx.median(ns), max(ns), stx.median([e["jsteps"] for e in h]),
                stx.median([e["runmax"] for e in h]), stx.median([e["maxslack"] for e in h])))
    print()
