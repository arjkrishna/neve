"""CELL 1 addendum 2: is the 12-step detector exposure-limited on these HOST runs?
Also: 10 mm modal onset band, and episode-length quantiles.
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
    eps = [json.loads(l) for l in open(lp)]
    for e in eps:
        e["sect"] = sect(e["pl"])
    print("=" * 100)
    print(nm)
    L = [e["steps"] for e in eps]
    print("  episode length: min=%d p10=%d med=%d p90=%d max=%d ; eps<12 steps=%d ; eps<50=%d" % (
        min(L), q(L, .1), int(stx.median(L)), q(L, .9), max(L),
        sum(1 for v in L if v < 12), sum(1 for v in L if v < 50)))
    for s in ("ALL", "CCA", "ICA-mid", "siphon"):
        g = [e for e in eps if s == "ALL" or e["sect"] == s]
        n = len(g)
        rm = [e["runmax"] for e in g]
        # exposure: episodes that COULD host a 12-run at all
        print("  %-8s n=%2d medlen=%3d | runmax: med=%3d %%>=4=%-6s %%>=6=%-6s %%>=8=%-6s %%>=12=%-6s | eps with 4<=runmax<12 (seen only by ss<12): %d (%s)" % (
            s, n, int(stx.median([e["steps"] for e in g])), int(stx.median(rm)),
            pct(sum(1 for v in rm if v >= 4), n), pct(sum(1 for v in rm if v >= 6), n),
            pct(sum(1 for v in rm if v >= 8), n), pct(sum(1 for v in rm if v >= 12), n),
            sum(1 for v in rm if 4 <= v < 12), pct(sum(1 for v in rm if 4 <= v < 12), n)))
    ons = [k["on"] - OFF for e in eps for k in e["ev"]["cmd12"]]
    b = Counter(int(v // 10) * 10 for v in ons)
    md = max(b.items(), key=lambda kv: kv[1])
    print("  onset s_RCCA hist(10mm): %s" % " ".join("%d:%d" % (a, b[a]) for a in sorted(b)))
    print("  MODAL onset s_RCCA band(10mm) = %d-%d mm, %d/%d stalls (%s)" % (md[0], md[0] + 10, md[1], len(ons), pct(md[1], len(ons))))
    print()
