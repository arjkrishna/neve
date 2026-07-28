"""Stuck / recovery analysis — explore-only, v1b vs v1bp.

UNITS OF ANALYSIS (the whole point):
  episode success       per EPISODE  — reached target
  recovery success      per EVENT    — escaped a stall (soft|hard|grind vs unrec)
  recovery->success     per EPISODE  — stalled, escaped, AND reached target
Mixing (1) and (2) is what makes "success% ~ recovery%" look true when it isn't.

Training-time axis = CUMULATIVE EXPLORE STEPS (sum of per-episode steps, chronological),
which is exact from the worker logs and directly comparable across runs.
"""
import json
import sys
from collections import defaultdict

CFG = sys.argv[3] if len(sys.argv) > 3 else "canon"
KINDS = ("soft", "hard", "grind", "unrec")
ESC = ("soft", "hard", "grind")           # escaped = any pass-through


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(r)
    rows.sort(key=lambda r: r["t"])
    cum = 0
    for r in rows:
        cum += r["steps"]
        r["cum"] = cum
    return rows


def ev(r, cfg=None):
    return r["ev"][cfg or CFG]


def sect(pl):
    return "CCA" if pl < 146 else ("ICA-mid" if pl < 210 else "siphon")


def ep_class(r):
    """Per-episode stall/recovery class."""
    e = ev(r)
    if not e:
        return "clean"                      # never stalled
    if any(x["k"] == "unrec" for x in e):
        return "unrecovered"                # ended still stuck
    return "recovered"                      # stalled, escaped every time


def pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def bins(rows, n=8):
    """Equal-episode-count chronological bins."""
    per = max(1, len(rows) // n)
    return [rows[i:i + per] for i in range(0, len(rows), per)][:n]


def T1(rows, name):
    print(f"\n### T1 — {name}: stuck & recovery vs training time (explore only, cfg={CFG})")
    print("| bin | explore steps (k) | eps | ep succ% | stalled-ep% | events | evt/stalled-ep | recovery succ% | soft% | hard% | grind% | unrec% |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, b in enumerate(bins(rows), 1):
        e = [x for r in b for x in ev(r)]
        stalled = [r for r in b if ev(r)]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        print(f"| {i} | {b[0]['cum']//1000}-{b[-1]['cum']//1000} | {len(b)} | "
              f"{pct(sum(1 for r in b if r['succ']), len(b)):.0f}% | "
              f"{pct(len(stalled), len(b)):.0f}% | {len(e)} | "
              f"{(len(e)/len(stalled) if stalled else 0):.1f} | "
              f"{pct(esc, len(e)):.0f}% | {pct(c['soft'], len(e)):.0f}% | "
              f"{pct(c['hard'], len(e)):.0f}% | {pct(c['grind'], len(e)):.0f}% | "
              f"{pct(c['unrec'], len(e)):.0f}% |")


def T2(rows, name):
    print(f"\n### T2 — {name}: THE THREE SUCCESSES — where episode success comes from")
    print("| bin | eps | ep succ% | clean eps (succ%) | recovered eps (succ%) | unrec eps (succ%) | % of successes via recovery |")
    print("|---|---|---|---|---|---|---|")
    for i, b in enumerate(bins(rows), 1):
        g = defaultdict(list)
        for r in b:
            g[ep_class(r)].append(r)
        s_tot = sum(1 for r in b if r["succ"])
        s_rec = sum(1 for r in g["recovered"] if r["succ"])
        row = [f"| {i} | {len(b)} | {pct(s_tot, len(b)):.0f}% "]
        for k in ("clean", "recovered", "unrecovered"):
            n = len(g[k]); s = sum(1 for r in g[k] if r["succ"])
            row.append(f"| {n} ({pct(s, n):.0f}%) " if n else "| 0 (—) ")
        row.append(f"| {pct(s_rec, s_tot):.0f}% |")
        print("".join(row))


def T3(rows, name):
    print(f"\n### T3 — {name}: does recovery TYPE predict episode success?")
    print("(episodes grouped by the best recovery they achieved; unrec excluded from the first three)")
    print("| recovery profile | episodes | episode succ% | median steps | mean retract mm |")
    print("|---|---|---|---|---|")
    groups = {"soft (any)": [], "hard (no soft)": [], "grind only": [], "unrecovered (any)": [], "never stalled": []}
    for r in rows:
        e = ev(r)
        if not e:
            groups["never stalled"].append(r); continue
        ks = {x["k"] for x in e}
        if "unrec" in ks:
            groups["unrecovered (any)"].append(r)
        elif "soft" in ks:
            groups["soft (any)"].append(r)
        elif "hard" in ks:
            groups["hard (no soft)"].append(r)
        else:
            groups["grind only"].append(r)
    for k, g in groups.items():
        if not g:
            continue
        st = sorted(r["steps"] for r in g)
        rr = [x["r"] for r in g for x in ev(r)]
        print(f"| {k} | {len(g)} | {pct(sum(1 for r in g if r['succ']), len(g)):.0f}% | "
              f"{st[len(st)//2]} | {(sum(rr)/len(rr) if rr else 0):.1f} |")


def T4(rows, name):
    print(f"\n### T4 — {name}: stuck & recovery by DEPTH (path_len band)")
    print("| band | eps | ep succ% | stalled-ep% | events | recovery succ% | soft% | hard% | grind% | unrec% |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in ("CCA", "ICA-mid", "siphon"):
        b = [r for r in rows if sect(r["pl"]) == s]
        if not b:
            continue
        e = [x for r in b for x in ev(r)]
        stalled = [r for r in b if ev(r)]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        print(f"| {s} | {len(b)} | {pct(sum(1 for r in b if r['succ']), len(b)):.0f}% | "
              f"{pct(len(stalled), len(b)):.0f}% | {len(e)} | {pct(esc, len(e)):.0f}% | "
              f"{pct(c['soft'], len(e)):.0f}% | {pct(c['hard'], len(e)):.0f}% | "
              f"{pct(c['grind'], len(e)):.0f}% | {pct(c['unrec'], len(e)):.0f}% |")


def T6(rows, name):
    """Correlation across bins: does recovery track episode success?"""
    xs, ys, zs = [], [], []
    for b in bins(rows, 10):
        e = [x for r in b for x in ev(r)]
        if not e:
            continue
        esc = sum(1 for x in e if x["k"] in ESC)
        xs.append(pct(sum(1 for r in b if r["succ"]), len(b)))
        ys.append(pct(esc, len(e)))
        zs.append(pct(sum(1 for x in e if x["k"] == "soft"), len(e)))

    def r(a, b_):
        n = len(a)
        if n < 3:
            return float("nan")
        ma, mb = sum(a) / n, sum(b_) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b_))
        da = sum((x - ma) ** 2 for x in a) ** .5
        db = sum((y - mb) ** 2 for y in b_) ** .5
        return num / (da * db) if da and db else float("nan")
    print(f"\n### T6 — {name}: correlation across {len(xs)} chronological bins")
    print(f"- episode-success% vs recovery-success%: r = {r(xs, ys):+.2f}")
    print(f"- episode-success% vs soft-share%:       r = {r(xs, zs):+.2f}")
    print(f"- episode succ% range {min(xs):.0f}-{max(xs):.0f} | recovery succ% range {min(ys):.0f}-{max(ys):.0f}")


def T7(rows, name):
    print(f"\n### T7 — {name}: threshold sensitivity (whole run)")
    print("| cfg | events | stalled-ep% | recovery succ% | soft% | hard% | grind% | unrec% |")
    print("|---|---|---|---|---|---|---|---|")
    for cfg in ("sens", "canon", "strict"):
        e = [x for r in rows for x in r["ev"][cfg]]
        stalled = sum(1 for r in rows if r["ev"][cfg])
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        print(f"| {cfg} | {len(e)} | {pct(stalled, len(rows)):.0f}% | {pct(esc, len(e)):.0f}% | "
              f"{pct(c['soft'], len(e)):.0f}% | {pct(c['hard'], len(e)):.0f}% | "
              f"{pct(c['grind'], len(e)):.0f}% | {pct(c['unrec'], len(e)):.0f}% |")


if __name__ == "__main__":
    a = load(sys.argv[1]); b = load(sys.argv[2])
    for rows, name in ((a, "v1b"), (b, "v1bp")):
        print(f"\n{'='*78}\n## {name} — {len(rows)} explore episodes, "
              f"{rows[-1]['cum']:,} explore steps, "
              f"{pct(sum(1 for r in rows if r['succ']), len(rows)):.1f}% episode success\n{'='*78}")
        T1(rows, name); T2(rows, name); T3(rows, name); T4(rows, name); T6(rows, name); T7(rows, name)

    print(f"\n{'='*78}\n### T5 — HEAD-TO-HEAD (whole run, cfg={CFG})\n{'='*78}")
    print("| metric | v1b | v1bp |")
    print("|---|---|---|")
    def agg(rows):
        e = [x for r in rows for x in ev(r)]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        g = defaultdict(list)
        for r in rows:
            g[ep_class(r)].append(r)
        s_tot = sum(1 for r in rows if r["succ"])
        return {
            "episodes": len(rows), "explore steps": rows[-1]["cum"],
            "episode succ%": pct(s_tot, len(rows)),
            "stalled-ep%": pct(sum(1 for r in rows if ev(r)), len(rows)),
            "events": len(e), "recovery succ%": pct(esc, len(e)),
            "soft%": pct(c["soft"], len(e)), "hard%": pct(c["hard"], len(e)),
            "grind%": pct(c["grind"], len(e)), "unrec%": pct(c["unrec"], len(e)),
            "clean-ep succ%": pct(sum(1 for r in g["clean"] if r["succ"]), len(g["clean"])),
            "recovered-ep succ%": pct(sum(1 for r in g["recovered"] if r["succ"]), len(g["recovered"])),
            "% successes via recovery": pct(sum(1 for r in g["recovered"] if r["succ"]), s_tot),
            "mean cath-lead frac": sum(r["cath_lead_frac"] for r in rows) / len(rows),
        }
    A, B = agg(a), agg(b)
    for k in A:
        va, vb = A[k], B[k]
        f = (lambda v: f"{v:,.0f}" if abs(v) > 999 else (f"{v:.2f}" if k.endswith("frac") else f"{v:.1f}"))
        print(f"| {k} | {f(va)} | {f(vb)} |")
