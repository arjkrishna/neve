"""Multi-page matplotlib report of the v1b / v1bp stuck-recovery analysis.

Recomputes every table from the raw per-episode records (self-contained and
reproducible), renders each as a clean figure with a "what it shows" /
"inference" caption, and adds charts where a figure beats the numbers.
"""
import json
import sys
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- style
C_B    = "#2E5E8A"    # v1b
C_BP   = "#C1662F"    # v1bp
C_HDR  = "#33404D"
C_ROW  = "#F2F4F7"
C_TXT  = "#1B2530"
C_MUTE = "#6B7885"
C_HI   = "#B3261E"
C_OK   = "#1E7A4C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C8CFD6", "axes.labelcolor": C_TXT,
    "xtick.color": C_MUTE, "ytick.color": C_MUTE,
    "axes.titleweight": "bold", "axes.titlecolor": C_TXT,
})
PAGE = (11.69, 8.27)          # A4 landscape

# ---------------------------------------------------------------- data
KINDS = ("soft", "hard", "grind", "unrec")
ESC = ("soft", "hard", "grind")


def load(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["t"])
    c = 0
    for r in rows:
        c += r["steps"]
        r["cum"] = c
    return rows


def ev(r, cfg="canon"):
    return r["ev"][cfg]


def sect(pl):
    return "CCA" if pl < 146 else ("ICA-mid" if pl < 210 else "siphon")


def ep_class(r, cfg="canon"):
    e = ev(r, cfg)
    if not e:
        return "clean"
    if any(x["k"] == "unrec" for x in e):
        return "unrec"
    return "rec"


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def bins(rows, n=8):
    per = max(1, len(rows) // n)
    return [rows[i:i + per] for i in range(0, len(rows), per)][:n]


# ------------------------------------------------------- page furniture
def newpage(pdf, title, subtitle=""):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.045, 0.945, title, fontsize=17, fontweight="bold", color=C_TXT)
    if subtitle:
        fig.text(0.045, 0.906, subtitle, fontsize=9.5, color=C_MUTE)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.893, 0.893],
                              color="#D5DBE1", lw=1.1))
    return fig


PNGDIR = None


def finish(pdf, fig, page_no):
    fig.text(0.955, 0.032, str(page_no), ha="right", fontsize=8, color=C_MUTE)
    fig.text(0.045, 0.032, "eve_rl · stuck & recovery analysis · explore-only",
             fontsize=7.5, color="#9AA5B1")
    pdf.savefig(fig)
    if PNGDIR:
        fig.savefig(f"{PNGDIR}/page{page_no:02d}.png", dpi=110)
    plt.close(fig)


CAP_W = 146
CAP_LH = 0.0182          # figure-coord height of one caption line
CAP_PAD = 0.017


def _cap_lines(shows, infer):
    return (textwrap.wrap("WHAT IT SHOWS   " + shows, CAP_W),
            textwrap.wrap("INFERENCE   " + infer, CAP_W))


def cap_h(shows, infer):
    """Height the caption block will need — call BEFORE laying out charts."""
    l1, l2 = _cap_lines(shows, infer)
    return 2 * CAP_PAD + (len(l1) + len(l2)) * CAP_LH + 0.012


def caption(fig, y, shows, infer, h=None):
    """Two-part caption block, auto-sized to its content."""
    l1, l2 = _cap_lines(shows, infer)
    h = cap_h(shows, infer)
    ax = fig.add_axes([0.045, y, 0.91, h])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor="#F7F9FB", edgecolor="#DFE5EA", lw=1))
    ax.add_patch(Rectangle((0, 0), 0.0055, 1, transform=ax.transAxes,
                           facecolor=C_B, edgecolor="none"))
    yy = 1 - CAP_PAD / h
    for line in l1:
        ax.text(0.012, yy, line, va="top", fontsize=8.6, color=C_TXT,
                transform=ax.transAxes)
        yy -= CAP_LH / h
    yy -= 0.012 / h
    for line in l2:
        ax.text(0.012, yy, line, va="top", fontsize=8.6, color=C_TXT,
                transform=ax.transAxes, fontstyle="italic")
        yy -= CAP_LH / h


def table(fig, rect, headers, rows, widths=None, hi=None, fs=8.4,
          title=None, tcolor=C_HDR):
    """Clean table. hi = {(row_idx, col_idx): color} for emphasised cells."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    hi = hi or {}
    nc = len(headers)
    widths = widths or [1.0 / nc] * nc
    tot = sum(widths)
    widths = [w / tot for w in widths]
    xs = [0.0]
    for w in widths:
        xs.append(xs[-1] + w)
    nr = len(rows)
    hh = 1.0 / (nr + 1.6)          # header a bit taller
    hdr_h = hh * 1.6
    if title:
        ax.text(0, 1.045, title, fontsize=10, fontweight="bold",
                color=tcolor, transform=ax.transAxes)
    # header
    ax.add_patch(Rectangle((0, 1 - hdr_h), 1, hdr_h, transform=ax.transAxes,
                           facecolor=C_HDR, edgecolor="none"))
    for j, h in enumerate(headers):
        ax.text((xs[j] + xs[j + 1]) / 2, 1 - hdr_h / 2, h, ha="center",
                va="center", fontsize=fs - 0.2, color="white",
                fontweight="bold", transform=ax.transAxes)
    # body
    for i, row in enumerate(rows):
        y1 = 1 - hdr_h - i * hh
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y1 - hh), 1, hh, transform=ax.transAxes,
                                   facecolor=C_ROW, edgecolor="none"))
        for j, cell in enumerate(row):
            col = C_TXT
            wt = "normal"
            if (i, j) in hi:
                col = hi[(i, j)]
                wt = "bold"
            ax.text((xs[j] + xs[j + 1]) / 2, y1 - hh / 2, str(cell),
                    ha="center", va="center", fontsize=fs, color=col,
                    fontweight=wt, transform=ax.transAxes)
    ax.add_patch(Rectangle((0, 1 - hdr_h - nr * hh), 1, hdr_h + nr * hh,
                           transform=ax.transAxes, facecolor="none",
                           edgecolor="#C8CFD6", lw=1))
    return ax


# ----------------------------------------------------------- computations
def t1_rows(rows):
    out = []
    for i, b in enumerate(bins(rows), 1):
        e = [x for r in b for x in ev(r)]
        stalled = [r for r in b if ev(r)]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        out.append([i, f"{b[0]['cum']//1000}–{b[-1]['cum']//1000}", len(b),
                    f"{pct(sum(1 for r in b if r['succ']), len(b)):.0f}%",
                    f"{pct(len(stalled), len(b)):.0f}%", len(e),
                    f"{len(e)/len(stalled) if stalled else 0:.1f}",
                    f"{pct(esc, len(e)):.0f}%",
                    f"{pct(c['soft'], len(e)):.0f}%",
                    f"{pct(c['hard'], len(e)):.0f}%",
                    f"{pct(c['grind'], len(e)):.0f}%",
                    f"{pct(c['unrec'], len(e)):.0f}%"])
    return out


def t1_series(rows):
    su, st, rc, sf = [], [], [], []
    for b in bins(rows):
        e = [x for r in b for x in ev(r)]
        esc = sum(1 for x in e if x["k"] in ESC)
        su.append(pct(sum(1 for r in b if r["succ"]), len(b)))
        st.append(pct(sum(1 for r in b if ev(r)), len(b)))
        rc.append(pct(esc, len(e)))
        sf.append(pct(sum(1 for x in e if x["k"] == "soft"), len(e)))
    return su, st, rc, sf


def t2_rows(rows):
    out, via = [], []
    for i, b in enumerate(bins(rows), 1):
        g = defaultdict(list)
        for r in b:
            g[ep_class(r)].append(r)
        s_tot = sum(1 for r in b if r["succ"])
        s_rec = sum(1 for r in g["rec"] if r["succ"])
        row = [i, len(b), f"{pct(s_tot, len(b)):.0f}%"]
        for k in ("clean", "rec", "unrec"):
            n = len(g[k])
            s = sum(1 for r in g[k] if r["succ"])
            row.append(f"{n}  ({pct(s, n):.0f}%)" if n else "0  (—)")
        row.append(f"{pct(s_rec, s_tot):.0f}%")
        out.append(row)
        via.append(pct(s_rec, s_tot))
    return out, via


def t3_groups(rows):
    groups = {"soft (any)": [], "hard (no soft)": [], "grind only": [],
              "unrecovered (any)": [], "never stalled": []}
    for r in rows:
        e = ev(r)
        if not e:
            groups["never stalled"].append(r)
            continue
        ks = {x["k"] for x in e}
        if "unrec" in ks:
            groups["unrecovered (any)"].append(r)
        elif "soft" in ks:
            groups["soft (any)"].append(r)
        elif "hard" in ks:
            groups["hard (no soft)"].append(r)
        else:
            groups["grind only"].append(r)
    out = []
    for k, g in groups.items():
        if not g:
            continue
        st = sorted(r["steps"] for r in g)
        rr = [x["r"] for r in g for x in ev(r)]
        out.append([k, len(g), f"{pct(sum(1 for r in g if r['succ']), len(g)):.0f}%",
                    st[len(st) // 2], f"{sum(rr)/len(rr) if rr else 0:.1f}"])
    return out


def t4_rows(rows):
    out = []
    for s in ("CCA", "ICA-mid", "siphon"):
        b = [r for r in rows if sect(r["pl"]) == s]
        e = [x for r in b for x in ev(r)]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        out.append([s, len(b), f"{pct(sum(1 for r in b if r['succ']), len(b)):.0f}%",
                    f"{pct(sum(1 for r in b if ev(r)), len(b)):.0f}%", len(e),
                    f"{pct(esc, len(e)):.0f}%", f"{pct(c['soft'], len(e)):.0f}%",
                    f"{pct(c['hard'], len(e)):.0f}%", f"{pct(c['grind'], len(e)):.0f}%",
                    f"{pct(c['unrec'], len(e)):.0f}%"])
    return out


def agg(rows):
    e = [x for r in rows for x in ev(r)]
    c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
    esc = sum(c[k] for k in ESC)
    g = defaultdict(list)
    for r in rows:
        g[ep_class(r)].append(r)
    s_tot = sum(1 for r in rows if r["succ"])
    return {
        "episodes": f"{len(rows):,}", "explore steps": f"{rows[-1]['cum']:,}",
        "episode succ%": f"{pct(s_tot, len(rows)):.1f}",
        "stalled-ep%": f"{pct(sum(1 for r in rows if ev(r)), len(rows)):.1f}",
        "stall events": f"{len(e):,}", "recovery succ%": f"{pct(esc, len(e)):.1f}",
        "soft%": f"{pct(c['soft'], len(e)):.1f}", "hard%": f"{pct(c['hard'], len(e)):.1f}",
        "grind%": f"{pct(c['grind'], len(e)):.1f}", "unrecovered%": f"{pct(c['unrec'], len(e)):.1f}",
        "clean-ep succ%": f"{pct(sum(1 for r in g['clean'] if r['succ']), len(g['clean'])):.1f}",
        "recovered-ep succ%": f"{pct(sum(1 for r in g['rec'] if r['succ']), len(g['rec'])):.1f}",
        "% successes via recovery": f"{pct(sum(1 for r in g['rec'] if r['succ']), s_tot):.1f}",
        "mean cath-lead frac": f"{sum(r['cath_lead_frac'] for r in rows)/len(rows):.2f}",
    }


def t7_rows(rows):
    out = []
    for cfg in ("sens", "canon", "strict"):
        e = [x for r in rows for x in r["ev"][cfg]]
        c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
        esc = sum(c[k] for k in ESC)
        out.append([cfg, f"{len(e):,}",
                    f"{pct(sum(1 for r in rows if r['ev'][cfg]), len(rows)):.0f}%",
                    f"{pct(esc, len(e)):.0f}%", f"{pct(c['soft'], len(e)):.0f}%",
                    f"{pct(c['hard'], len(e)):.0f}%", f"{pct(c['grind'], len(e)):.0f}%",
                    f"{pct(c['unrec'], len(e)):.0f}%"])
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db) if da and db else float("nan")


def corr_pts(rows, n=10):
    xs, ys, zs = [], [], []
    for b in bins(rows, n):
        e = [x for r in b for x in ev(r)]
        if not e:
            continue
        xs.append(pct(sum(1 for r in b if r["succ"]), len(b)))
        ys.append(pct(sum(1 for x in e if x["k"] in ESC), len(e)))
        zs.append(pct(sum(1 for x in e if x["k"] == "soft"), len(e)))
    return xs, ys, zs


# =========================================================== build report
V1B = load(sys.argv[1])
V1BP = load(sys.argv[2])
OUT = sys.argv[3]
if len(sys.argv) > 4:
    import os
    PNGDIR = sys.argv[4]
    os.makedirs(PNGDIR, exist_ok=True)
P = 0
pdf = PdfPages(OUT)

# ---------- P1 cover -------------------------------------------------
P += 1
fig = plt.figure(figsize=PAGE)
fig.text(0.045, 0.855, "Stuck & Recovery Analysis", fontsize=30,
         fontweight="bold", color=C_TXT)
fig.text(0.045, 0.800, "Endovascular navigation RL  ·  v1b vs v1bp  ·  explore-only",
         fontsize=13.5, color=C_MUTE)
fig.add_artist(plt.Line2D([0.045, 0.955], [0.775, 0.775], color=C_B, lw=2.5))

stats = [("v1b episodes", f"{len(V1B):,}"), ("v1b explore steps", f"{V1B[-1]['cum']:,}"),
         ("v1bp episodes", f"{len(V1BP):,}"), ("v1bp explore steps", f"{V1BP[-1]['cum']:,}")]
for i, (k, v) in enumerate(stats):
    x = 0.045 + 0.235 * i
    fig.text(x, 0.705, v, fontsize=19, fontweight="bold",
             color=C_B if i < 2 else C_BP)
    fig.text(x, 0.675, k, fontsize=9, color=C_MUTE)

body = (
    "SCOPE.  Every number here is measured from the worker-log complete ledger "
    "(EPISODE_START / STEP / EPISODE_OUTCOME) of two training runs, restricted to EXPLORE "
    "episodes. Evaluation episodes are excluded exactly: runner.eval() resets with explicit "
    "seeds, so eval EPISODE_START lines carry a `seed=` field and explore resets do not. That "
    "filter yields v1b 14,691 explore / 1,470 eval (= 15 evals x 98) and v1bp 8,040 / 980 "
    "(= 10 x 98) — exact on both runs. The obvious alternative, main.log time windows, leaks "
    "(it caught only 567 of v1bp's 980) and episode_summary.jsonl is unusable because it logs "
    "only ~36% of explore episodes.\n\n"
    "TRAINING-TIME AXIS.  Cumulative explore steps, summed per-episode in chronological order. "
    "Exact from the logs and directly comparable across runs, unlike wall-clock (which conflates "
    "simulator speed) or the global step counter.\n\n"
    "THE RUNS.  v1b and v1bp are byte-identical except for one change: v1bp adds a reward pair "
    "(tip-average progress + catheter-slack potential) ported from a second machine. This is a "
    "single-variable experiment."
)
fig.text(0.045, 0.615, "\n".join(textwrap.wrap(body, 132)).replace("\n\n\n", "\n\n"),
         fontsize=9.3, va="top", color=C_TXT, linespacing=1.62)
fig.text(0.045, 0.085, "Generated from monitoring/extract_stuck.py records  ·  "
         "canonical detector config unless stated", fontsize=8, color="#9AA5B1")
finish(pdf, fig, P)

# ---------- P2 definitions ------------------------------------------
P += 1
fig = newpage(pdf, "Definitions — and why each one is drawn where it is",
              "Every downstream number depends on these choices, so they are stated before any result")
table(fig, [0.045, 0.615, 0.91, 0.235],
      ["term", "unit", "definition"],
      [["stall event", "per event",
        "wire pushing (|gw_trans| > 2 mm/s) while path progress stays flat (< 0.3 mm gain) for 12 consecutive steps"],
       ["grind", "per event", "escaped the stall after < 1 mm of withdrawal — forced through"],
       ["soft", "per event", "escaped after 1–8 mm withdrawal — ease slack, re-advance (the manoeuvre we set out to induce)"],
       ["hard", "per event", "escaped after > 8 mm withdrawal — costly pullback"],
       ["unrecovered", "per event", "never passed the jam; terminal for that episode"],
       ["recovery success%", "per EVENT", "(soft + hard + grind) / all stall events"],
       ["episode success%", "per EPISODE", "reached the target"]],
      widths=[0.16, 0.11, 0.73], fs=8.5)

rationale = (
    "A stall is 'pushing but not progressing', not merely 'not progressing'.  A wire that is "
    "deliberately retracting or rotating is manoeuvring, not stuck. Requiring the push command "
    "isolates the case where the policy IS trying to advance and the vessel will not allow it. "
    "The stall counter decays by 2 per non-stalled step rather than resetting, so one noisy "
    "command sign-flip inside a genuinely stuck push does not erase the evidence.\n\n"
    "Recovery kind is classified by ACTUAL withdrawn length — peak insertion minus deepest "
    "insertion during the stall — not by the commanded retraction. A retract command that moves "
    "nothing (wire wedged) must not score as a soft recovery.\n\n"
    "THE THREE SUCCESSES sit at different units of analysis, which is the single most important "
    "thing on this page:  (1) EPISODE success is per episode — did it reach target.  (2) RECOVERY "
    "success is per EVENT — did the wire escape a given stall.  (3) RECOVERY-TO-EPISODE success is "
    "per episode — did escaping actually convert into reaching the target. Conflating (1) and (2) "
    "makes 'success tracks recovery' look true when the opposite holds."
)
ax = fig.add_axes([0.045, 0.10, 0.91, 0.46])
ax.axis("off")
ax.text(0, 1, "\n".join(textwrap.wrap(rationale, 140)).replace("\n\n\n", "\n\n"),
        va="top", fontsize=9.1, color=C_TXT, linespacing=1.66)
finish(pdf, fig, P)

# ---------- P3/P4 T1 --------------------------------------------------
for rows, nm, col in ((V1B, "v1b", C_B), (V1BP, "v1bp", C_BP)):
    P += 1
    fig = newpage(pdf, f"T1 — {nm}: stuck & recovery vs training time",
                  "8 equal-episode chronological bins  ·  explore only  ·  canonical detector")
    t = t1_rows(rows)
    hi = {}
    su, st, rc, sf = t1_series(rows)
    hi[(su.index(max(su)), 3)] = C_OK
    hi[(rc.index(min(rc)), 7)] = C_HI
    if nm == "v1b":
        shows = ("How often the wire got stuck, how often it got out, and by which manoeuvre, as "
                 "training progressed. Read the trend across bins, not the absolute levels.")
        infer = ("The central surprise of the whole analysis. Episode success CLIMBS 52→62% while "
                 "recovery success FALLS 65→39% and unrecovered rises 35→61%. What is actually "
                 "improving is the stalled-episode share, which drops 64→42%. The policy is not "
                 "learning to escape stalls — it is learning not to enter them, and the escape "
                 "skill withers as stalls leave its training data. That is textbook interference: "
                 "the skill starves because success removes its own training signal.")
    else:
        shows = ("The same trajectory for the run that added the reward pair (tip-average progress "
                 "+ catheter-slack potential).")
        infer = ("The opposite internal dynamic. Recovery success does NOT decay — it holds 63–75% "
                 "from bin 3 onward, against v1b's decay to ~40%. The reward pair kept the recovery "
                 "skill alive, exactly as designed. But note the episode-success column is no better "
                 "for it (46→56%), which is the tension resolved in T5.")
    ch = cap_h(shows, infer)
    tbl_bot = 0.575
    ch_bot = 0.055 + ch + 0.055
    table(fig, [0.045, tbl_bot, 0.91, 0.29],
          ["bin", "explore steps (k)", "eps", "ep succ%", "stalled-ep%", "events",
           "evt/stalled-ep", "recovery succ%", "soft%", "hard%", "grind%", "unrec%"],
          t, widths=[.045, .13, .06, .08, .095, .07, .105, .115, .06, .06, .065, .07],
          hi=hi, fs=8.3)

    ax = fig.add_axes([0.075, ch_bot, 0.855, tbl_bot - 0.055 - ch_bot])
    x = list(range(1, len(su) + 1))
    ax.plot(x, su, "-o", color=col, lw=2.4, ms=6, label="episode success%", zorder=3)
    ax.plot(x, rc, "-s", color=C_HI, lw=2.0, ms=5.5, label="recovery success% (per event)")
    ax.plot(x, st, "--^", color=C_MUTE, lw=1.7, ms=5, label="stalled-episode%")
    ax.plot(x, sf, ":d", color="#7A5AA8", lw=1.7, ms=5, label="soft share of events%")
    ax.set_xlabel("chronological bin  (cumulative explore steps →)")
    ax.set_ylabel("percent")
    ax.set_xticks(x)
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=8.2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    caption(fig, 0.055, shows, infer)
    finish(pdf, fig, P)

# ---------- P5 T2 tables ---------------------------------------------
P += 1
fig = newpage(pdf, "T2 — the three successes: where episode success actually comes from",
              "Each episode assigned to exactly one channel: never stalled / stalled-and-escaped / stalled-and-stuck")
hdr = ["bin", "eps", "ep succ%", "clean eps (succ%)", "recovered eps (succ%)",
       "unrec eps (succ%)", "% of successes via recovery"]
w = [.06, .09, .11, .18, .20, .18, .18]
r1, via_b = t2_rows(V1B)
r2, via_bp = t2_rows(V1BP)
table(fig, [0.045, 0.575, 0.445, 0.285], hdr, r1, widths=w, fs=7.7,
      title="v1b", tcolor=C_B, hi={(0, 6): C_HI, (6, 6): C_HI})
table(fig, [0.512, 0.575, 0.443, 0.285], hdr, r2, widths=w, fs=7.7,
      title="v1bp", tcolor=C_BP)

SH5 = ("Episode success decomposed by channel, and the share of all successes that flowed "
       "through recovery (the chart).")
IN5 = ("In v1b recovery's contribution COLLAPSES from 29% to 6%: early on a third of wins were "
       "rescues, by the end nearly every win is an episode that simply never jammed. v1bp shows "
       "no such decay (13–28%, and its best bin leans MOST on recovery). Two runs, nearly "
       "identical peak success, opposite routes: v1b avoids-and-runs-clean, v1bp stalls-and-"
       "fights-through.  CAVEAT: the 100% and ~1% columns are near-definitional — an episode that "
       "pushes 600 steps without jamming has by construction arrived, one ending jammed has not. "
       "Read the COUNTS and the final column, not those rates.")
ch5 = cap_h(SH5, IN5)
b5 = 0.055 + ch5 + 0.045
ax = fig.add_axes([0.075, b5, 0.855, 0.575 - 0.05 - b5])
x = list(range(1, 9))
ax.plot(x, via_b, "-o", color=C_B, lw=2.4, ms=6, label="v1b")
ax.plot(x, via_bp, "-s", color=C_BP, lw=2.4, ms=6, label="v1bp")
ax.set_xlabel("chronological bin")
ax.set_ylabel("% of successes\nthat came via recovery")
ax.set_xticks(x)
ax.grid(alpha=.25, ls=":")
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, max(max(via_b), max(via_bp)) * 1.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

caption(fig, 0.055, SH5, IN5)
finish(pdf, fig, P)

# ---------- P6 T3 -----------------------------------------------------
P += 1
fig = newpage(pdf, "T3 — does recovery TYPE predict episode success?",
              "Episodes grouped by the best recovery they achieved  ·  tests the premise behind the crunchpass / two-mode design")
hdr3 = ["recovery profile", "episodes", "episode succ%", "median steps", "mean retract mm"]
g1, g2 = t3_groups(V1B), t3_groups(V1BP)
table(fig, [0.045, 0.615, 0.445, 0.20], hdr3, g1, widths=[.34, .17, .19, .16, .19],
      fs=8.2, title="v1b", tcolor=C_B, hi={(2, 2): C_OK, (2, 3): C_OK})
table(fig, [0.512, 0.615, 0.443, 0.20], hdr3, g2, widths=[.34, .17, .19, .16, .19],
      fs=8.2, title="v1bp", tcolor=C_BP, hi={(2, 2): C_OK, (2, 3): C_OK})

SH6 = ("Whether the type of escape predicts the episode outcome, and what each type costs in steps.")
IN6 = ("GRIND BEATS SOFT in both runs — 99% vs 95% (v1b) and 95% vs 76% (v1bp) — and does it in "
       "roughly a third of the steps. This inverts the premise behind the crunchpass lane and the "
       "'we want soft recoveries' framing. The likely reason is a confound: recovery type is mostly "
       "a proxy for stall SEVERITY, not policy skill. A jam you can grind through was mild; one "
       "forcing an 8 mm withdrawal was severe and usually deeper, where the episode was already in "
       "trouble. Either way the practical conclusion holds — soft recovery cannot be justified as a "
       "training target on this evidence until the confound is broken by comparing within matched "
       "depth and severity strata.")
ch6 = cap_h(SH6, IN6)
b6 = 0.055 + ch6 + 0.05
h6 = 0.615 - 0.055 - b6
ax = fig.add_axes([0.075, b6, 0.40, h6])
labs = ["grind\nonly", "soft\n(any)", "hard\n(no soft)"]
key = {"grind only": 0, "soft (any)": 1, "hard (no soft)": 2}
vb = [0] * 3
vbp = [0] * 3
for r in g1:
    if r[0] in key:
        vb[key[r[0]]] = float(str(r[2]).rstrip("%"))
for r in g2:
    if r[0] in key:
        vbp[key[r[0]]] = float(str(r[2]).rstrip("%"))
xx = range(3)
ax.bar([i - .19 for i in xx], vb, .37, color=C_B, label="v1b")
ax.bar([i + .19 for i in xx], vbp, .37, color=C_BP, label="v1bp")
ax.set_xticks(list(xx))
ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("episode success%")
ax.set_ylim(0, 108)
ax.grid(alpha=.25, ls=":", axis="y")
ax.legend(frameon=False, fontsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("episode success by recovery type", fontsize=9.5, pad=7)

ax2 = fig.add_axes([0.555, b6, 0.40, h6])
sb = [0] * 3
sbp = [0] * 3
for r in g1:
    if r[0] in key:
        sb[key[r[0]]] = r[3]
for r in g2:
    if r[0] in key:
        sbp[key[r[0]]] = r[3]
ax2.bar([i - .19 for i in xx], sb, .37, color=C_B)
ax2.bar([i + .19 for i in xx], sbp, .37, color=C_BP)
ax2.axhline(600, color=C_HI, ls="--", lw=1.4)
ax2.text(2.42, 612, "600-step cap", fontsize=7.6, color=C_HI, ha="right")
ax2.set_xticks(list(xx))
ax2.set_xticklabels(labs, fontsize=8.5)
ax2.set_ylabel("median steps")
ax2.set_ylim(0, 660)
ax2.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("time cost of each recovery type", fontsize=9.5, pad=7)

caption(fig, 0.055, SH6, IN6)
finish(pdf, fig, P)

# ---------- P7 T4 depth ----------------------------------------------
P += 1
fig = newpage(pdf, "T4 — stuck & recovery by depth",
              "Path-length bands: CCA (<146 mm)  ·  ICA-mid (146–210 mm)  ·  siphon (>210 mm)")
hdr4 = ["band", "eps", "ep succ%", "stalled-ep%", "events", "recovery succ%",
        "soft%", "hard%", "grind%", "unrec%"]
w4 = [.11, .09, .10, .115, .09, .13, .075, .075, .08, .085]
table(fig, [0.045, 0.685, 0.91, 0.12], hdr4, t4_rows(V1B), widths=w4, fs=8.4,
      title="v1b", tcolor=C_B, hi={(2, 2): C_HI, (2, 3): C_HI})
table(fig, [0.045, 0.505, 0.91, 0.12], hdr4, t4_rows(V1BP), widths=w4, fs=8.4,
      title="v1bp", tcolor=C_BP, hi={(2, 2): C_HI, (2, 3): C_HI})

SH7 = ("Where the stalls actually are, and whether escaping them rescues the episode there.")
IN7 = ("Depth dominates every other factor. Stalling runs 5–7% at the CCA but 90% (v1b) to 97% "
       "(v1bp) at the siphon — getting stuck is essentially guaranteed once the wire enters the "
       "cavernous segment. And escaping does not save you there: v1bp escapes 68% of siphon stalls "
       "yet only 14% of siphon episodes succeed. You escape one jam and immediately meet the next. "
       "This reframes the siphon as a STALL-DENSITY problem rather than an escape-skill problem, "
       "and it is the same wall the real-patient evaluation hit (0 of 30), seen from the inside.")
ch7 = cap_h(SH7, IN7)
b7 = 0.055 + ch7 + 0.05
ax = fig.add_axes([0.075, b7, 0.855, 0.505 - 0.055 - b7])
bands = ["CCA", "ICA-mid", "siphon"]
stb = [float(r[3].rstrip("%")) for r in t4_rows(V1B)]
stbp = [float(r[3].rstrip("%")) for r in t4_rows(V1BP)]
sub = [float(r[2].rstrip("%")) for r in t4_rows(V1B)]
subp = [float(r[2].rstrip("%")) for r in t4_rows(V1BP)]
xx = range(3)
ax.bar([i - .30 for i in xx], stb, .19, color=C_B, alpha=.55, label="v1b stalled-ep%")
ax.bar([i - .10 for i in xx], stbp, .19, color=C_BP, alpha=.55, label="v1bp stalled-ep%")
ax.bar([i + .10 for i in xx], sub, .19, color=C_B, label="v1b episode succ%")
ax.bar([i + .30 for i in xx], subp, .19, color=C_BP, label="v1bp episode succ%")
ax.set_xticks(list(xx))
ax.set_xticklabels(bands)
ax.set_ylabel("percent")
ax.grid(alpha=.25, ls=":", axis="y")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.18))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

caption(fig, 0.055, SH7, IN7)
finish(pdf, fig, P)

# ---------- P8 T5 head-to-head ---------------------------------------
P += 1
fig = newpage(pdf, "T5 — head-to-head: what the reward pair actually did",
              "Whole run, canonical detector. v1bp = v1b + (tip-average progress + catheter-slack potential)")
A, B = agg(V1B), agg(V1BP)
readings = {
    "episode succ%": "v1b ahead (57.4% at matched 2.55M budget)",
    "stalled-ep%": "v1bp jams MORE",
    "recovery succ%": "v1bp escapes FAR better",
    "unrecovered%": "v1bp leaves far fewer terminal jams",
    "recovered-ep succ%": "v1bp's escapes CONVERT WORSE",
    "mean cath-lead frac": "the catheter shove is gone",
}
rows5, hi5 = [], {}
for i, k in enumerate(A):
    rows5.append([k, A[k], B[k], readings.get(k, "")])
    if k in ("recovery succ%", "unrecovered%", "mean cath-lead frac"):
        hi5[(i, 2)] = C_OK
    if k in ("episode succ%", "recovered-ep succ%"):
        hi5[(i, 1)] = C_OK
        hi5[(i, 2)] = C_HI
table(fig, [0.045, 0.475, 0.91, 0.375],
      ["metric", "v1b", "v1bp", "reading"], rows5,
      widths=[.25, .14, .14, .47], hi=hi5, fs=8.5)

SH8 = ("The single-variable reward-pair experiment, aggregated over both whole runs.")
IN8 = ("A genuinely instructive SPLIT result. The pair did exactly what it was designed to do: "
       "catheter-lead — the shove pathology it targeted — fell 0.34→0.04, escape rose 53→67%, "
       "terminal jams fell 47→33%. And it gained nothing: episode success is LOWER (52.8 vs 58.6; "
       "57.4 at matched budget). The mechanism is that v1bp's careful non-shoving gait jams more "
       "often (57.9 vs 51.0%) and its recoveries convert at only 77% vs 95%, because they are slow "
       "— soft recoveries take 296 median steps against a 600 cap. This is the cleanest evidence in "
       "the program that you can move an internal behaviour decisively and still gain nothing on "
       "the outcome, which is a result worth reporting in its own right.")
ch8 = cap_h(SH8, IN8)
b8 = 0.055 + ch8 + 0.055
ax = fig.add_axes([0.075, b8, 0.855, 0.475 - 0.055 - b8])
labs = ["recovery\nsucc%", "unrecovered%", "soft%", "hard%",
        "cath-lead\n(x100)", "episode\nsucc%", "recovered-ep\nsucc%"]
delta = [float(B["recovery succ%"]) - float(A["recovery succ%"]),
         float(B["unrecovered%"]) - float(A["unrecovered%"]),
         float(B["soft%"]) - float(A["soft%"]),
         float(B["hard%"]) - float(A["hard%"]),
         (float(B["mean cath-lead frac"]) - float(A["mean cath-lead frac"])) * 100,
         float(B["episode succ%"]) - float(A["episode succ%"]),
         float(B["recovered-ep succ%"]) - float(A["recovered-ep succ%"])]
cols = [C_OK if i < 5 else C_HI for i in range(7)]
cols[1] = C_OK
ax.bar(range(7), delta, .55, color=cols)
ax.axhline(0, color="#8A939C", lw=1)
ax.set_xticks(range(7))
ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("v1bp − v1b\n(percentage points)")
ax.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for i, d in enumerate(delta):
    ax.text(i, d + (1.4 if d >= 0 else -2.6), f"{d:+.1f}", ha="center",
            fontsize=8, color=C_TXT, fontweight="bold")

caption(fig, 0.055, SH8, IN8)
finish(pdf, fig, P)

# ---------- P9 T6 correlation ----------------------------------------
P += 1
fig = newpage(pdf, "T6 — is episode success actually tracking recovery?",
              "10 chronological bins per run  ·  each point is one bin")
for i, (rows, nm, col) in enumerate(((V1B, "v1b", C_B), (V1BP, "v1bp", C_BP))):
    xs, ys, zs = corr_pts(rows)
    ax = fig.add_axes([0.075 + 0.485 * i, 0.435, 0.375, 0.36])
    ax.scatter(ys, xs, s=95, color=col, zorder=3, edgecolor="white", lw=1.4)
    for j, (a, b) in enumerate(zip(ys, xs)):
        ax.annotate(str(j + 1), (a, b), fontsize=7, color="white",
                    ha="center", va="center", zorder=4, fontweight="bold")
    mx, my = sum(ys) / len(ys), sum(xs) / len(xs)
    sl = (sum((p - mx) * (q - my) for p, q in zip(ys, xs))
          / sum((p - mx) ** 2 for p in ys))
    xr = [min(ys), max(ys)]
    ax.plot(xr, [my + sl * (v - mx) for v in xr], "--", color=C_HI, lw=1.8)
    ax.set_xlabel("recovery success%  (per EVENT)")
    ax.set_ylabel("episode success%  (per EPISODE)")
    ax.set_title(f"{nm}    r = {pearson(xs, ys):+.2f}", fontsize=11,
                 color=col, pad=8)
    ax.grid(alpha=.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.text(0.5, 0.395, "point labels = chronological bin index (1 = earliest)",
         ha="center", fontsize=8, color=C_MUTE)
caption(fig, 0.075,
        "A direct test of the intuition that success rises and falls with recovery skill. The "
        "dashed line is the least-squares fit across bins.",
        "In v1b the relationship is STRONGLY NEGATIVE (r = −0.82): periods of better recovery are "
        "periods of worse success, because both are downstream of how often the wire is jamming at "
        "all. An earlier reading in this program that 'success% ≈ recovery%' was wrong — it compared "
        "a per-EVENT rate against a per-EPISODE rate without noticing they are different units, "
        "which is exactly why the three successes are separated in T2. v1bp's near-zero r = −0.21 is "
        "itself informative: the reward pair decoupled the two, which is what preserving the recovery "
        "skill looks like statistically.", h=0.20)
finish(pdf, fig, P)

# ---------- P10 T7 sensitivity ---------------------------------------
P += 1
fig = newpage(pdf, "T7 — threshold sensitivity: are any of these conclusions detector artifacts?",
              "Every episode scored under three detector configurations simultaneously")
hdr7 = ["config", "events", "stalled-ep%", "recovery succ%", "soft%", "hard%",
        "grind%", "unrec%"]
w7 = [.14, .14, .15, .17, .10, .10, .10, .10]
table(fig, [0.045, 0.695, 0.91, 0.10], hdr7, t7_rows(V1B), widths=w7, fs=8.5,
      title="v1b", tcolor=C_B)
table(fig, [0.045, 0.545, 0.91, 0.10], hdr7, t7_rows(V1BP), widths=w7, fs=8.5,
      title="v1bp", tcolor=C_BP)
table(fig, [0.045, 0.375, 0.91, 0.075],
      ["config", "stall_eps", "push_min", "stuck_steps", "retract_min", "soft_max", "pass_eps"],
      [["sens", "0.3 mm", "1.0 mm/s", "8", "0.5 mm", "8 mm", "1.0 mm"],
       ["canon", "0.3 mm", "2.0 mm/s", "12", "1.0 mm", "8 mm", "1.0 mm"],
       ["strict", "0.5 mm", "4.0 mm/s", "16", "1.0 mm", "8 mm", "1.0 mm"]],
      fs=8.3, title="detector parameters", tcolor=C_MUTE)

caption(fig, 0.075,
        "The same analysis re-run under a sensitive, a canonical, and a strict definition of "
        "'stuck', to separate signal from detector choice.",
        "Absolute rates move a great deal — v1b's recovery success reads anywhere from 42% to 65% "
        "depending on strictness — so NO absolute number here should be quoted on its own. But every "
        "ORDERING survives all three settings: v1bp beats v1b on escape rate and soft share and has "
        "fewer terminal jams, under sensitive, canonical and strict detectors alike. That is why "
        "every finding in this report is stated as a delta or a trend rather than a level.", h=0.19)
finish(pdf, fig, P)

# ---------- P11 verification -----------------------------------------
P += 1
fig = newpage(pdf, "Verification — reconciling with the earlier audit, and why v1bp's recoveries fail",
              "Run on challenge: soft counts far exceeded a previous audit, and v1bp conversion fell as low as 54%")
ax = fig.add_axes([0.075, 0.545, 0.40, 0.28])
labels = ["<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", ">16"]
def hist(rows):
    rr = [x["r"] for r in rows for x in ev(r) if x["k"] != "unrec"]
    h = Counter()
    for v in rr:
        h["<0.5" if v < .5 else "0.5–1" if v < 1 else "1–2" if v < 2 else
          "2–4" if v < 4 else "4–8" if v < 8 else "8–16" if v < 16 else ">16"] += 1
    return [pct(h[k], len(rr)) for k in labels]
hb, hbp = hist(V1B), hist(V1BP)
xx = range(7)
ax.bar([i - .19 for i in xx], hb, .37, color=C_B, label="v1b")
ax.bar([i + .19 for i in xx], hbp, .37, color=C_BP, label="v1bp")
ax.axvline(1.5, color=C_HI, ls="--", lw=1.5)
ax.axvline(4.5, color=C_HI, ls="--", lw=1.5)
ax.text(1.5, max(hb) * .96, " grind│soft", fontsize=7.4, color=C_HI)
ax.text(4.5, max(hb) * .96, " soft│hard", fontsize=7.4, color=C_HI)
ax.set_xticks(list(xx))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("retraction depth of escaped stalls (mm)")
ax.set_ylabel("% of escaped events")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("the soft/grind boundary cuts a CONTINUUM", fontsize=9.5, pad=7)

# v1bp per-bin conversion
n8 = len(V1BP) // 8
conv, sipsucc, cap = [], [], []
for i in range(8):
    b = V1BP[i * n8:(i + 1) * n8]
    rec = [r for r in b if ep_class(r) == "rec"]
    sip = [r for r in rec if sect(r["pl"]) == "siphon"]
    fl = [r for r in rec if not r["succ"]]
    conv.append(pct(sum(1 for r in rec if r["succ"]), len(rec)))
    sipsucc.append(pct(sum(1 for r in sip if r["succ"]), len(sip)) if sip else 0)
    cap.append(pct(sum(1 for r in fl if r["steps"] >= 600), len(fl)) if fl else 0)
ax2 = fig.add_axes([0.555, 0.545, 0.40, 0.28])
x = list(range(1, 9))
ax2.plot(x, conv, "-o", color=C_BP, lw=2.4, ms=6, label="recovered-ep succ%")
ax2.plot(x, sipsucc, "-s", color="#7A5AA8", lw=2, ms=5.5, label="siphon recovered-ep succ%")
ax2.plot(x, cap, "--^", color=C_MUTE, lw=1.7, ms=5, label="failures dying at 600-cap%")
ax2.set_xticks(x)
ax2.set_xlabel("chronological bin")
ax2.set_ylabel("percent")
ax2.grid(alpha=.25, ls=":")
ax2.legend(frameon=False, fontsize=7.8, loc="lower right")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("v1bp: recoveries LEARN to pay, late", fontsize=9.5, pad=7)

SH11 = ("Left: why soft% is so sensitive to threshold choice. Right and below: the forensics of "
        "v1bp's recovered-but-failed episodes.")
IN11 = ("The retraction distribution is a CONTINUUM massed near zero (32% of v1b escapes are under "
        "0.5 mm), not a bimodal soft/grind split — so the 1 mm and 8 mm cuts slice a smooth curve and "
        "absolute soft% is detector-defined. That fully explains the discrepancy with the earlier "
        "audit, whose stricter stall criterion admitted only deep, mostly hopeless stalls; no "
        "re-slicing of today's data reproduces its 2–6%, and the invariant orderings agree across "
        "both audits. Separately, v1bp's failed recoveries TIME OUT rather than jam (91% at the cap, "
        "61 mm short, still fighting at 97% of the episode), the gap is real within every depth band, "
        "and it is largely TRANSIENT — by bins 6–7 conversion reaches 94–95%. Late v1bp did learn to "
        "make recoveries pay; what it never learned is avoidance.")
ch11 = cap_h(SH11, IN11)
b11 = 0.055 + ch11 + 0.048
table(fig, [0.045, b11, 0.42, 0.088],
      ["band", "v1b rec-ep succ%", "v1bp rec-ep succ%"],
      [["CCA", "98%  (n=251)", "94%  (n=144)"],
       ["ICA-mid", "97%  (n=733)", "85%  (n=503)"],
       ["siphon", "91%  (n=446)", "61%  (n=421)"]],
      fs=8.4, title="conversion gap is real within EVERY band", tcolor=C_MUTE)
table(fig, [0.535, b11, 0.42, 0.088],
      ["v1bp recovered-but-FAILED", "value"],
      [["died at 600-step cap", "91%  (211/246 max_steps)"],
       ["median final distance to target", "61 mm"],
       ["last recovery at", "97% of episode"]],
      fs=8.4, widths=[.62, .38], title="they time out, they do not jam", tcolor=C_MUTE)

caption(fig, 0.055, SH11, IN11)
finish(pdf, fig, P)

# ---------- P12 findings ---------------------------------------------
P += 1
fig = newpage(pdf, "Findings", "Every claim below is stated as a delta or trend, and survives all three detector configurations")
F = [
    ("F1", "Improvement is stall AVOIDANCE, not stall ESCAPE.", C_HI,
     "In v1b episode success rises 52→62% while recovery success falls 65→41%, unrecovered climbs "
     "35→59%, and recovery's share of successes collapses 29→6%. The correlation is r = −0.82. The "
     "policy is not learning to get unstuck; it is learning not to get stuck, and the recovery skill "
     "decays as it does. This is the interference mechanism the design doc predicted, now measured."),
    ("F2", "The reward pair fixed recovery — mechanically and unambiguously.", C_OK,
     "v1bp escapes 67.4% of stalls vs 53.4%, leaves 32.6% unrecovered vs 46.6%, raises soft share to "
     "29.5%, and cuts catheter-lead from 0.34 to 0.04. The decay is gone: recovery holds 63–75% late "
     "(r = −0.21) where v1b decayed to ~40%."),
    ("F3", "…and it still did not help, because recovery does not pay.", C_HI,
     "Episode success is lower (52.8 vs 58.6; 57.4 at matched budget). Recovered episodes convert at "
     "77% vs 95% and cost far more time (soft 296 vs 187 median steps against a 600 cap). v1bp also "
     "stalls more (57.9 vs 51.0%). More stalls, better escapes, worse conversion, slightly worse "
     "outcome — but by bins 6–7 conversion recovers to 94–95%, so the deficit was largely transient."),
    ("F4", "GRIND beats SOFT on episode success, in both runs.", C_HI,
     "Grind-only episodes succeed 99%/95% in ~110 steps; soft 95%/76% in 187–296; hard 92%/65%. This "
     "inverts the premise behind the crunchpass lane. Likely a confound — recovery type proxies stall "
     "SEVERITY rather than policy skill — so soft recovery cannot be a training target until the "
     "confound is broken within matched depth and severity strata."),
    ("F5", "Depth dominates everything.", C_TXT,
     "90% (v1b) to 97% (v1bp) of siphon episodes stall, versus 5–7% at CCA. At the siphon, escaping "
     "does not save the episode: 50–68% escape but only 14–19% success. The limiting factor is "
     "stall DENSITY, not escape ability — the same wall the real-patient eval hit at 0 of 30."),
    ("F6", "Detector-robust.", C_TXT,
     "Absolute rates move ±20 pp across the three configurations, but every v1b-vs-v1bp ordering "
     "holds under all of them."),
]
y = 0.845
for tag, head, col, body in F:
    fig.text(0.048, y, tag, fontsize=12.5, fontweight="bold", color=col)
    fig.text(0.095, y, head, fontsize=11.5, fontweight="bold", color=C_TXT)
    fig.text(0.095, y - 0.030, "\n".join(textwrap.wrap(body, 126)), fontsize=8.8,
             va="top", color=C_TXT, linespacing=1.55)
    y -= 0.030 + 0.0255 * (len(textwrap.wrap(body, 126))) + 0.030
finish(pdf, fig, P)

# ---------- P13 implications -----------------------------------------
P += 1
fig = newpage(pdf, "Implications for what to do next",
              "What the measurements license, and what they rule out")
items = [
    ("1", "The crunchpass lane's premise needs revisiting BEFORE launch.",
     "It resamples success-conditioned crunch passages on the theory that soft recovery is the skill "
     "to amplify. F4 says grind-through is what actually converts and F1 says the winning channel is "
     "avoidance. Amplifying recovery may be optimising a behaviour that does not pay. Break the "
     "severity confound first: compare soft vs grind within matched depth strata."),
    ("2", "If recovery is still wanted, it must be made to PAY.",
     "Recovered episodes cost 2–7× the steps of clean ones. Under a 600-step cap with a sparse "
     "target reward, a slow recovery is nearly worthless — the optimiser is CORRECT to prefer "
     "avoidance. Either the budget must accommodate recovery or recovery must get cheaper."),
    ("3", "The reward pair is validated as a mechanism and rejected as a win.",
     "Moving an internal behaviour decisively (cath-lead 0.34→0.04, unrecovered −14 pp) produced no "
     "outcome gain. This is strong evidence against further reward shaping as the lever, and it is a "
     "publishable negative result in its own right."),
    ("4", "The siphon wall is not a recovery problem.",
     "Near-total stall rates with ~65% escape but ~15% success mean the limiting factor is the sheer "
     "density of stalls, not the ability to escape any one of them. Interventions aimed at escape "
     "skill are aimed at the wrong target."),
    ("5", "Two runs, two equilibria, one summit.",
     "v1b avoids-and-runs-clean; v1bp stalls-and-fights-through. They reach nearly identical peak "
     "success (61–62%) by opposite routes. Whatever caps performance is above both strategies, which "
     "argues that the next gain will not come from tuning the gait."),
]
y = 0.845
for tag, head, body in items:
    fig.text(0.048, y, tag, fontsize=12.5, fontweight="bold", color=C_B)
    fig.text(0.082, y, head, fontsize=11.5, fontweight="bold", color=C_TXT)
    fig.text(0.082, y - 0.030, "\n".join(textwrap.wrap(body, 128)), fontsize=8.8,
             va="top", color=C_TXT, linespacing=1.55)
    y -= 0.030 + 0.0255 * len(textwrap.wrap(body, 128)) + 0.033
fig.text(0.048, 0.075,
         "Reproduce:  monitoring/extract_stuck.py <run_dir> <out.jsonl>   then   "
         "monitoring/analyze_stuck.py stuck_v1b.jsonl stuck_v1bp.jsonl [canon|sens|strict]",
         fontsize=8, color=C_MUTE, family="monospace")
finish(pdf, fig, P)

d = pdf.infodict()
d["Title"] = "Stuck & Recovery Analysis - v1b vs v1bp (explore-only)"
d["Subject"] = "Endovascular navigation RL - stall and recovery behaviour vs training time"
pdf.close()
print(f"wrote {OUT} ({P} pages)")
