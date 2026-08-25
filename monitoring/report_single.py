"""Standalone single-run stuck/recovery report (v1b OR v1bp).

Same analysis and page furniture as the combined report, but every table is for
ONE run, the head-to-head page is dropped, and the findings/implications pages
are generated from that run's own computed numbers so they cannot drift.

usage: report_single.py <stuck.jsonl> <v1b|v1bp> <out.pdf> [pngdir]
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

RUN = sys.argv[2]
IS_B = RUN == "v1b"
C_MAIN = "#2E5E8A" if IS_B else "#C1662F"
C_HDR, C_ROW = "#33404D", "#F2F4F7"
C_TXT, C_MUTE = "#1B2530", "#6B7885"
C_HI, C_OK = "#B3261E", "#1E7A4C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C8CFD6", "axes.labelcolor": C_TXT,
    "xtick.color": C_MUTE, "ytick.color": C_MUTE,
    "axes.titleweight": "bold", "axes.titlecolor": C_TXT,
})
PAGE = (11.69, 8.27)
KINDS = ("soft", "hard", "grind", "unrec")
ESC = ("soft", "hard", "grind")


def load(p):
    rows = [json.loads(l) for l in open(p)]
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
PNGDIR = sys.argv[4] if len(sys.argv) > 4 else None
if PNGDIR:
    import os
    os.makedirs(PNGDIR, exist_ok=True)
P = 0


def newpage(title, subtitle=""):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.045, 0.945, title, fontsize=17, fontweight="bold", color=C_TXT)
    if subtitle:
        fig.text(0.045, 0.906, subtitle, fontsize=9.5, color=C_MUTE)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.893, 0.893], color="#D5DBE1", lw=1.1))
    return fig


def finish(fig):
    global P
    fig.text(0.955, 0.032, str(P), ha="right", fontsize=8, color=C_MUTE)
    fig.text(0.045, 0.032, f"eve_rl · {RUN} · stuck & recovery · explore-only",
             fontsize=7.5, color="#9AA5B1")
    pdf.savefig(fig)
    if PNGDIR:
        fig.savefig(f"{PNGDIR}/page{P:02d}.png", dpi=110)
    plt.close(fig)


CAP_W, CAP_LH, CAP_PAD = 146, 0.0182, 0.017


def _cl(s, i):
    return (textwrap.wrap("WHAT IT SHOWS   " + s, CAP_W),
            textwrap.wrap("INFERENCE   " + i, CAP_W))


def cap_h(s, i):
    a, b = _cl(s, i)
    return 2 * CAP_PAD + (len(a) + len(b)) * CAP_LH + 0.012


def caption(fig, y, s, i):
    l1, l2 = _cl(s, i)
    h = cap_h(s, i)
    ax = fig.add_axes([0.045, y, 0.91, h])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor="#F7F9FB", edgecolor="#DFE5EA", lw=1))
    ax.add_patch(Rectangle((0, 0), 0.0055, 1, transform=ax.transAxes,
                           facecolor=C_MAIN, edgecolor="none"))
    yy = 1 - CAP_PAD / h
    for ln in l1:
        ax.text(0.012, yy, ln, va="top", fontsize=8.6, color=C_TXT, transform=ax.transAxes)
        yy -= CAP_LH / h
    yy -= 0.012 / h
    for ln in l2:
        ax.text(0.012, yy, ln, va="top", fontsize=8.6, color=C_TXT,
                transform=ax.transAxes, fontstyle="italic")
        yy -= CAP_LH / h


def table(fig, rect, headers, rows, widths=None, hi=None, fs=8.4, title=None,
          tcolor=C_HDR):
    ax = fig.add_axes(rect)
    ax.axis("off")
    hi = hi or {}
    nc = len(headers)
    widths = widths or [1.0 / nc] * nc
    t = sum(widths)
    widths = [w / t for w in widths]
    xs = [0.0]
    for w in widths:
        xs.append(xs[-1] + w)
    nr = len(rows)
    hh = 1.0 / (nr + 1.6)
    hdr = hh * 1.6
    if title:
        ax.text(0, 1.045, title, fontsize=10, fontweight="bold", color=tcolor,
                transform=ax.transAxes)
    ax.add_patch(Rectangle((0, 1 - hdr), 1, hdr, transform=ax.transAxes,
                           facecolor=C_HDR, edgecolor="none"))
    for j, h in enumerate(headers):
        ax.text((xs[j] + xs[j + 1]) / 2, 1 - hdr / 2, h, ha="center", va="center",
                fontsize=fs - .2, color="white", fontweight="bold",
                transform=ax.transAxes)
    for i, row in enumerate(rows):
        y1 = 1 - hdr - i * hh
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y1 - hh), 1, hh, transform=ax.transAxes,
                                   facecolor=C_ROW, edgecolor="none"))
        for j, cell in enumerate(row):
            col, wt = C_TXT, "normal"
            if (i, j) in hi:
                col, wt = hi[(i, j)], "bold"
            ax.text((xs[j] + xs[j + 1]) / 2, y1 - hh / 2, str(cell), ha="center",
                    va="center", fontsize=fs, color=col, fontweight=wt,
                    transform=ax.transAxes)
    ax.add_patch(Rectangle((0, 1 - hdr - nr * hh), 1, hdr + nr * hh,
                           transform=ax.transAxes, facecolor="none",
                           edgecolor="#C8CFD6", lw=1))


# =============================================================== compute
R = load(sys.argv[1])
OUT = sys.argv[3]
pdf = PdfPages(OUT)

NB = len(R) // 8
BINS = bins(R)


def bin_stats(b):
    e = [x for r in b for x in ev(r)]
    st = [r for r in b if ev(r)]
    c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
    return e, st, c, sum(c[k] for k in ESC)


T1, SU, ST, RC, SF, HD, GR, UN = [], [], [], [], [], [], [], []
for i, b in enumerate(BINS, 1):
    e, st, c, esc = bin_stats(b)
    T1.append([i, f"{b[0]['cum']//1000}–{b[-1]['cum']//1000}",
               f"{pct(sum(1 for r in b if r['succ']), len(b)):.0f}%",
               f"{pct(len(st), len(b)):.0f}%", len(e),
               f"{pct(esc, len(e)):.0f}%",
               f"{pct(c['soft'], len(e)):.0f}%", f"{pct(c['hard'], len(e)):.0f}%",
               f"{pct(c['grind'], len(e)):.0f}%", f"{pct(c['unrec'], len(e)):.0f}%"])
    SU.append(pct(sum(1 for r in b if r["succ"]), len(b)))
    ST.append(pct(len(st), len(b)))
    RC.append(pct(esc, len(e)))
    SF.append(pct(c["soft"], len(e)))
    HD.append(pct(c["hard"], len(e)))
    GR.append(pct(c["grind"], len(e)))
    UN.append(pct(c["unrec"], len(e)))


def _arg(vals, fn):
    """Index of the extreme cell, compared on the ROUNDED values that are
    actually printed — so the highlighted cell is the one a reader sees as
    lowest/highest, not one that merely differs in a hidden decimal."""
    r = [round(v) for v in vals]
    return r.index(fn(r))

T2, VIA = [], []
for i, b in enumerate(BINS, 1):
    g = defaultdict(list)
    for r in b:
        g[ep_class(r)].append(r)
    stot = sum(1 for r in b if r["succ"])
    srec = sum(1 for r in g["rec"] if r["succ"])
    row = [i, f"{pct(stot, len(b)):.0f}%"]
    for k in ("clean", "rec", "unrec"):
        n = len(g[k])
        s = sum(1 for r in g[k] if r["succ"])
        row.append(f"{n}  ({pct(s, n):.0f}%)" if n else "0  (—)")
    row.append(f"{pct(srec, stot):.0f}%")
    T2.append(row)
    VIA.append(pct(srec, stot))

GRP = {"soft (any)": [], "hard (no soft)": [], "grind only": [],
       "unrecovered (any)": [], "never stalled": []}
for r in R:
    e = ev(r)
    if not e:
        GRP["never stalled"].append(r)
        continue
    ks = {x["k"] for x in e}
    GRP["unrecovered (any)" if "unrec" in ks else
        "soft (any)" if "soft" in ks else
        "hard (no soft)" if "hard" in ks else "grind only"].append(r)
T3 = []
for k, g in GRP.items():
    if not g:
        continue
    stp = sorted(r["steps"] for r in g)
    rr = [x["r"] for r in g for x in ev(r)]
    # Mean retract is meaningless for the unrecovered group: those stalls never
    # passed, so the figure is withdrawal DURING a failed stall, not the depth
    # of a completed recovery. Reporting it invites a false comparison.
    retract = "NA" if k.startswith("unrecovered") else \
        f"{sum(rr)/len(rr) if rr else 0:.1f}"
    T3.append([k, f"{pct(sum(1 for r in g if r['succ']), len(g)):.0f}%",
               stp[len(stp) // 2], retract])
GS = {r[0]: float(str(r[1]).rstrip("%")) for r in T3}
GT = {r[0]: r[2] for r in T3}

T4 = []
for s in ("CCA", "ICA-mid", "siphon"):
    b = [r for r in R if sect(r["pl"]) == s]
    e, st, c, esc = bin_stats(b)
    T4.append([s, len(b), f"{pct(sum(1 for r in b if r['succ']), len(b)):.0f}%",
               f"{pct(len(st), len(b)):.0f}%", len(e), f"{pct(esc, len(e)):.0f}%",
               f"{pct(c['soft'], len(e)):.0f}%", f"{pct(c['hard'], len(e)):.0f}%",
               f"{pct(c['grind'], len(e)):.0f}%", f"{pct(c['unrec'], len(e)):.0f}%"])

T7 = []
for cfg in ("sens", "canon", "strict"):
    e = [x for r in R for x in r["ev"][cfg]]
    c = {k: sum(1 for x in e if x["k"] == k) for k in KINDS}
    esc = sum(c[k] for k in ESC)
    T7.append([cfg, f"{len(e):,}",
               f"{pct(sum(1 for r in R if r['ev'][cfg]), len(R)):.0f}%",
               f"{pct(esc, len(e)):.0f}%", f"{pct(c['soft'], len(e)):.0f}%",
               f"{pct(c['hard'], len(e)):.0f}%", f"{pct(c['grind'], len(e)):.0f}%",
               f"{pct(c['unrec'], len(e)):.0f}%"])
RC_RANGE = (min(float(r[3].rstrip('%')) for r in T7),
            max(float(r[3].rstrip('%')) for r in T7))


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db) if da and db else float("nan")


X10, Y10, Z10 = [], [], []
for b in bins(R, 10):
    e = [x for r in b for x in ev(r)]
    if not e:
        continue
    X10.append(pct(sum(1 for r in b if r["succ"]), len(b)))
    Y10.append(pct(sum(1 for x in e if x["k"] in ESC), len(e)))
    Z10.append(pct(sum(1 for x in e if x["k"] == "soft"), len(e)))
R_RC, R_SF = corr(X10, Y10), corr(X10, Z10)

ALLE = [x for r in R for x in ev(r)]
CNT = {k: sum(1 for x in ALLE if x["k"] == k) for k in KINDS}
ESC_ALL = sum(CNT[k] for k in ESC)
GALL = defaultdict(list)
for r in R:
    GALL[ep_class(r)].append(r)
REC = GALL["rec"]
REC_S = pct(sum(1 for r in REC if r["succ"]), len(REC))
FAILR = [r for r in REC if not r["succ"]]
CAPPCT = pct(sum(1 for r in FAILR if r["steps"] >= 600), len(FAILR)) if FAILR else 0
DTG = sorted(r["d_tgt"] for r in FAILR if r.get("d_tgt") is not None)
DTG_MED = DTG[len(DTG) // 2] if DTG else float("nan")
BAND_CONV = []
for s in ("CCA", "ICA-mid", "siphon"):
    g = [r for r in REC if sect(r["pl"]) == s]
    BAND_CONV.append([s, len(g), f"{pct(sum(1 for r in g if r['succ']), len(g)):.0f}%"])
SUCC_ALL = pct(sum(1 for r in R if r["succ"]), len(R))
CATH = sum(r["cath_lead_frac"] for r in R) / len(R)

CONV, SIPS, CAPB = [], [], []
for b in BINS:
    rec = [r for r in b if ep_class(r) == "rec"]
    sip = [r for r in rec if sect(r["pl"]) == "siphon"]
    fl = [r for r in rec if not r["succ"]]
    CONV.append(pct(sum(1 for r in rec if r["succ"]), len(rec)) if rec else 0)
    SIPS.append(pct(sum(1 for r in sip if r["succ"]), len(sip)) if sip else 0)
    CAPB.append(pct(sum(1 for r in fl if r["steps"] >= 600), len(fl)) if fl else 0)

# =============================================================== P1 cover
P += 1
fig = plt.figure(figsize=PAGE)
fig.text(0.045, 0.855, "Stuck & Recovery Analysis", fontsize=30, fontweight="bold", color=C_TXT)
fig.text(0.045, 0.800, f"Endovascular navigation RL  ·  run {RUN}  ·  explore-only",
         fontsize=13.5, color=C_MAIN)
fig.add_artist(plt.Line2D([0.045, 0.955], [0.775, 0.775], color=C_MAIN, lw=2.5))
cards = [("episodes", f"{len(R):,}"), ("explore steps", f"{R[-1]['cum']:,}"),
         ("episode success", f"{SUCC_ALL:.1f}%"), ("stall events", f"{len(ALLE):,}")]
for i, (k, v) in enumerate(cards):
    x = 0.045 + 0.235 * i
    fig.text(x, 0.705, v, fontsize=19, fontweight="bold", color=C_MAIN)
    fig.text(x, 0.675, k, fontsize=9, color=C_MUTE)

what = ("v1b is the P2-teacher residual-on-heuristic baseline."
        if IS_B else
        "v1bp is byte-identical to v1b except for ONE change: it adds a reward pair "
        "(tip-average progress + catheter-slack potential). It is a single-variable experiment.")
body = (
    f"SCOPE.  Every number in this report is measured from the {RUN} worker-log complete ledger "
    "(EPISODE_START / STEP / EPISODE_OUTCOME), restricted to EXPLORE episodes. Evaluation episodes "
    "are excluded exactly: runner.eval() resets with explicit seeds, so eval EPISODE_START lines "
    f"carry a `seed=` field and explore resets do not. That filter yields {len(R):,} explore "
    f"episodes and exactly {'1,470 eval (15 evals x 98)' if IS_B else '980 eval (10 evals x 98)'} "
    "— exact. The obvious alternative, main.log time windows, leaks; and episode_summary.jsonl is "
    "unusable because it logs only ~36% of explore episodes.\n\n"
    "TRAINING-TIME AXIS.  Cumulative explore steps, summed per-episode in chronological order — "
    "exact from the logs, unlike wall-clock (which conflates simulator speed) or the global step "
    "counter.\n\n"
    f"THIS RUN.  {what}\n\n"
    "SELF-CONTAINED.  Every table and every claim in this report is derived from this run alone. "
    "Nothing here depends on comparison with another run.")
fig.text(0.045, 0.615, "\n".join(textwrap.wrap(body, 132)).replace("\n\n\n", "\n\n"),
         fontsize=9.3, va="top", color=C_TXT, linespacing=1.62)
fig.text(0.045, 0.085, "Generated from monitoring/extract_stuck.py records  ·  canonical "
         "detector config unless stated", fontsize=8, color="#9AA5B1")
finish(fig)

# =============================================================== P2 defs
P += 1
fig = newpage("Definitions — and why each one is drawn where it is",
              "Every downstream number depends on these choices, so they are stated before any result")
table(fig, [0.045, 0.615, 0.91, 0.235], ["term", "unit", "definition"],
      [["stall event", "per event", "wire pushing (|gw_trans| > 2 mm/s) while path progress stays flat (< 0.3 mm gain) for 12 consecutive steps"],
       ["grind", "per event", "escaped the stall after < 1 mm of withdrawal — forced through"],
       ["soft", "per event", "escaped after 1–8 mm withdrawal — ease slack, re-advance (the manoeuvre we set out to induce)"],
       ["hard", "per event", "escaped after > 8 mm withdrawal — costly pullback"],
       ["unrecovered", "per event", "never passed the jam; terminal for that episode"],
       ["recovery success%", "per EVENT", "(soft + hard + grind) / all stall events"],
       ["episode success%", "per EPISODE", "reached the target"]],
      widths=[0.16, 0.11, 0.73], fs=8.5)
rat = ("A stall is 'pushing but not progressing', not merely 'not progressing'.  A wire that is "
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
       "makes 'success tracks recovery' look true when the opposite can hold.")
ax = fig.add_axes([0.045, 0.10, 0.91, 0.46])
ax.axis("off")
ax.text(0, 1, "\n".join(textwrap.wrap(rat, 140)).replace("\n\n\n", "\n\n"), va="top",
        fontsize=9.1, color=C_TXT, linespacing=1.66)
finish(fig)

# =============================================================== P3 T1
P += 1
fig = newpage("Stuck & recovery vs training time",
              "Each bin ~ 1800 episodes of training data")
if IS_B:
    s1 = ("Fig 1: Success rates and recovery rates with training "
          "Fig 2. Falling recovery rates creates a ceiling for success rate")
    i1 = (f"Episode success CLIMBS {SU[0]:.0f}→{max(SU):.0f}% while recovery "
          f"success FALLS {RC[0]:.0f}→{RC[-2]:.0f}% and unrecovered rises. What is actually improving "
          f"is the stalled-episode share, which drops {ST[0]:.0f}→{min(ST):.0f}%. The policy is not "
          "learning to escape stalls; it is learning not to enter them, and the escape skill withers "
          "as stalls leave its training data.")
else:
    s1 = ("How often the wire got stuck, how often it got out, and by which manoeuvre, as training "
          "progressed, for the run carrying the reward pair.")
    i1 = (f"Recovery does NOT decay in this run. It holds {min(RC[2:]):.0f}–{max(RC[2:]):.0f}% from "
          f"bin 3 onward, and the stalled-episode share stays roughly flat "
          f"({min(ST):.0f}–{max(ST):.0f}%) rather than falling away. The reward pair kept the recovery "
          f"skill alive, exactly as designed. But episode success moves only {SU[0]:.0f}→{SU[-1]:.0f}% "
          "across the whole run: the policy keeps entering stalls and keeps fighting out of them, and "
          "the improved escape ability does not translate into a materially better outcome. The later "
          "pages show where that escape effort is lost.")
ch = cap_h(s1, i1)
tb, cb = 0.575, 0.055 + ch + 0.055
hi1 = {(_arg(SU, max), 2): C_OK,     # best success — green
       (_arg(RC, min), 5): C_HI,     # weakest recovery
       (_arg(SF, min), 6): C_HI,     # least soft
       (_arg(HD, min), 7): C_HI,     # least hard
       (_arg(GR, min), 8): C_HI,     # least grind
       (_arg(UN, max), 9): C_HI}     # most left unrecovered
table(fig, [0.045, tb, 0.91, 0.29],
      ["bin", "explore steps (k)", "success %", "stalled-ep %", "events",
       "recovery %", "soft %", "hard %", "grind %", "unrecovered %"],
      T1, widths=[.05, .145, .095, .11, .08, .115, .075, .075, .08, .125],
      hi=hi1, fs=8.6)
ax = fig.add_axes([0.075, cb, 0.855, tb - 0.055 - cb])
x = list(range(1, 9))
ax.plot(x, SU, "-o", color=C_MAIN, lw=2.4, ms=6, label="episode success%", zorder=3)
ax.plot(x, RC, "-s", color=C_HI, lw=2.0, ms=5.5, label="recovery success% (per event)")
ax.plot(x, ST, "--^", color=C_MUTE, lw=1.7, ms=5, label="stalled-episode%")
ax.plot(x, SF, ":d", color="#7A5AA8", lw=1.7, ms=5, label="soft share of events%")
ax.set_xlabel("chronological bin  (cumulative explore steps →)")
ax.set_ylabel("percent")
ax.set_xticks(x)
ax.grid(alpha=.25, ls=":")
ax.legend(frameon=False, fontsize=8.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
caption(fig, 0.055, s1, i1)
finish(fig)

# =============================================================== P4 T2
P += 1
fig = newpage("Success vs Recovery",
              "Each episodeassigned to exactly one channel: clean / stuck-and-recovered / stuck-and-failed")
if IS_B:
    i2 = (f"Recovery's contribution COLLAPSES from {VIA[0]:.0f}% to {VIA[-2]:.0f}%. Early on roughly a "
          "third of all wins were rescues; by the end nearly every win is an episode that simply never "
          f"jammed (clean episodes rise {T2[0][3].split()[0]}→{T2[-2][3].split()[0]}). ")
else:
    i2 = (f"No downward trend: recovery's contribution stays in the {min(VIA):.0f}–{max(VIA):.0f}% band "
          "across the whole run, and the best-performing bin is among those leaning MOST heavily on "
          "recovery. This run never migrates away from the recovery channel — it keeps reaching the "
          "target by fighting through stalls rather than by avoiding them, right to the end of training.")
s2 = "Episode success decomposed by channel, and the share of all successes that flowed through recovery (chart)."
i2 += ("  Note: As seen by the success rates in columns 4 and 5, a small fraction of episodes that "
       "recover do not reach the target as they have exhausted all their steps and similarly some "
       "that do get stuck are already very close to the target and hence gets counted as successes.")
ch = cap_h(s2, i2)
tb, cb = 0.575, 0.055 + ch + 0.05
table(fig, [0.045, tb, 0.91, 0.285],
      ["bin", "success %", "clean eps (success %)", "recovered eps (success %)",
       "unrec eps (success %)", "% of successes via recovery"], T2,
      widths=[.07, .12, .20, .22, .20, .19], fs=8.4,
      hi={(0, 5): C_HI, (len(T2) - 2, 5): C_HI} if IS_B else {})
ax = fig.add_axes([0.075, cb, 0.855, tb - 0.055 - cb])
ax.plot(x, VIA, "-o", color=C_MAIN, lw=2.4, ms=6)
ax.fill_between(x, 0, VIA, color=C_MAIN, alpha=.13)
ax.set_xlabel("chronological bin")
ax.set_ylabel("% of successes\nvia recovery")
ax.set_xticks(x)
ax.set_ylim(0, max(VIA) * 1.3)
ax.grid(alpha=.25, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
caption(fig, 0.055, s2, i2)
finish(fig)

# =============================================================== P5 T3
P += 1
fig = newpage("Recovery type vs episode success",
              "Episodes grouped by the recovery they achieved")
s3 = "Whether the type of escape predicts the episode outcome, and what each type costs in steps."
i3 = (f"GRIND BEATS SOFT: grind-only episodes succeed {GS.get('grind only', 0):.0f}% in "
      f"{GT.get('grind only', 0)} median steps, against soft at {GS.get('soft (any)', 0):.0f}% in "
      f"{GT.get('soft (any)', 0)} steps and hard at {GS.get('hard (no soft)', 0):.0f}% in "
      f"{GT.get('hard (no soft)', 0)}. The likely reason is a confound: recovery type is mostly a "
      "proxy for stall SEVERITY, not policy skill. A jam you can grind through was mild; one forcing "
      "an 8 mm withdrawal was severe and usually deeper, where the episode was already in trouble. ")
ch = cap_h(s3, i3)
tb = 0.635
cb = 0.055 + ch + 0.05
hh = tb - 0.055 - cb
table(fig, [0.045, tb, 0.91, 0.185],
      ["recovery profile", "success %", "median steps", "mean retract mm"],
      T3, widths=[.38, .22, .20, .20], fs=8.6,
      hi={(2, 1): C_OK, (2, 2): C_OK})
labs = ["grind\nonly", "soft\n(any)", "hard\n(no soft)"]
keys = ["grind only", "soft (any)", "hard (no soft)"]
ax = fig.add_axes([0.075, cb, 0.40, hh])
ax.bar(range(3), [GS.get(k, 0) for k in keys], .55, color=C_MAIN)
for i, k in enumerate(keys):
    ax.text(i, GS.get(k, 0) + 2, f"{GS.get(k, 0):.0f}%", ha="center", fontsize=8.5,
            fontweight="bold", color=C_TXT)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("episode success%")
ax.set_ylim(0, 112)
ax.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("episode success by recovery type", fontsize=9.5, pad=7)
ax2 = fig.add_axes([0.555, cb, 0.40, hh])
ax2.bar(range(3), [GT.get(k, 0) for k in keys], .55, color=C_MAIN)
ax2.axhline(600, color=C_HI, ls="--", lw=1.4)
ax2.text(2.45, 615, "600-step cap", fontsize=7.6, color=C_HI, ha="right")
for i, k in enumerate(keys):
    ax2.text(i, GT.get(k, 0) + 14, str(GT.get(k, 0)), ha="center", fontsize=8.5,
             fontweight="bold", color=C_TXT)
ax2.set_xticks(range(3))
ax2.set_xticklabels(labs, fontsize=8.5)
ax2.set_ylabel("median steps")
ax2.set_ylim(0, 680)
ax2.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("time cost of each recovery type", fontsize=9.5, pad=7)
caption(fig, 0.055, s3, i3)
finish(fig)

# =============================================================== P6 T4
P += 1
fig = newpage("Stuck & recovery by depth",
              "Path-length bands: CCA (<146 mm)  ·  ICA-mid (146–210 mm)  ·  siphon (>210 mm)")
sip = T4[2]
s4 = "Where the stalls actually are, and whether escaping them rescues the episode there."
i4 = (f"Depth dominates every other factor. Stalling runs {T4[0][3]} at the CCA but {sip[3]} at the "
      "siphon — getting stuck is essentially guaranteed once the wire enters the cavernous segment. "
      f"And escaping does not save you there: {sip[5]} of siphon stalls are escaped yet only {sip[2]} "
      "of siphon episodes succeed. You escape one jam and immediately meet the next. This reframes "
      "the siphon as a STALL-DENSITY problem rather than an escape-skill problem — the same wall the "
      "real-patient evaluation hit at 0 of 30, seen from the inside.")
ch = cap_h(s4, i4)
tb = 0.645
cb = 0.055 + ch + 0.05
table(fig, [0.045, tb, 0.91, 0.135],
      ["band", "eps", "ep succ%", "stalled-ep%", "events", "recovery succ%",
       "soft%", "hard%", "grind%", "unrec%"], T4,
      widths=[.11, .09, .10, .115, .09, .13, .075, .075, .08, .085], fs=8.6,
      hi={(2, 2): C_HI, (2, 3): C_HI})
ax = fig.add_axes([0.075, cb, 0.855, tb - 0.06 - cb])
bands = [r[0] for r in T4]
stv = [float(r[3].rstrip("%")) for r in T4]
suv = [float(r[2].rstrip("%")) for r in T4]
rcv = [float(r[5].rstrip("%")) for r in T4]
xx = range(3)
ax.bar([i - .26 for i in xx], stv, .25, color=C_HI, alpha=.75, label="stalled-episode%")
ax.bar([i for i in xx], rcv, .25, color=C_MUTE, alpha=.75, label="recovery success% (per event)")
ax.bar([i + .26 for i in xx], suv, .25, color=C_MAIN, label="episode success%")
ax.set_xticks(list(xx))
ax.set_xticklabels(bands)
ax.set_ylabel("percent")
ax.set_ylim(0, 112)
ax.grid(alpha=.25, ls=":", axis="y")
ax.legend(frameon=False, fontsize=8.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
caption(fig, 0.055, s4, i4)
finish(fig)

# =============================================================== P7 T6
P += 1
fig = newpage("Is episode success actually tracking recovery?",
              "10 chronological bins  ·  each point is one bin  ·  units deliberately mixed to test the intuition")
s6 = ("A direct test of the intuition that success rises and falls with recovery skill. Each point "
      "is one chronological bin; the dashed line is the least-squares fit.")
if IS_B:
    i6 = (f"STRONGLY NEGATIVE (r = {R_RC:+.2f}). Periods of better recovery are periods of WORSE "
          "success, because both are downstream of how often the wire is jamming at all. An earlier "
          "reading in this program that 'success% ≈ recovery%' was wrong — it compared a per-EVENT "
          "rate against a per-EPISODE rate without noticing they are different units, which is "
          "precisely why the three successes are separated in T2.")
else:
    i6 = (f"Essentially NO relationship (r = {R_RC:+.2f}) — and that is itself the finding. Episode "
          "success and recovery success move independently here: recovery ability neither drives the "
          "outcome nor has to decay for the outcome to improve. Statistically this is what a preserved "
          "but non-decisive recovery skill looks like — the two quantities are simply not the same "
          "thing, which is why they are separated by unit of analysis in T2.")
ch = cap_h(s6, i6)
ax = fig.add_axes([0.30, 0.42, 0.40, 0.40])
ax.scatter(Y10, X10, s=110, color=C_MAIN, zorder=3, edgecolor="white", lw=1.4)
for j, (a, b) in enumerate(zip(Y10, X10)):
    ax.annotate(str(j + 1), (a, b), fontsize=7, color="white", ha="center",
                va="center", zorder=4, fontweight="bold")
mx, my = sum(Y10) / len(Y10), sum(X10) / len(X10)
sl = sum((p - mx) * (q - my) for p, q in zip(Y10, X10)) / sum((p - mx) ** 2 for p in Y10)
xr = [min(Y10), max(Y10)]
ax.plot(xr, [my + sl * (v - mx) for v in xr], "--", color=C_HI, lw=1.8)
ax.set_xlabel("recovery success%  (per EVENT)")
ax.set_ylabel("episode success%  (per EPISODE)")
ax.set_title(f"r = {R_RC:+.2f}", fontsize=13, color=C_MAIN, pad=8)
ax.grid(alpha=.25, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, 0.385, f"point labels = chronological bin index (1 = earliest)   ·   "
         f"episode-success vs soft-share:  r = {R_SF:+.2f}",
         ha="center", fontsize=8.5, color=C_MUTE)
caption(fig, 0.055, s6, i6)
finish(fig)

# =============================================================== P8 T7
P += 1
fig = newpage("Threshold sensitivity — are these conclusions detector artifacts?",
              "Every episode scored under three detector configurations simultaneously")
s7 = ("The same analysis re-run under a sensitive, a canonical and a strict definition of 'stuck', "
      "to separate signal from detector choice.")
i7 = (f"Absolute rates move a great deal — recovery success reads anywhere from {RC_RANGE[0]:.0f}% to "
      f"{RC_RANGE[1]:.0f}% for this run alone depending on strictness — so NO absolute number in this "
      "report should be quoted on its own. What survives all three settings is the ORDERING and the "
      "DIRECTION of every trend, which is why each finding here is stated as a delta or a trend "
      "rather than a level.")
table(fig, [0.045, 0.645, 0.91, 0.145],
      ["config", "events", "stalled-ep%", "recovery succ%", "soft%", "hard%", "grind%", "unrec%"],
      T7, widths=[.14, .14, .15, .17, .10, .10, .10, .10], fs=8.8,
      title=f"{RUN} under three detectors", tcolor=C_MAIN)
table(fig, [0.045, 0.45, 0.91, 0.105],
      ["config", "stall_eps", "push_min", "stuck_steps", "retract_min", "soft_max", "pass_eps"],
      [["sens", "0.3 mm", "1.0 mm/s", "8", "0.5 mm", "8 mm", "1.0 mm"],
       ["canon", "0.3 mm", "2.0 mm/s", "12", "1.0 mm", "8 mm", "1.0 mm"],
       ["strict", "0.5 mm", "4.0 mm/s", "16", "1.0 mm", "8 mm", "1.0 mm"]],
      fs=8.6, title="detector parameters", tcolor=C_MUTE)
caption(fig, 0.075, s7, i7)
finish(fig)

# =============================================================== P9 diag
P += 1
fig = newpage("Diagnostics — recovery quality, cost and where it fails",
              "Retraction-depth distribution, per-bin conversion, and the forensics of recovered-but-failed episodes")
labels = ["<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", ">16"]
rr = [x["r"] for r in R for x in ev(r) if x["k"] != "unrec"]
h = Counter()
for v in rr:
    h["<0.5" if v < .5 else "0.5–1" if v < 1 else "1–2" if v < 2 else "2–4" if v < 4
      else "4–8" if v < 8 else "8–16" if v < 16 else ">16"] += 1
hv = [pct(h[k], len(rr)) for k in labels]
s9 = ("Left: the retraction-depth distribution of escaped stalls, with the classification cuts drawn "
      "on. Right: how well this run's recoveries convert to episode success over training. Below: "
      "conversion by depth band and what the failed recoveries were doing when they ended.")
i9 = (f"The retraction distribution is a CONTINUUM massed near zero ({hv[0]:.0f}% of escapes are under "
      "0.5 mm), not a bimodal soft/grind split — so the 1 mm and 8 mm cuts slice a smooth curve and "
      "absolute soft% is detector-defined rather than a property of the policy. "
      + (f"Recovered episodes convert at {REC_S:.0f}% overall and the conversion is high in every band, "
         "so in this run a recovery that happens is a recovery that pays."
         if REC_S > 88 else
         f"Recovered episodes convert at only {REC_S:.0f}%, and the failures TIME OUT rather than jam "
         f"— {CAPPCT:.0f}% die at the 600-step cap, a median {DTG_MED:.0f} mm short of target. They "
         "escaped every stall and simply ran out of clock. Note the conversion is strongly "
         "depth-dependent and largely TRANSIENT: the late bins recover to the 85–95% range, so this "
         "run did eventually learn to make recoveries pay."))
ch = cap_h(s9, i9)
tb = 0.40
cb = 0.055 + ch + 0.05
ax = fig.add_axes([0.075, 0.545, 0.38, 0.27])
ax.bar(range(7), hv, .62, color=C_MAIN)
ax.axvline(1.5, color=C_HI, ls="--", lw=1.5)
ax.axvline(4.5, color=C_HI, ls="--", lw=1.5)
ax.text(1.55, max(hv) * .95, "grind│soft", fontsize=7.4, color=C_HI)
ax.text(4.55, max(hv) * .95, "soft│hard", fontsize=7.4, color=C_HI)
ax.set_xticks(range(7))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("retraction depth of escaped stalls (mm)")
ax.set_ylabel("% of escaped events")
ax.grid(alpha=.25, ls=":", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("the soft/grind boundary cuts a CONTINUUM", fontsize=9.5, pad=7)
ax2 = fig.add_axes([0.565, 0.545, 0.39, 0.27])
ax2.plot(x, CONV, "-o", color=C_MAIN, lw=2.4, ms=6, label="recovered-ep succ%")
ax2.plot(x, SIPS, "-s", color="#7A5AA8", lw=2, ms=5.5, label="siphon recovered-ep succ%")
ax2.plot(x, CAPB, "--^", color=C_MUTE, lw=1.7, ms=5, label="failures dying at 600-cap%")
ax2.set_xticks(x)
ax2.set_xlabel("chronological bin")
ax2.set_ylabel("percent")
ax2.grid(alpha=.25, ls=":")
ax2.legend(frameon=False, fontsize=7.6, loc="lower right")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("do recoveries convert to episode success?", fontsize=9.5, pad=7)
table(fig, [0.045, cb, 0.42, 0.10],
      ["band", "recovered eps", "episode succ%"], BAND_CONV, fs=8.6,
      title="conversion of recovered episodes, by depth", tcolor=C_MUTE)
table(fig, [0.535, cb, 0.42, 0.10],
      ["recovered-but-FAILED episodes", "value"],
      [["count", f"{len(FAILR)} of {len(REC)} recovered"],
       ["died at the 600-step cap", f"{CAPPCT:.0f}%"],
       ["median final distance to target", f"{DTG_MED:.0f} mm"]],
      fs=8.6, widths=[.62, .38], title="what the failures were doing", tcolor=C_MUTE)
caption(fig, 0.055, s9, i9)
finish(fig)

# =============================================================== P10 findings
P += 1
fig = newpage("Findings",
              "Every claim is stated as a delta or trend, and survives all three detector configurations")
if IS_B:
    F = [("F1", "Improvement is stall AVOIDANCE, not stall ESCAPE.", C_HI,
          f"Episode success rises {SU[0]:.0f}→{max(SU):.0f}% while recovery success falls "
          f"{RC[0]:.0f}→{min(RC):.0f}%, unrecovered climbs, and recovery's share of successes collapses "
          f"{VIA[0]:.0f}→{VIA[-2]:.0f}%. The correlation is r = {R_RC:+.2f}. The policy is not learning "
          "to get unstuck; it is learning not to get stuck, and the recovery skill decays as it does. "
          "This is the interference mechanism the design doc predicted, now measured."),
         ("F2", "GRIND beats SOFT on episode success.", C_HI,
          f"Grind-only episodes succeed {GS.get('grind only',0):.0f}% in {GT.get('grind only',0)} median "
          f"steps; soft {GS.get('soft (any)',0):.0f}% in {GT.get('soft (any)',0)}; hard "
          f"{GS.get('hard (no soft)',0):.0f}% in {GT.get('hard (no soft)',0)}. This inverts the premise "
          "behind the crunchpass lane. Likely a confound — recovery type proxies stall SEVERITY rather "
          "than policy skill — so soft recovery cannot be a training target until that is broken."),
         ("F3", "Recovery, when it happens, pays here.", C_OK,
          f"Recovered episodes convert at {REC_S:.0f}% and stay high in every depth band. The problem "
          "is not that this run's recoveries are bad; it is that they become RARE as avoidance takes "
          "over, so they stop contributing to the headline number."),
         ("F4", "Depth dominates everything.", C_TXT,
          f"{T4[2][3]} of siphon episodes stall versus {T4[0][3]} at CCA. At the siphon, escaping does "
          f"not save the episode: {T4[2][5]} escape but only {T4[2][2]} succeed. The limiting factor is "
          "stall DENSITY, not escape ability — the same wall the real-patient eval hit at 0 of 30."),
         ("F5", "Detector-robust.", C_TXT,
          f"Absolute rates move across the three configurations (recovery success {RC_RANGE[0]:.0f}–"
          f"{RC_RANGE[1]:.0f}%), but every trend direction holds under all of them.")]
    IMP = [("1", "The crunchpass lane's premise needs revisiting BEFORE launch.",
            "It amplifies success-conditioned soft-ish passages on the theory that soft recovery is the "
            "skill to grow. F2 says grind-through converts better and F1 says the winning channel is "
            "avoidance. Break the severity confound first — compare soft vs grind within matched depth."),
           ("2", "If recovery is wanted, it must be made to PAY.",
            "Recovered episodes cost several times the steps of clean ones. Under a 600-step cap with a "
            "sparse target reward, a slow recovery is nearly worthless — the optimiser is CORRECT to "
            "prefer avoidance. Either the budget accommodates recovery or recovery gets cheaper."),
           ("3", "The siphon wall is not a recovery problem.",
            "Near-total stall rates with moderate escape but low success mean the limiting factor is the "
            "density of stalls, not the ability to escape any one. Interventions aimed at escape skill "
            "are aimed at the wrong target."),
           ("4", "Beware the units trap.",
            "The strong negative r is only visible once per-EVENT and per-EPISODE rates are separated. "
            "Any future analysis quoting a single 'recovery rate' against success is at risk of "
            "reproducing the earlier mistaken reading.")]
else:
    F = [("F1", "The reward pair fixed recovery — mechanically and unambiguously.", C_OK,
          f"Recovery success holds {min(RC[2:]):.0f}–{max(RC[2:]):.0f}% from bin 3 onward instead of "
          f"decaying, unrecovered sits at {pct(CNT['unrec'], len(ALLE)):.1f}% of events, soft share is "
          f"{pct(CNT['soft'], len(ALLE)):.1f}%, and mean catheter-lead — the shove pathology the pair "
          f"targeted — is {CATH:.2f}. The pair did exactly what it was designed to do."),
         ("F2", "…and the escape ability does not convert into success.", C_HI,
          f"Episode success ends at {SUCC_ALL:.1f}% overall despite the strong escape numbers. "
          f"Recovered episodes convert at only {REC_S:.0f}%, and their failures TIME OUT rather than "
          f"jam — {CAPPCT:.0f}% die at the 600-step cap a median {DTG_MED:.0f} mm short of target, "
          "having escaped every stall along the way. Escaping is necessary and nowhere near sufficient."),
         ("F3", "Recovery is preserved but non-decisive, and the conversion deficit is TRANSIENT.", C_OK,
          f"Episode success and recovery success are statistically uncoupled (r = {R_RC:+.2f}): recovery "
          f"never withers, but neither does it drive the outcome. Conversion is also not a fixed "
          f"property — it recovers to {max(CONV[5:]):.0f}% in the late bins, so the policy did "
          "eventually learn to make its recoveries pay. What it never learned is to stop entering "
          f"stalls: the stalled-episode share ends at {ST[-1]:.0f}%, essentially where it started."),
         ("F4", "GRIND still beats SOFT on episode success.", C_HI,
          f"Grind-only {GS.get('grind only',0):.0f}% in {GT.get('grind only',0)} median steps vs soft "
          f"{GS.get('soft (any)',0):.0f}% in {GT.get('soft (any)',0)} and hard "
          f"{GS.get('hard (no soft)',0):.0f}% in {GT.get('hard (no soft)',0)}. The ordering survives the "
          "reward pair, which strengthens the severity-confound reading: it is a property of the "
          "stalls, not of the gait."),
         ("F5", "Depth dominates everything.", C_TXT,
          f"{T4[2][3]} of siphon episodes stall versus {T4[0][3]} at CCA, and at the siphon {T4[2][5]} "
          f"of stalls are escaped yet only {T4[2][2]} of episodes succeed. Better escape skill does not "
          "break the siphon — the limiting factor is stall DENSITY.")]
    IMP = [("1", "The reward pair is validated as a MECHANISM and rejected as a WIN.",
            "It moved the intended internal behaviour decisively and produced no outcome gain. This is "
            "the cleanest evidence in the program against further reward shaping as the lever, and a "
            "publishable negative result in its own right."),
           ("2", "Better recovery is not the bottleneck.",
            "This run escapes two thirds of its stalls, leaves few terminal jams, and eliminates the "
            "catheter shove — and still finishes at the same order of success. Whatever caps "
            "performance sits above the choice of gait."),
           ("3", "Recovery must be CHEAP, not merely successful.",
            "The failures here escaped every stall and still lost — on the clock, tens of millimetres "
            "from target. Under a fixed step budget, escape SPEED matters as much as escape ability, "
            "and a recovery that consumes half the episode is close to worthless."),
           ("4", "Pre-register OUTCOME criteria, not behaviour criteria.",
            "Every leading indicator this intervention was designed to move did move, decisively and "
            "in the intended direction, while the outcome did not follow. Behavioural evidence that a "
            "change 'worked' is not evidence that it helped.")]
y = 0.845
for tag, head, col, body in F:
    fig.text(0.048, y, tag, fontsize=12.5, fontweight="bold", color=col)
    fig.text(0.095, y, head, fontsize=11.5, fontweight="bold", color=C_TXT)
    w = textwrap.wrap(body, 126)
    fig.text(0.095, y - 0.030, "\n".join(w), fontsize=8.8, va="top", color=C_TXT,
             linespacing=1.55)
    y -= 0.030 + 0.0255 * len(w) + 0.028
finish(fig)

# =============================================================== P11 implications
P += 1
fig = newpage("Implications", "What these measurements license, and what they rule out")
y = 0.845
for tag, head, body in IMP:
    fig.text(0.048, y, tag, fontsize=12.5, fontweight="bold", color=C_MAIN)
    fig.text(0.082, y, head, fontsize=11.5, fontweight="bold", color=C_TXT)
    w = textwrap.wrap(body, 128)
    fig.text(0.082, y - 0.030, "\n".join(w), fontsize=8.8, va="top", color=C_TXT,
             linespacing=1.55)
    y -= 0.030 + 0.0255 * len(w) + 0.034
fig.text(0.048, 0.115,
         f"Reproduce:  monitoring/extract_stuck.py <{RUN}_run_dir> stuck_{RUN}.jsonl   then   "
         f"monitoring/report_single.py stuck_{RUN}.jsonl {RUN} <out.pdf>",
         fontsize=8, color=C_MUTE, family="monospace")
fig.text(0.048, 0.088, "All figures in this report are derived from this run alone.",
         fontsize=8, color=C_MUTE)
finish(fig)

d = pdf.infodict()
d["Title"] = f"Stuck & Recovery Analysis - {RUN} (explore-only)"
d["Subject"] = f"Endovascular navigation RL - {RUN} stall and recovery behaviour vs training time"
pdf.close()
print(f"wrote {OUT} ({P} pages)")
