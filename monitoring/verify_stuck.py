"""Re-verification pass.

Q1: reconcile today's soft%=22.6% (v1b, per-EVENT) with the 2026-07-19 audit's
    "Soft% FLAT 2-6% across 5200 explore eps". Candidate explanations tested:
      a) per-EVENT share vs per-EPISODE share (episodes whose BEST recovery was soft)
      b) per-EPISODE share (episodes containing ANY soft event)
      c) the old audit only saw the FIRST ~5930 episodes (run was mid-flight July 19)
      d) detector differences (report strict config too)
Q2: v1bp recovered-episodes convert at 54-95% per bin vs v1b's 91-99% — what kills
    the recovered-but-FAILED episodes? reason / steps / depth / time-after-last-recovery.
Q3: independent recount of the headline T5 numbers (different code path).
"""
import json, sys
from collections import Counter, defaultdict

def load(p):
    rows = [json.loads(l) for l in open(p)]
    rows.sort(key=lambda r: r["t"])
    c = 0
    for r in rows:
        c += r["steps"]; r["cum"] = c
    return rows

def sect(pl): return "CCA" if pl < 146 else ("ICA-mid" if pl < 210 else "siphon")

def ep_class(r, cfg="canon"):
    e = r["ev"][cfg]
    if not e: return "clean"
    if any(x["k"] == "unrec" for x in e): return "unrec"
    return "rec"

def pct(a, b): return 100.0 * a / b if b else float("nan")

v1b = load(sys.argv[1]); v1bp = load(sys.argv[2])

# ---------- Q1: which definition reproduces "2-6% flat"? (v1b) ----------
print("## Q1 — v1b soft%% under candidate definitions, 8 chronological bins")
print("| bin | soft/events (T1 col) | eps-with-ANY-soft/eps | eps-BEST-soft/eps (T3 class) | soft events per 400 eps | [strict] soft/events |")
n8 = len(v1b) // 8
for i in range(8):
    b = v1b[i*n8:(i+1)*n8]
    e = [x for r in b for x in r["ev"]["canon"]]
    es = [x for r in b for x in r["ev"]["strict"]]
    nsoft = sum(1 for x in e if x["k"] == "soft")
    anysoft = sum(1 for r in b if any(x["k"] == "soft" for x in r["ev"]["canon"]))
    bestsoft = sum(1 for r in b if ep_class(r) == "rec"
                   and "soft" in {x["k"] for x in r["ev"]["canon"]})
    print(f"| {i+1} | {pct(nsoft,len(e)):.1f}% | {pct(anysoft,len(b)):.1f}% | "
          f"{pct(bestsoft,len(b)):.1f}% | {400.0*nsoft/len(b):.0f} | "
          f"{pct(sum(1 for x in es if x['k']=='soft'),len(es)):.1f}% |")

w = v1b[:5930]   # what the July-19 audit could see
e = [x for r in w for x in r["ev"]["canon"]]
print(f"\nfirst-5930-episode window (the July-19 audit's whole universe):")
print(f"  events={len(e)}  soft/events={pct(sum(1 for x in e if x['k']=='soft'),len(e)):.1f}%  "
      f"eps-with-any-soft={pct(sum(1 for r in w if any(x['k']=='soft' for x in r['ev']['canon'])),len(w)):.1f}%  "
      f"eps-BEST-soft={pct(sum(1 for r in w if ep_class(r)=='rec' and 'soft' in {x['k'] for x in r['ev']['canon']}),len(w)):.1f}%")

# retract-depth histogram: is 1mm cutting a continuum or a real mode?
print("\nretract-depth histogram of ESCAPED events (mm):")
for name, rows in (("v1b", v1b), ("v1bp", v1bp)):
    rr = [x["r"] for r in rows for x in r["ev"]["canon"] if x["k"] != "unrec"]
    hist = Counter()
    for v in rr:
        hist["<0.5" if v < .5 else "0.5-1" if v < 1 else "1-2" if v < 2 else "2-4" if v < 4
             else "4-8" if v < 8 else "8-16" if v < 16 else ">16"] += 1
    tot = len(rr)
    print(f"  {name}: " + "  ".join(f"{k}:{pct(hist[k],tot):.0f}%"
          for k in ("<0.5","0.5-1","1-2","2-4","4-8","8-16",">16")))

# ---------- Q2: recovered-but-FAILED forensics ----------
print("\n## Q2 — recovered episodes: who fails and why")
for name, rows in (("v1b", v1b), ("v1bp", v1bp)):
    rec = [r for r in rows if ep_class(r) == "rec"]
    print(f"\n### {name}: {len(rec)} recovered episodes, "
          f"{pct(sum(1 for r in rec if r['succ']),len(rec)):.1f}% succeed")
    print("| band | rec eps | share of pool | succ% |")
    for s in ("CCA", "ICA-mid", "siphon"):
        g = [r for r in rec if sect(r["pl"]) == s]
        print(f"| {s} | {len(g)} | {pct(len(g),len(rec)):.0f}% | "
              f"{pct(sum(1 for r in g if r['succ']),len(g)):.0f}% |")
    fail = [r for r in rec if not r["succ"]]
    print(f"FAILED recovered eps: {len(fail)}")
    print(f"  reasons: {Counter(r['reason'] for r in fail).most_common(5)}")
    print(f"  steps==600 (cap): {pct(sum(1 for r in fail if r['steps']>=600),len(fail)):.0f}%")
    print(f"  band mix: {Counter(sect(r['pl']) for r in fail).most_common()}")
    dt = sorted(r["d_tgt"] for r in fail if r.get("d_tgt") is not None)
    if dt:
        print(f"  final d_tgt mm: median {dt[len(dt)//2]:.0f}  p25 {dt[len(dt)//4]:.0f}  p75 {dt[3*len(dt)//4]:.0f}")
    # time budget: when did the LAST recovery happen relative to episode end?
    for lbl, g in (("failed", fail), ("succeeded", [r for r in rec if r["succ"]])):
        ls = [max(x["s"] for x in r["ev"]["canon"]) for r in g if r["ev"]["canon"]]
        st = [r["steps"] for r in g if r["ev"]["canon"]]
        if ls:
            fr = sorted(l/s for l, s in zip(ls, st))
            nev = sorted(len(r["ev"]["canon"]) for r in g)
            print(f"  {lbl}: median last-recovery at {100*fr[len(fr)//2]:.0f}% of episode, "
                  f"median events/ep {nev[len(nev)//2]}")

# same-band conversion: is the gap composition or genuine?
print("\n### matched-band conversion (recovered eps succ%, v1b vs v1bp)")
for s in ("CCA", "ICA-mid", "siphon"):
    out = []
    for name, rows in (("v1b", v1b), ("v1bp", v1bp)):
        g = [r for r in rows if ep_class(r) == "rec" and sect(r["pl"]) == s]
        out.append(f"{name} {pct(sum(1 for r in g if r['succ']),len(g)):.0f}% (n={len(g)})")
    print(f"  {s}: " + "  vs  ".join(out))

# v1bp per-bin: WHY do bins 3-5 convert at 54-68%?
print("\n### v1bp recovered pool per bin: composition vs conversion")
print("| bin | rec eps | succ% | %siphon in pool | %ICA-mid | failed@cap% | siphon-rec succ% |")
n8 = len(v1bp) // 8
for i in range(8):
    b = v1bp[i*n8:(i+1)*n8]
    rec = [r for r in b if ep_class(r) == "rec"]
    if not rec: continue
    sip = [r for r in rec if sect(r["pl"]) == "siphon"]
    ica = [r for r in rec if sect(r["pl"]) == "ICA-mid"]
    fail = [r for r in rec if not r["succ"]]
    print(f"| {i+1} | {len(rec)} | {pct(sum(1 for r in rec if r['succ']),len(rec)):.0f}% | "
          f"{pct(len(sip),len(rec)):.0f}% | {pct(len(ica),len(rec)):.0f}% | "
          f"{pct(sum(1 for r in fail if r['steps']>=600),len(fail)) if fail else 0:.0f}% | "
          f"{pct(sum(1 for r in sip if r['succ']),len(sip)) if sip else float('nan'):.0f}% |")

# ---------- Q3: independent recount of headline numbers ----------
print("\n## Q3 — independent recount (cross-check of T5)")
for name, rows in (("v1b", v1b), ("v1bp", v1bp)):
    kc = Counter(x["k"] for r in rows for x in r["ev"]["canon"])
    tot = sum(kc.values()); esc = tot - kc["unrec"]
    print(f"{name}: eps={len(rows)} succ={pct(sum(1 for r in rows if r['succ']),len(rows)):.1f}% "
          f"events={tot} esc={pct(esc,tot):.1f}% soft={pct(kc['soft'],tot):.1f}% "
          f"hard={pct(kc['hard'],tot):.1f}% grind={pct(kc['grind'],tot):.1f}% unrec={pct(kc['unrec'],tot):.1f}%")
