"""TASK 3 part 4 -- arrest vs budget-exhaustion from the 220-episode STEP log."""
import os, re, csv, glob, json
import numpy as np

RUN = ("D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/"
       "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
LOGD = os.path.join(RUN, "logs", "20260828_053306")
GEOM = json.load(open("D:/Arjun/workspace/neve/monitoring/task3_geom.json")) \
    if os.path.exists("D:/Arjun/workspace/neve/monitoring/task3_geom.json") else {}
OFFSET = 33.314

# ---- episodes.csv ground truth ----
csvrows = {}
with open(os.path.join(RUN, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        csvrows[int(r["seed"])] = dict(plen=float(r["path_len_mm"]), sec=r["section"],
                                       steps=int(r["steps"]), succ=int(r["success"]),
                                       anat=r["anatomy"])

num = r"[-+0-9.eE]+"
eps = {}          # (pid, ep) -> dict
for fp in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    cur = {}
    with open(fp, "r", errors="replace") as f:
        for ln in f:
            if "EPISODE_START" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                ep = re.search(r"ep=(\d+)", ln).group(1)
                sd = re.search(r"seed=(\d+)", ln)
                mf = re.search(r"mesh_fp=(\S+)", ln)
                k = (pid, ep)
                cur[pid] = k
                eps[k] = dict(seed=int(sd.group(1)) if sd else None,
                              mesh=mf.group(1).strip() if mf else None,
                              s=[], d=[], st=[], plen=None, outcome=None)
            elif "STEP |" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                k = cur.get(pid)
                if k is None:
                    continue
                e = eps[k]
                m = re.search(r"proj_s=(" + num + r")", ln)
                dt = re.search(r"d_tgt=(" + num + r")", ln)
                pl = re.search(r"path_len=(" + num + r")", ln)
                es = re.search(r"ep_step=(\d+)", ln)
                if m:
                    e["s"].append(float(m.group(1)))
                if dt:
                    e["d"].append(float(dt.group(1)))
                if es:
                    e["st"].append(int(es.group(1)))
                if pl and e["plen"] is None:
                    e["plen"] = float(pl.group(1))
            elif "EPISODE_OUTCOME" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                k = cur.get(pid)
                if k:
                    eps[k]["outcome"] = re.search(r"reason=(\S+)", ln).group(1)

print("parsed episodes: {}   with seed: {}   with >=1 STEP: {}"
      .format(len(eps), sum(1 for e in eps.values() if e["seed"]),
              sum(1 for e in eps.values() if e["s"])))

recs = []
for k, e in eps.items():
    if not e["seed"] or e["seed"] not in csvrows or not e["s"]:
        continue
    c = csvrows[e["seed"]]
    s = np.array(e["s"])
    d = np.array(e["d"]) if e["d"] else np.array([np.nan])
    n = len(s)
    smax = float(s.max())
    imax = int(np.argmax(s))
    tail = s[-100:] if n >= 100 else s
    prog100 = float(tail[-1] - tail[0])
    tail50 = s[-50:] if n >= 50 else s
    recs.append(dict(seed=e["seed"], anat=c["anat"], plen=c["plen"], sec=c["sec"],
                     succ=c["succ"], steps=c["steps"], nlog=n,
                     smax=smax, sfin=float(s[-1]),
                     deficit=c["plen"] - smax,
                     frac_at_max=imax / max(n - 1, 1),
                     prog100=prog100,
                     prog50=float(tail50[-1] - tail50[0]),
                     dmin=float(np.nanmin(d)), dfin=float(d[-1]),
                     dft=(GEOM.get("topcow_mr_" + c["anat"].replace("topcowmr", ""), {})
                          .get("L", np.nan) - (c["plen"] - OFFSET))))

print("joined to csv: {} of 220".format(len(recs)))
F = [r for r in recs if r["succ"] == 0]
S = [r for r in recs if r["succ"] == 1]
print("  failures {}  successes {}".format(len(F), len(S)))


def band(p):
    if p < 167:
        return "<167"
    if p < 200:
        return "167-200"
    if p < 240:
        return "200-240"
    return ">=240"


print()
print("A. FAILURES -- how far did they get, and were they still moving at the cap?")
print("{:>9} {:>4} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8}".format(
    "band", "n", "med_def", "p25_def", "p75_def", "med_%rt", "med_p100", "n_prog<1", "med_dmin", "med_fmax"))
for b in ["<167", "167-200", "200-240", ">=240"]:
    sub = [r for r in F if band(r["plen"]) == b]
    if not sub:
        print("{:>9} {:>4}  (none)".format(b, 0))
        continue
    de = np.array([r["deficit"] for r in sub])
    rt = np.array([100.0 * r["smax"] / r["plen"] for r in sub])
    p1 = np.array([r["prog100"] for r in sub])
    dm = np.array([r["dmin"] for r in sub])
    fm = np.array([r["frac_at_max"] for r in sub])
    print("{:>9} {:4d} | {:8.1f} {:8.1f} {:8.1f} {:8.1f} | {:8.2f} {:8d} {:8.2f} {:8.2f}".format(
        b, len(sub), np.median(de), np.percentile(de, 25), np.percentile(de, 75),
        np.median(rt), np.median(p1), int((np.abs(p1) < 1.0).sum()), np.median(dm), np.median(fm)))

print()
print("B. >=240 FAILURES, one line each  (deficit = target proj_s not covered; "
      "prog100 = proj_s gained in last 100 steps; frac_at_max = position in episode of deepest reach)")
print("{:>12} {:>7} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}".format(
    "anat", "plen", "dft", "smax", "deficit", "reach%", "prog100", "frac@max", "dmin"))
for r in sorted([r for r in F if r["plen"] >= 240], key=lambda r: -r["deficit"]):
    print("{:>12} {:7.1f} {:6.2f} {:7.1f} {:8.1f} {:8.1f} {:8.2f} {:8.2f} {:7.1f}".format(
        r["anat"], r["plen"], r["dft"], r["smax"], r["deficit"],
        100.0 * r["smax"] / r["plen"], r["prog100"], r["frac_at_max"], r["dmin"]))

print()
print("C. SUCCESSES -- steps needed vs path_len (is the 600 cap binding?)")
print("{:>9} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
    "band", "n", "med", "p75", "p90", "p95", "max", "n>=500"))
for b in ["<167", "167-200", "200-240", ">=240"]:
    sub = np.array([r["steps"] for r in S if band(r["plen"]) == b])
    if not len(sub):
        continue
    print("{:>9} {:4d} {:7.0f} {:7.0f} {:7.0f} {:7.0f} {:7.0f} {:7d}".format(
        b, len(sub), np.median(sub), np.percentile(sub, 75), np.percentile(sub, 90),
        np.percentile(sub, 95), sub.max(), int((sub >= 500).sum())))

# survival: among episodes that reached step t, hazard of success
print()
print("D. KM-style survival on step budget (censoring = 600-step truncation), per band")
print("   S(t) = prob still running at step t ; success events only")
for b in ["167-200", "200-240", ">=240"]:
    sub = [r for r in recs if band(r["plen"]) == b]
    n = len(sub)
    ts = sorted(r["steps"] for r in sub if r["succ"] == 1)
    surv = 1.0
    atrisk = n
    marks = []
    prev = 0
    for t in ts:
        # everyone with steps>=t still at risk
        atrisk = sum(1 for r in sub if r["steps"] >= t)
        surv *= (1 - 1.0 / atrisk)
        marks.append((t, 1 - surv, atrisk))
    fin = marks[-1] if marks else (0, 0, 0)
    print("   {:>8} n={:3d}  raw succ {:.3f}   KM cum-incidence at 600 = {:.3f}  "
          "at-risk just before last event = {}"
          .format(b, n, sum(r["succ"] for r in sub) / n, fin[1], fin[2]))
    # incidence in step windows
    win = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]
    row = []
    for a, z in win:
        atr = sum(1 for r in sub if r["steps"] >= a)
        ev = sum(1 for r in sub if r["succ"] == 1 and a <= r["steps"] < z)
        row.append("{}-{}: {}/{}".format(a, z, ev, atr))
    print("        window hazards: " + "  ".join(row))

print()
print("E. LATE SUCCESSES: successes finishing after step 400, by band + dft")
for r in sorted([r for r in S if r["steps"] >= 400], key=lambda r: r["steps"]):
    print("   {:>12} plen={:6.1f} dft={:6.2f} steps={:4d} band={}".format(
        r["anat"], r["plen"], r["dft"], r["steps"], band(r["plen"])))

print()
print("F. proj_s-per-step throughput on the DEEP leg (successes vs failures)")
for b in ["167-200", "200-240", ">=240"]:
    su = [r for r in S if band(r["plen"]) == b]
    fa = [r for r in F if band(r["plen"]) == b]
    def thr(rs):
        v = [r["smax"] / r["nlog"] for r in rs if r["nlog"] > 0]
        return np.median(v) if v else float("nan")
    print("   {:>8}  succ med mm/step {:.3f} (n={})   fail med mm/step {:.3f} (n={})"
          .format(b, thr(su), len(su), thr(fa), len(fa)))

print()
print("G. FAILURE arrest classification (>=240 band)")
sub = [r for r in F if r["plen"] >= 240]
arrested = [r for r in sub if abs(r["prog100"]) < 1.0]
moving = [r for r in sub if abs(r["prog100"]) >= 1.0]
print("   n={}  ARRESTED (proj_s moved <1mm over last 100 steps) = {}   STILL MOVING = {}"
      .format(len(sub), len(arrested), len(moving)))
print("   arrested: median deficit {:.1f} mm, median reach {:.1f}%".format(
    np.median([r["deficit"] for r in arrested]) if arrested else float("nan"),
    np.median([100 * r["smax"] / r["plen"] for r in arrested]) if arrested else float("nan")))
if moving:
    print("   moving  : median deficit {:.1f} mm, median reach {:.1f}%, median prog100 {:.2f}".format(
        np.median([r["deficit"] for r in moving]),
        np.median([100 * r["smax"] / r["plen"] for r in moving]),
        np.median([r["prog100"] for r in moving])))

print()
print("H. same classification, ALL failures by band")
for b in ["167-200", "200-240", ">=240"]:
    sub = [r for r in F if band(r["plen"]) == b]
    if not sub:
        continue
    a = sum(1 for r in sub if abs(r["prog100"]) < 1.0)
    print("   {:>8} n={:3d}  arrested {:3d} ({:.0f}%)  moving {:3d}".format(
        b, len(sub), a, 100.0 * a / len(sub), len(sub) - a))

print()
print("I. deficit vs dft -- do the failures die AT the cap region or well short of it?")
sub = [r for r in F if r["plen"] >= 240]
near = [r for r in sub if r["dft"] < 8]
far = [r for r in sub if r["dft"] >= 8]
for nm, g in [("target within 8mm of terminus", near), ("target >=8mm from terminus", far)]:
    if g:
        print("   {:>32}: n={:2d} med deficit {:6.1f} mm  med reach {:5.1f}%  med dmin {:5.1f} mm"
              .format(nm, len(g), np.median([r["deficit"] for r in g]),
                      np.median([100 * r["smax"] / r["plen"] for r in g]),
                      np.median([r["dmin"] for r in g])))
