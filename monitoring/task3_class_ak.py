"""TASK 3 final -- classify all 55 failures; re-score with each mode removed."""
import os, re, csv, glob, json
import numpy as np

RUN = ("D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/"
       "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
LOGD = os.path.join(RUN, "logs", "20260828_053306")
OFFSET = 33.314
L = {"topcowmr" + k: v for k, v in json.load(open(
    "D:/Arjun/workspace/neve/monitoring/task3_geom.json")).items()}
L = {k.replace("topcow_mr_", ""): v["L"] for k, v in json.load(open(
    "D:/Arjun/workspace/neve/monitoring/task3_geom.json")).items()}

csvrows = {}
with open(os.path.join(RUN, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        csvrows[int(r["seed"])] = dict(plen=float(r["path_len_mm"]), sec=r["section"],
                                       steps=int(r["steps"]), succ=int(r["success"]),
                                       anat=r["anatomy"])

eps = {}
for fp in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    cur = {}
    with open(fp, "r", errors="replace") as f:
        for ln in f:
            if "EPISODE_START" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                sd = re.search(r"seed=(\d+)", ln)
                cur[pid] = int(sd.group(1)) if sd else None
                if cur[pid] is not None:
                    eps[cur[pid]] = dict(s=[], d=[], f=[], ic=[], iw=[], tol=[], lr=[], br=[])
            elif "STEP |" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                k = cur.get(pid)
                if k is None:
                    continue
                e = eps[k]
                g = lambda p: re.search(p, ln)
                m = g(r"proj_s=([-+0-9.eE]+)")
                if m: e["s"].append(float(m.group(1)))
                m = g(r"d_tgt=([-+0-9.eE]+)")
                if m: e["d"].append(float(m.group(1)))
                m = g(r"fold=(\d+)/(\d+)")
                if m: e["f"].append(int(m.group(1)))
                m = g(r"inserted=\[([-+0-9.eE]+),([-+0-9.eE]+)\]")
                if m:
                    e["ic"].append(float(m.group(1))); e["iw"].append(float(m.group(2)))
                m = g(r"tol=([-+0-9.eE]+)")
                if m: e["tol"].append(float(m.group(1)))
                m = g(r"local_r=([-+0-9.eE]+)")
                if m: e["lr"].append(float(m.group(1)))
                m = g(r"cur_branch=([^|]+)")
                if m: e["br"].append(m.group(1).strip())

R = []
for sd, e in eps.items():
    c = csvrows.get(sd)
    if not c or not e["s"]:
        continue
    s = np.array(e["s"]); d = np.array(e["d"]); f = np.array(e["f"]) if e["f"] else np.array([0])
    ic = np.array(e["ic"]); iw = np.array(e["iw"])
    n = len(s)
    t100 = slice(max(n - 100, 0), n)
    prog = float(s[-1] - s[t100.start])
    ins_gain = float(max(ic[-1] - ic[t100.start], iw[-1] - iw[t100.start])) if len(ic) else 0.0
    R.append(dict(seed=sd, **c, n=n, smax=float(s.max()), sfin=float(s[-1]),
                  reach=100.0 * s.max() / c["plen"], dmin=float(d.min()), dfin=float(d[-1]),
                  foldmax=int(f.max()), foldfin=int(f[-1]),
                  prog100=prog, insgain100=ins_gain,
                  tolfin=float(e["tol"][-1]) if e["tol"] else np.nan,
                  insfin=float(max(ic[-1], iw[-1])) if len(ic) else np.nan,
                  srcca=float(s.max()) - OFFSET,
                  dft=L[c["anat"].replace("topcowmr", "")] - (c["plen"] - OFFSET)))

F = [r for r in R if r["succ"] == 0]
print("episodes {}  failures {}".format(len(R), len(F)))


def cls(r):
    if r["reach"] < 30:
        return "P_ostium_stall"
    if r["foldmax"] > 20 and r["prog100"] < 5:
        return "B_buckle_arrest"
    if r["prog100"] < 1.0:
        return "A_arrest_other"
    return "T_timeout_advancing"


for r in F:
    r["mode"] = cls(r)

print()
print("FAILURE MODE TABLE  (55 failures)")
print("{:>21} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "mode", "n", "med_plen", "med_rch%", "med_smax", "med_srcca", "med_dmin", "med_fold", "med_dft"))
for m in ["P_ostium_stall", "B_buckle_arrest", "A_arrest_other", "T_timeout_advancing"]:
    g = [r for r in F if r["mode"] == m]
    if not g:
        continue
    q = lambda k: np.median([r[k] for r in g])
    print("{:>21} {:5d} {:8.1f} {:8.1f} {:8.1f} {:9.1f} {:8.1f} {:8.0f} {:8.1f}".format(
        m, len(g), q("plen"), q("reach"), q("smax"), q("srcca"), q("dmin"), q("foldmax"), q("dft")))

print()
print("FAILURE MODE x PATH_LEN BAND")
bands = [("<167", 0, 167), ("167-200", 167, 200), ("200-240", 200, 240), (">=240", 240, 999)]
modes = ["P_ostium_stall", "B_buckle_arrest", "A_arrest_other", "T_timeout_advancing"]
print("{:>9} {:>4} {:>5} | {}".format("band", "n", "fail", "  ".join("{:>19}".format(m) for m in modes)))
for nm, lo, hi in bands:
    sub = [r for r in R if lo <= r["plen"] < hi]
    fs = [r for r in sub if r["succ"] == 0]
    cells = ["{:>19}".format(sum(1 for r in fs if r["mode"] == m)) for m in modes]
    print("{:>9} {:4d} {:5d} | {}".format(nm, len(sub), len(fs), "  ".join(cells)))

print()
print("WHERE DO THE ARRESTS SIT IN s_RCCA?  (seam=133.6, decl-radii divergence=102.5)")
for m in modes:
    g = sorted(r["srcca"] for r in F if r["mode"] == m)
    if g:
        print("  {:>21} n={:2d}  s_RCCA of deepest reach: {}".format(
            m, len(g), " ".join("{:.0f}".format(v) for v in g)))

print()
print("NEAR MISSES: failures whose min d_tgt was within 2x final tol")
nm = [r for r in F if r["dmin"] <= 2 * r["tolfin"]]
print("  n={} of 55".format(len(nm)))
for r in sorted(nm, key=lambda r: r["dmin"]):
    print("   {:>12} plen={:6.1f} dmin={:5.1f} tol={:5.1f} fold_max={:3d} mode={}".format(
        r["anat"], r["plen"], r["dmin"], r["tolfin"], r["foldmax"], r["mode"]))

print()
print("BUCKLING: final inserted length vs proj_s reached")
for m in modes:
    g = [r for r in F if r["mode"] == m]
    if g:
        print("  {:>21} med final inserted {:6.1f} mm for med proj_s {:6.1f} mm  (ratio {:.2f}); "
              "med insertion gain in last 100 steps {:6.1f} mm".format(
                  m, np.median([r["insfin"] for r in g]), np.median([r["sfin"] for r in g]),
                  np.median([r["insfin"] for r in g]) / max(np.median([r["sfin"] for r in g]), 1e-9),
                  np.median([r["insgain100"] for r in g])))
g = [r for r in R if r["succ"] == 1]
print("  {:>21} med final inserted {:6.1f} mm for med proj_s {:6.1f} mm".format(
    "SUCCESSES", np.median([r["insfin"] for r in g]), np.median([r["sfin"] for r in g])))

print()
print("COUNTERFACTUAL RE-SCORES on the 220-episode run")
tot = len(R)


def rate(sub):
    return "{:3d}/{:3d}={:5.1f}%".format(sum(r["succ"] for r in sub), len(sub),
                                         100.0 * sum(r["succ"] for r in sub) / max(len(sub), 1))


def bandsub(sub, lo, hi):
    return [r for r in sub if lo <= r["plen"] < hi]


scen = [
    ("as measured", R),
    ("drop targets within 12mm of terminus", [r for r in R if r["dft"] >= 12]),
    ("drop mr_024/025/027 (geom defects)", [r for r in R if r["anat"] not in
                                            ("topcowmr024", "topcowmr025", "topcowmr027")]),
    ("credit P_ostium_stall as non-distal", None),
    ("credit all >20 fold buckles", None),
]
print("{:>40} {:>18} {:>18} {:>18} {:>18}".format("scenario", "OVERALL", "167-200", "200-240", ">=240"))
for nm2, sub in scen:
    if sub is None:
        continue
    print("{:>40} {:>18} {:>18} {:>18} {:>18}".format(
        nm2, rate(sub), rate(bandsub(sub, 167, 200)), rate(bandsub(sub, 200, 240)),
        rate(bandsub(sub, 240, 999))))
# treat specific modes as if they were solved (upper bound on that mode's cost)
for mode in modes:
    sub = [dict(r, succ=1 if (r["succ"] == 0 and r["mode"] == mode) else r["succ"]) for r in R]
    print("{:>40} {:>18} {:>18} {:>18} {:>18}".format(
        "if " + mode + " were fixed", rate(sub), rate(bandsub(sub, 167, 200)),
        rate(bandsub(sub, 200, 240)), rate(bandsub(sub, 240, 999))))

print()
print("BAND COMPOSITION (which anatomies populate 167-200 vs 200-240)")
for nm2, lo, hi in bands[1:]:
    sub = bandsub(R, lo, hi)
    from collections import Counter
    c = Counter(r["anat"] for r in sub)
    cs = Counter(r["anat"] for r in sub if r["succ"] == 1)
    print("  {:>8} n={:3d}: {}".format(nm2, len(sub), " ".join(
        "{}={}/{}".format(a.replace("topcowmr", ""), cs[a], c[a]) for a in sorted(c))))
