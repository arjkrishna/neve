"""Trace the proj_s~45mm arrest cluster and the 'still moving' cluster."""
import os, re, csv, glob, json
import numpy as np

RUN = ("D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/"
       "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
LOGD = os.path.join(RUN, "logs", "20260828_053306")
num = r"[-+0-9.eE]+"

csvrows = {}
with open(os.path.join(RUN, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        csvrows[int(r["seed"])] = dict(plen=float(r["path_len_mm"]), sec=r["section"],
                                       steps=int(r["steps"]), succ=int(r["success"]),
                                       anat=r["anatomy"])

FIELDS = ["proj_s", "d_tgt", "inserted", "cur_branch", "on_br", "on_path", "off_br",
          "local_r", "tol", "xt_true", "d_corr_arc", "arc_past", "fold", "cath_slack",
          "nearest_named", "entries_passed", "daughters_passed", "overshoot", "phys"]

eps = {}
for fp in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    cur = {}
    with open(fp, "r", errors="replace") as f:
        for ln in f:
            if "EPISODE_START" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                sd = re.search(r"seed=(\d+)", ln)
                k = int(sd.group(1)) if sd else None
                cur[pid] = k
                if k is not None:
                    eps[k] = []
            elif "STEP |" in ln:
                pid = re.search(r"pid=(\d+)", ln).group(1)
                k = cur.get(pid)
                if k is None:
                    continue
                d = {}
                for fld in FIELDS:
                    m = re.search(re.escape(fld) + r"=([^|]+)", ln)
                    if m:
                        d[fld] = m.group(1).strip()
                eps[k].append(d)

# pick episodes: arrest cluster (smax ~45) and moving cluster
targets = []
for sd, tr in eps.items():
    c = csvrows.get(sd)
    if not c or c["succ"] or c["plen"] < 240:
        continue
    s = np.array([float(x.get("proj_s", "nan")) for x in tr])
    targets.append((sd, c, tr, float(np.nanmax(s))))

targets.sort(key=lambda t: t[3])
print("### ALL >=240 failures, seed / anat / plen / smax")
for sd, c, tr, sm in targets:
    print("  seed={} {:>12} plen={:6.1f} smax={:6.1f} nsteps={}".format(sd, c["anat"], c["plen"], sm, len(tr)))

print()
for sd, c, tr, sm in targets[:3] + targets[-3:]:
    print("=" * 130)
    print("TRACE seed={} {} plen={:.1f} smax={:.1f} steps={}".format(sd, c["anat"], c["plen"], sm, len(tr)))
    idx = list(range(0, len(tr), max(len(tr) // 30, 1)))
    if idx[-1] != len(tr) - 1:
        idx.append(len(tr) - 1)
    print("{:>5} {:>8} {:>8} {:>14} {:>7} {:>6} {:>6} {:>6} {:>7} {:>7} {:>26}".format(
        "step", "proj_s", "d_tgt", "inserted", "on_br", "on_pth", "off_br", "fold", "xt", "d_arc", "cur_branch"))
    for i in idx:
        x = tr[i]
        print("{:>5} {:>8} {:>8} {:>14} {:>7} {:>6} {:>6} {:>6} {:>7} {:>7} {:>26}".format(
            i + 1, x.get("proj_s", "-"), x.get("d_tgt", "-"), x.get("inserted", "-"),
            x.get("on_br", "-"), x.get("on_path", "-"), x.get("off_br", "-"),
            x.get("fold", "-"), x.get("xt_true", "-"), x.get("d_corr_arc", "-"),
            str(x.get("cur_branch", "-"))[:26]))

# how much wire is inserted at the end of each failure vs each success
print()
print("### INSERTED length at final step  (device length exhaustion check)")
def fin_ins(tr):
    for x in reversed(tr):
        if "inserted" in x:
            v = x["inserted"].strip("[]").split(",")
            return float(v[0]), float(v[1])
    return float("nan"), float("nan")

rows = []
for sd, tr in eps.items():
    c = csvrows.get(sd)
    if not c or not tr:
        continue
    a, b = fin_ins(tr)
    s = np.array([float(x.get("proj_s", "nan")) for x in tr])
    rows.append((c, float(np.nanmax(s)), a, b))
for lo, hi, nm in [(0, 167, "<167"), (167, 200, "167-200"), (200, 240, "200-240"), (240, 999, ">=240")]:
    for succ in [1, 0]:
        g = [r for r in rows if lo <= r[0]["plen"] < hi and r[0]["succ"] == succ]
        if not g:
            continue
        cath = np.array([r[2] for r in g])
        wire = np.array([r[3] for r in g])
        print("  {:>8} succ={} n={:3d}  final inserted cath med {:6.1f} max {:6.1f} | wire med {:6.1f} max {:6.1f}"
              .format(nm, succ, len(g), np.median(cath), cath.max(), np.median(wire), wire.max()))
