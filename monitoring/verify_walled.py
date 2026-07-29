"""Test the decisive corollary: are the 50 GENERATED anatomies individually 'walled'?

Pooled arrest on the 50-anatomy eval scatters (SD 34 mm), which does NOT by itself
refute a per-anatomy wall — each anatomy could arrest deterministically at its OWN
station. This groups failures by the anatomy branch-hash logged in EPISODE_START and
measures WITHIN-anatomy arrest spread.

walled(anatomy) := >=2 deep failures AND within-anatomy SD of max-proj_s < 2 mm
"""
import glob, os, re, sys
from collections import defaultdict

PROJ = re.compile(r"proj_s=([-0-9.]+)")
PL = re.compile(r"path_len=([0-9.]+)")
ANA = re.compile(r"anatomy=([a-f0-9]+)")


def parse(log_dir):
    eps = []
    for path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        live = {}
        for line in open(path, errors="replace"):
            i = line.find("pid=")
            pid = line[i + 4:].split(" ")[0].strip() if i >= 0 else "?"
            if "EPISODE_START" in line:
                m = ANA.search(line)
                live[pid] = {"ana": m.group(1) if m else None, "maxp": -1e9,
                             "pl": None, "succ": False}
                continue
            st = live.get(pid)
            if st is None:
                continue
            if "EPISODE_OUTCOME" in line:
                j = line.find("reason=")
                if j >= 0 and line[j + 7:].split(" ")[0].strip() == "success":
                    st["succ"] = True
                if st["maxp"] > -1e8 and st["ana"]:
                    eps.append(st)
                live.pop(pid, None)
                continue
            if " STEP |" in line:
                m = PROJ.search(line)
                if m:
                    st["maxp"] = max(st["maxp"], float(m.group(1)))
                if st["pl"] is None:
                    m = PL.search(line)
                    if m:
                        st["pl"] = float(m.group(1))
        for st in live.values():
            if st["maxp"] > -1e8 and st["ana"]:
                eps.append(st)
    return eps


def sd(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** .5


for arg in sys.argv[1:]:
    tag, d = arg.split("=", 1)
    eps = parse(d)
    by = defaultdict(list)
    for e in eps:
        by[e["ana"]].append(e)
    print(f"\n=== {tag} ===  episodes {len(eps)}  anatomies {len(by)}")
    walled, unwalled, singles = [], [], 0
    for a, g in by.items():
        f = [e["maxp"] for e in g if not e["succ"]]
        if len(f) < 2:
            singles += 1
            continue
        (walled if sd(f) < 2.0 else unwalled).append((a, len(f), sum(f) / len(f), sd(f)))
    n = len(walled) + len(unwalled)
    print(f"  anatomies with >=2 deep failures: {n}   (excluded, <2 failures: {singles})")
    if n:
        print(f"  WALLED (within-anatomy SD < 2mm): {len(walled)}/{n} = {100*len(walled)/n:.0f}%")
        print(f"  scattered                       : {len(unwalled)}/{n}")
        if walled:
            ws = sorted(w[2] for w in walled)
            print(f"  walled arrest stations: min {ws[0]:.1f}  median {ws[len(ws)//2]:.1f}  max {ws[-1]:.1f}")
            print(f"  examples: " + "; ".join(f"{a[:6]} n={k} @{m:.1f}±{s:.2f}" for a, k, m, s in walled[:5]))
        if unwalled:
            us = sorted(w[3] for w in unwalled)
            print(f"  scattered within-anatomy SD: median {us[len(us)//2]:.1f} mm  max {us[-1]:.1f} mm")
    tot_f = sum(1 for e in eps if not e["succ"])
    wf = sum(w[1] for w in walled)
    print(f"  failure mass in walled anatomies: {wf}/{tot_f} = {100*wf/tot_f:.0f}% of all failures")
