"""INDEPENDENT verification of the workflow's central claim.

CLAIM: on the real-patient anatomy every deep failure arrests at proj_s = 153.4 mm,
identically for v1b, v1bp AND the hand-written heuristic H0 — i.e. an ENVIRONMENTAL
wall, not a policy failure. Corollaries: max solved path_len 156.2 / min failed 164.2
(clean cut), and the wall sits BEFORE the siphon band (>=210 mm).

Method: parse each real-patient eval's worker logs directly; per episode record
max proj_s, success, path_len, final tip3d. Report the arrest distribution.
"""
import glob, os, re, sys
from collections import Counter

PROJ = re.compile(r"proj_s=([-0-9.]+)")
PL = re.compile(r"path_len=([0-9.]+)")
SEED = re.compile(r"seed=(\d+)")
TIP = re.compile(r"tip3d=\[([-0-9.]+),([-0-9.]+),([-0-9.]+)\]")


def parse(log_dir):
    eps = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        live = {}
        for line in open(path, errors="replace"):
            i = line.find("pid=")
            pid = line[i + 4:].split(" ")[0].strip() if i >= 0 else "?"
            if "EPISODE_START" in line:
                m = SEED.search(line)
                live[pid] = {"seed": int(m.group(1)) if m else None,
                             "maxp": -1e9, "pl": None, "succ": False, "tip": None}
                continue
            st = live.get(pid)
            if st is None:
                continue
            if "EPISODE_OUTCOME" in line:
                j = line.find("reason=")
                if j >= 0 and line[j + 7:].split(" ")[0].strip() == "success":
                    st["succ"] = True
                if st["seed"] is not None and st["maxp"] > -1e8:
                    eps[st["seed"]] = st
                live.pop(pid, None)
                continue
            if " STEP |" in line:
                m = PROJ.search(line)
                if m:
                    v = float(m.group(1))
                    if v > st["maxp"]:
                        st["maxp"] = v
                        mt = TIP.search(line)
                        if mt:
                            st["tip"] = tuple(round(float(g), 1) for g in mt.groups())
                if st["pl"] is None:
                    m = PL.search(line)
                    if m:
                        st["pl"] = float(m.group(1))
        for st in live.values():
            if st["seed"] is not None and st["maxp"] > -1e8:
                eps.setdefault(st["seed"], st)
    return eps


def report(tag, eps):
    if not eps:
        print(f"{tag}: NO EPISODES PARSED")
        return None
    succ = [e for e in eps.values() if e["succ"]]
    fail = [e for e in eps.values() if not e["succ"]]
    fm = sorted(e["maxp"] for e in fail)
    print(f"\n=== {tag} ===   n={len(eps)}  succ={len(succ)}  fail={len(fail)}")
    if fm:
        print(f"  FAILED max-proj_s : min {fm[0]:.1f}  p25 {fm[len(fm)//4]:.1f}  "
              f"MEDIAN {fm[len(fm)//2]:.1f}  p75 {fm[3*len(fm)//4]:.1f}  max {fm[-1]:.1f}")
        band = Counter(round(v, 1) for v in fm)
        print(f"  most common arrest values: {band.most_common(6)}")
        n153 = sum(1 for v in fm if 152.5 <= v <= 154.5)
        print(f"  in [152.5,154.5] mm : {n153}/{len(fm)} = {100*n153/len(fm):.0f}%")
        sd = (sum((v - sum(fm)/len(fm))**2 for v in fm)/len(fm))**.5
        print(f"  SD of failed arrest : {sd:.2f} mm")
        tips = Counter(e["tip"] for e in fail if e["tip"])
        print(f"  most common arrest tip3d: {tips.most_common(3)}")
    if succ:
        sp = sorted(e["pl"] for e in succ if e["pl"] is not None)
        fp = sorted(e["pl"] for e in fail if e["pl"] is not None)
        if sp and fp:
            print(f"  path_len  MAX SOLVED {sp[-1]:.1f}   MIN FAILED {fp[0]:.1f}   "
                  f"=> {'CLEAN CUT' if sp[-1] < fp[0] else 'OVERLAP (no clean cut)'}")
    return fm


for tag, d in [a.split("=", 1) for a in sys.argv[1:]]:
    report(tag, parse(d))
