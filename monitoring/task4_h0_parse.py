import os, re, json, glob, collections

RUN = r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke"
LOGD = os.path.join(RUN, "logs_subprocesses")
SNAP = os.path.join(RUN, "diagnostics/snapshots/eval/RCCA")

# 1) ground-truth outcomes from snapshot filenames
snap_out = {}
for cls in ("success", "max_steps"):
    for f in os.listdir(os.path.join(SNAP, cls)):
        m = re.match(r"ep(\d+)_pid(\d+)_step(\d+)_R([-+][\d.]+)_", f)
        assert m, f
        key = (int(m.group(2)), int(m.group(1)))
        snap_out[key] = dict(cls=cls, steps=int(m.group(3)), R=float(m.group(4)), fname=f)
print("snapshots:", collections.Counter(v['cls'] for v in snap_out.values()), "n=", len(snap_out))

# 2) parse eval episodes from worker logs
def kv(line):
    d = {}
    for part in line.split(" | ")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d

eps = {}   # (pid,ep) -> dict
steps = collections.defaultdict(list)
for wf in sorted(glob.glob(os.path.join(LOGD, "worker_*.log"))):
    with open(wf, "r", errors="replace") as fh:
        for line in fh:
            if "step_logger_eval_" not in line:
                continue
            if "EPISODE_START" in line:
                d = kv(line[line.index("EPISODE_START"):])
                if "seed" not in d:
                    continue
                key = (int(d["pid"]), int(d["ep"]))
                eps[key] = dict(start=d, outcome=None)
            elif "EPISODE_OUTCOME" in line:
                d = kv(line[line.index("EPISODE_OUTCOME"):])
                key = (int(d["pid"]), int(d["ep"]))
                if key in eps:
                    eps[key]["outcome"] = d
            elif " - STEP | " in line:
                d = kv(line[line.index("STEP |"):])
                key = (int(d["pid"]), int(d["ep"]))
                steps[key].append(d)

print("eval episodes parsed:", len(eps))
print("with outcome:", sum(1 for v in eps.values() if v["outcome"]))
print("keys match snapshots:", set(eps) == set(snap_out))
print("step-line episodes:", len(steps))
missing = [k for k in eps if k not in steps]
print("episodes w/o step lines:", missing)
for k in []:
    if len(steps.get(k, [])) == 0:
        print("EMPTY", k)
import pickle
with open(r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/h0.pkl", "wb") as fh:
    pickle.dump(dict(eps=eps, steps=dict(steps), snap=snap_out), fh)
